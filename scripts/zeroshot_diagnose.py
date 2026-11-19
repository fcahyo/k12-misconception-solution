"""Zero-shot root-misconception diagnosis with a vision-language model served by vLLM (teachers and listwise rerankers).

Per row the prompt holds the full taxonomy (or, with --cands, only the row's candidate codes with their definitions, convention
rules and annotated training examples), the question with its diagram, the reference solution, the reader's transcription and/or
the photo of the working. The model ends with a line 'ROOT: C1, C2, ...'; outputs that hit the token budget get a short forced
second pass that asks for that line. Writes <out>/shard_NNNN.csv (trace_id, top3, ranked, tail, answer, forced, out_tokens),
resumable shard by shard, and prints MAP@3 on the training split.
"""
import argparse
import sys
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from k12.data import DATA, image_url, image_path, labels, load, question_image_path
from k12.metric import map3_lists
from k12.prompt import SYSTEM, question_block, taxonomy_block

GUIDE = (
    "Diagnose the student's working.\n"
    "1. Work out the correct solution path.\n"
    "2. Read the student's steps and find every error; decide which errors are MAJOR (they change the "
    "reasoning or the result) and which are minor (notation, a slip that does not change the result).\n"
    "3. The ROOT is the major error that the rest of the working follows from (usually the earliest major one).\n"
    "4. If there is no major error, the root is NONE.\n"
    "Finish with exactly one line: 'ROOT: <code1>, <code2>, <code3>' — your three most likely root codes, most likely first."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--input", default="image", help="image (photo of the working) | csv path (trace_id,transcription) for text only")
    p.add_argument("--solutions", default=None, help="csv(question_id, solution) of reference solutions")
    p.add_argument("--rows", default=None, help="optional csv of trace_ids to restrict to")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--quant", default=None)
    p.add_argument("--think", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--gpu-mem", type=float, default=0.92)
    p.add_argument("--kv-fp8", type=int, default=1)
    p.add_argument("--trans-csv", default=None, help="reader transcriptions added as text (with --input image)")
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--force-rc", type=int, default=1,
                   help="budget forcing: 1 = pass the kept reasoning as the assistant 'reasoning' field; "
                        "0 = keep it inside the message content")
    p.add_argument("--reasoning-effort", default=None, help="chat-template reasoning_effort (Qwen3.8-Flash-Next: xhigh | medium | low)")
    p.add_argument("--temperature", type=float, default=None, help="sampling temperature (default: 0.6 thinking / 0.0 otherwise)")
    p.add_argument("--cands", default=None, help="listwise reranker: csv(trace_id, codes) -> the prompt lists only these codes (rerank_prompt.py)")
    p.add_argument("--cand-examples", type=int, default=1, help="annotated training examples per non-NONE candidate")
    p.add_argument("--cand-rules", type=int, default=1, help="attach the convention rules that mention each candidate")
    p.add_argument("--cand-note-chars", type=int, default=160, help="characters of the annotator note shown per candidate example")
    p.add_argument("--save-text", type=int, default=0, help="1 = also write the full output text per row to shard_NNNN.jsonl")
    p.add_argument("--top-k", type=int, default=None, help="top-k for sampling (default: off)")
    a = p.parse_args()
    from vllm import LLM, SamplingParams

    df = load(a.split)
    if a.rows:
        keep = set(pd.read_csv(a.rows).trace_id)
        df = df[df.trace_id.isin(keep)]
    df = df.reset_index(drop=True)
    sol = dict(pd.read_csv(a.solutions, keep_default_na=False)[["question_id", "solution"]].values) if a.solutions else {}
    trans = None
    if a.trans_csv:
        t = pd.read_csv(a.trans_csv, keep_default_na=False)
        trans = dict(zip(t.trace_id, t.transcription))
    if a.input != "image":
        t = pd.read_csv(a.input, keep_default_na=False)
        trans = dict(zip(t.trace_id, t.transcription))
    L = set(labels())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs, max_model_len=a.max_model_len,
              limit_mm_per_prompt={"image": 2, "video": 0}, allowed_local_media_path=str(DATA), trust_remote_code=False,
              **({"quantization": a.quant} if a.quant else {}), **({"kv_cache_dtype": "fp8"} if a.kv_fp8 else {}))
    ctk = {"enable_thinking": bool(a.think), **({"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {})}
    temp = a.temperature if a.temperature is not None else (0.6 if a.think else 0.0)
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed,
                        **({"top_k": a.top_k} if a.top_k else {}))
    guide = GUIDE if a.think else GUIDE.replace("Finish with", "Write a brief analysis (at most 8 short lines), then finish with")
    cand_block = None
    if a.cands:
        sys.path.insert(0, "scripts")
        from rerank_prompt import RR_GUIDE, make_cand_block
        cand_block, cand_map = make_cand_block(a.split, a.cands, a.cand_examples, bool(a.cand_rules), note_chars=a.cand_note_chars)
        df = df[df.trace_id.isin(set(cand_map))].reset_index(drop=True)
    for s0 in range(0, len(df), a.shard_size):
        fn = out / f"shard_{s0 // a.shard_size:04d}.csv"
        if fn.exists():
            continue
        sub = df.iloc[s0:s0 + a.shard_size]
        convs = []
        for _, r in sub.iterrows():
            if cand_block is not None:
                c = [{"type": "text", "text": cand_block(r) + "\n\n" + question_block(r)}]
            else:
                c = [{"type": "text", "text": taxonomy_block("full") + "\n\n" + question_block(r)}]
            qi = question_image_path(r.question_image_file) if isinstance(r.question_image_file, str) else None
            if qi is not None and qi.exists():
                c += [{"type": "text", "text": "\nQuestion diagram:"}, {"type": "image_url", "image_url": {"url": image_url(qi)}}]
            if r.question_id in sol:
                c.append({"type": "text", "text": "\n\nReference solution (teacher):\n" + sol[r.question_id]})
            if a.input == "image":
                c += [{"type": "text", "text": "\n\nStudent's working (photo):"},
                      {"type": "image_url", "image_url": {"url": image_url(image_path(a.split, r.image_file))}}]
            if trans is not None:
                c.append({"type": "text", "text": "\n\nTranscription of the student's working:\n" + trans.get(r.trace_id, "")})
            c.append({"type": "text", "text": "\n\n" + (RR_GUIDE.format(k=len(cand_map[r.trace_id])) if cand_block is not None else guide)})
            convs.append([{"role": "system", "content": SYSTEM}, {"role": "user", "content": c}])
        t0 = time.time()
        res = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs=ctk)
        texts = [o.outputs[0].text for o in res]
        ntok = [len(o.outputs[0].token_ids) for o in res]
        # budget forcing: outputs with no ROOT line get '</think>\n\nROOT:' appended and a short second pass
        def _missing(t):
            if a.think and "</think>" not in t:
                return True  # truncated inside the thinking: always force
            return not re.search(r"ROOT\s*:\s*\S", t.split("</think>")[-1])

        miss = [i for i, t in enumerate(texts) if _missing(t)]
        forced = np.zeros(len(texts), dtype=bool)
        forced[miss] = True
        if miss:
            try:
                conv2 = []
                tok = llm.get_tokenizer()
                for i in miss:
                    body = texts[i].split("</think>")[0]
                    ids = tok.encode(body, add_special_tokens=False)
                    if len(ids) > 4000:  # keep the end of the reasoning (conclusions) so the prompt fits
                        body = tok.decode(ids[-4000:])
                    if a.think and a.force_rc:
                        conv2.append(convs[i] + [{"role": "assistant", "content": "ROOT:", "reasoning": body.replace("<think>", "").strip()}])
                    else:
                        pre = ("<think>\n" + body + "\n</think>\n\n") if a.think else (body.rstrip() + "\n")
                        conv2.append(convs[i] + [{"role": "assistant", "content": pre + "ROOT:"}])
                sp2 = SamplingParams(temperature=0.0, max_tokens=96 if a.cands else 32)
                res2 = llm.chat(conv2, sp2, use_tqdm=False, add_generation_prompt=False, continue_final_message=True,
                                chat_template_kwargs=ctk)
                for i, o in zip(miss, res2):
                    texts[i] = texts[i] + "\n</think>\n\nROOT:" + o.outputs[0].text
                print(f"budget-forced {len(miss)} rows", flush=True)
            except Exception as e:
                print("budget forcing failed:", type(e).__name__, str(e)[:200], flush=True)
        tops, tails, answers, ranked = [], [], [], []
        for txt in texts:
            ans = txt.split("</think>")[-1]
            m = re.findall(r"ROOT\s*:\s*(.+)", ans)
            codes = []
            if m:
                for c in re.split(r"[,\s]+", m[-1]):
                    c = c.strip().strip(".;`*'\"")
                    if c.upper() == "NONE":
                        c = "NONE"
                    if c in L and c not in codes:
                        codes.append(c)
            tops.append(" ".join(codes[:3]))
            ranked.append(" ".join(codes))
            tails.append(ans[-300:].replace("\n", " "))
            answers.append(re.sub(r"\s*ROOT\s*:.*$", "", ans.strip(), flags=re.S)[-1200:].strip())
        pd.DataFrame({"trace_id": sub.trace_id.values, "top3": tops, "ranked": ranked, "tail": tails, "answer": answers,
                      "forced": forced, "out_tokens": ntok}).to_csv(fn, index=False)
        if a.save_text:
            import json as _json
            with open(fn.with_suffix(".jsonl"), "w") as fh:
                for t, txt, fz, nt in zip(sub.trace_id.values, texts, forced, ntok):
                    fh.write(_json.dumps({"trace_id": t, "text": txt, "forced": bool(fz), "out_tokens": int(nt)}, ensure_ascii=False) + "\n")
        print(f"{fn.name}: {len(sub)} rows {time.time()-t0:.0f}s", flush=True)
    res = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(out.glob("shard_*.csv"))])
    if a.split == "train":
        m = res.merge(load("train")[["trace_id", "root_code"]], on="trace_id")
        s = map3_lists(m.root_code.tolist(), [t.split() for t in m.top3])
        nn = m.root_code != "NONE"
        print(f"n={len(m)} MAP@3={s.mean():.4f} top1={np.mean([t.split()[:1]==[y] for t,y in zip(m.top3,m.root_code)]):.3f} "
              f"nonNONE MAP@3={s[nn.values].mean():.4f} parsed={np.mean([len(t)>0 for t in m.top3]):.3f}")


if __name__ == "__main__":
    main()
