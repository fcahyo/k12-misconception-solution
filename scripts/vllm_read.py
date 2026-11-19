"""Transcribe the photos of students' workings line by line with a vision-language model served by vLLM (zero-shot).

Prompt versions (k12.prompt): v2 = question block (question text and answer key) + photo; v3q = question text as context
without the answer key, so the reader cannot 'correct' a wrong answer toward the key. Both are used downstream.
Output: <out>/shard_NNNN.csv with columns trace_id, transcription. Existing shards are skipped, so a run can be resumed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from k12.data import DATA, image_url, image_path, load
from k12.prompt import reader_messages, reader_messages_v3q


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True, choices=["train", "test", "unlabelled"])
    p.add_argument("--out", required=True)
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--quant", default=None, help="e.g. fp8 for online fp8")
    p.add_argument("--max-pixels", type=int, default=1024 * 1024)
    p.add_argument("--min-pixels", type=int, default=0, help="upscale smaller photos to this many pixels")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--rows", default=None, help="optional csv of trace_ids to restrict to")
    p.add_argument("--prompt", default="v2", choices=["v2", "v3q"])
    return p.parse_args()


def main():
    a = parse()
    from vllm import LLM, SamplingParams

    df = load(a.split)
    if a.rows:
        keep = set(pd.read_csv(a.rows).trace_id)
        df = df[df.trace_id.isin(keep)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = [df.iloc[i:i + a.shard_size] for i in range(0, len(df), a.shard_size)]
    todo = [(k, s) for k, s in enumerate(shards) if not (out / f"shard_{k:04d}.csv").exists()]
    if not todo:
        print("nothing to do")
        return
    kw = dict(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs,
              max_model_len=a.max_model_len, limit_mm_per_prompt={"image": 1, "video": 0},
              allowed_local_media_path=str(DATA), trust_remote_code=False)
    if a.quant:
        kw["quantization"] = a.quant
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    for k, s in todo:
        t0 = time.time()
        convs = []
        for _, r in s.iterrows():
            msgs = (reader_messages_v3q if a.prompt == "v3q" else reader_messages)(r, image_path(a.split, r.image_file))
            for m in msgs:
                if isinstance(m["content"], list):
                    m["content"] = [
                        {"type": "image_url", "image_url": {"url": image_url(c["image"], a.max_pixels, a.min_pixels)}} if c["type"] == "image" else c
                        for c in m["content"]
                    ]
            convs.append(msgs)
        res = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        texts = [o.outputs[0].text.strip() for o in res]
        pd.DataFrame({"trace_id": s.trace_id.values, "transcription": texts}).to_csv(out / f"shard_{k:04d}.csv", index=False)
        ntok = sum(len(o.outputs[0].token_ids) for o in res)
        print(f"shard {k}: {len(s)} rows {time.time()-t0:.1f}s ({len(s)/(time.time()-t0):.2f} img/s, {ntok/(time.time()-t0):.0f} out tok/s)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
