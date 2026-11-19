"""LoRA fine-tuning of a Qwen3.5-family vision-language model that *writes* the root code (copyable from the taxonomy in
the prompt), optionally followed by auxiliary text (final-answer correctness, the given transcription).
Evaluation scores every label string exactly (k12.gen.score_labels), giving a full 74-way distribution per row.

One call trains one fold: runs/<run>/fold<k>/{adapter/, args.json, val_probs.npy, val_rows.npy, test_probs.npy}.
The pipeline uses two student configurations (see scripts/stages.sh).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from k12.data import RUNS, folds, image_path, labels, load, question_image_path
from k12.gen import label_token_ids, score_labels
from k12.metric import report
from k12.prompt import classify_messages


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=1024 * 1024)
    p.add_argument("--min-pixels", type=int, default=256 * 256)
    p.add_argument("--qimg", type=int, default=1)
    p.add_argument("--image", type=int, default=1)
    p.add_argument("--transcription", default="none")
    p.add_argument("--test-transcription", default=None)
    p.add_argument("--aux", default="", help="comma list of: fac (final-answer correctness), trans (the given transcription)")
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--solutions", default=None, help="csv(question_id, solution) of reference solutions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--predict-test", type=int, default=1)
    p.add_argument("--save-adapter", type=int, default=1)
    p.add_argument("--eval-epochs", default="", help="comma list of epochs to eval at (default: last only)")
    a = p.parse_args()
    unknown = set(x for x in a.aux.split(",") if x) - {"fac", "trans"}
    if unknown:
        p.error(f"--aux: unknown kinds {sorted(unknown)} (allowed: fac, trans)")
    return a


INSTR = "\n\nAnswer with the root misconception code only."


def aux_text(r, kinds):
    parts = []
    if "fac" in kinds:
        fac = str(r.final_answer_correct).lower() == "true"
        parts.append("Final answer: " + ("correct" if fac else "incorrect"))
    if "trans" in kinds:
        parts.append("Transcription:\n" + str(r.transcription_steps))
    return "\n".join(parts)


class Rows(torch.utils.data.Dataset):
    def __init__(self, df, split, proc, a, trans=None, train=True):
        self.df, self.split, self.proc, self.a, self.trans, self.train = df, split, proc, a, trans, train
        self.kinds = set(x for x in a.aux.split(",") if x)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        a = self.a
        msgs = classify_messages(
            r,
            transcription=(self.trans.get(r.trace_id) if self.trans is not None else None),
            solution=(SOLUTIONS.get(r.question_id) if SOLUTIONS else None),
            with_image=bool(a.image), image_path=image_path(getattr(r, "src_split", None) or self.split, r.image_file) if a.image else None,
            with_question_image=bool(a.qimg),
            question_image_path=question_image_path(r.question_image_file) if a.qimg else None,
        )
        msgs[-1]["content"].append({"type": "text", "text": INSTR})
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        images = []
        from PIL import Image

        for m in msgs:
            if isinstance(m["content"], list):
                for c in m["content"]:
                    if c["type"] == "image":
                        images.append(Image.open(c["image"]).convert("RGB"))
        kw = dict(text=[prompt], return_tensors="pt")
        if images:
            kw["images"] = images
        enc_p = self.proc(**kw)
        item = {"enc_p": dict(enc_p), "row": i}
        if self.train:
            code = r.root_code
            aux = aux_text(r, self.kinds) if self.kinds else ""
            full = prompt + code + ("\n" + aux if aux else "") + "<|im_end|>\n"
            kw["text"] = [full]
            enc = self.proc(**kw)
            n_p = enc_p["input_ids"].shape[1]
            assert torch.equal(enc["input_ids"][0, :n_p], enc_p["input_ids"][0]), "prefix mismatch"
            n_code = len(self.proc.tokenizer(code, add_special_tokens=False).input_ids)
            item.update({"enc": dict(enc), "n_p": n_p, "n_code": n_code})
        return item


SOLUTIONS = None


def main():
    global SOLUTIONS
    a = parse()
    if a.solutions:
        _s = pd.read_csv(a.solutions, keep_default_na=False)
        SOLUTIONS = dict(zip(_s.question_id, _s.solution))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda:0"
    L = labels()
    K = len(L)
    out = RUNS / a.run / (f"fold{a.fold}" if a.fold >= 0 else "full")
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    logf = open(out / "log.txt", "a")

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    proc = AutoProcessor.from_pretrained(a.model)
    proc.image_processor.size = {"shortest_edge": a.min_pixels, "longest_edge": a.max_pixels}
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    model.to(device)
    tgt = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=tgt))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    inner = model.base_model.model
    lab_ids = label_token_ids(proc.tokenizer, L)

    tr, te = load("train"), load("test")
    fo = folds().fold.values
    tr_mask = fo != a.fold if a.fold >= 0 else np.ones(len(tr), bool)
    va_mask = fo == a.fold if a.fold >= 0 else np.zeros(len(tr), bool)
    tr_df, va_df = tr[tr_mask].reset_index(drop=True), tr[va_mask].reset_index(drop=True)
    tr_df = tr_df.assign(src_split="train")

    def trans_map(spec, df):
        if spec in (None, "none"):
            return None
        if spec == "gold":
            return dict(zip(df.trace_id, df.transcription_steps))
        t = pd.read_csv(spec, keep_default_na=False)
        return dict(zip(t.trace_id, t.transcription))

    trans_tr = trans_map(a.transcription, tr)
    trans_te = trans_map(a.test_transcription, te) if a.test_transcription else None
    ds = Rows(tr_df, "train", proc, a, trans_tr, train=True)
    dl_kw = dict(batch_size=1, collate_fn=lambda b: b[0], num_workers=6, prefetch_factor=4)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, shuffle=True, generator=g, **dl_kw)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    spe = len(ds) // a.grad_accum
    total = max(1, int(round(spe * a.epochs)))
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, (s - warm) / max(1, total - warm)))))
    eval_steps = {int(round(spe * float(e))) for e in a.eval_epochs.split(",") if e} | {total}

    def run_eval(df, split, trans):
        model.eval()
        dsv = Rows(df, split, proc, a, trans, train=False)
        dlv = torch.utils.data.DataLoader(dsv, shuffle=False, **dl_kw)
        P = np.zeros((len(df), K), dtype=np.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for b in dlv:
                s = score_labels(model, b["enc_p"], lab_ids, device)
                P[b["row"]] = torch.softmax(s.float(), -1).cpu().numpy()
        model.train()
        return P

    idx = {c: j for j, c in enumerate(L)}
    y_va = np.array([idx[r.root_code] for r in va_df.itertuples()]) if len(va_df) else np.array([])
    log(f"train rows {len(ds)} val rows {len(va_df)} steps {total} eval at {sorted(eval_steps)}")
    model.train()
    step = micro = 0
    t0 = time.time(); run_c = run_a = 0.0; ntok = 0
    done = False
    pv = None
    while not done:
        for b in dl:
            enc, n_p, n_code = b["enc"], b["n_p"], b["n_code"]
            kw = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inner.model.rope_deltas = None
                h = inner.model(**kw, use_cache=False).last_hidden_state[0, n_p - 1:-1]
                logits = inner.lm_head(h).float()
            tgt_ids = kw["input_ids"][0, n_p:]
            ce = F.cross_entropy(logits, tgt_ids, reduction="none")
            loss_c = ce[:n_code].mean()
            loss = loss_c
            if len(ce) > n_code and a.aux:
                loss_a = ce[n_code:].mean()
                loss = loss + a.aux_weight * loss_a
                run_a += loss_a.item()
            (loss / a.grad_accum).backward()
            run_c += loss_c.item(); micro += 1; ntok += kw["input_ids"].shape[1]
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0:
                    dt = time.time() - t0
                    log(f"step {step}/{total} code_loss {run_c/micro:.4f} aux_loss {run_a/micro:.4f} lr {sched.get_last_lr()[0]:.2e} {dt/micro:.2f}s/sample {ntok/dt:.0f} tok/s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G")
                if step in eval_steps and len(va_df):
                    te0 = time.time()
                    pv = run_eval(va_df, "train", trans_tr)
                    met = report(pv, y_va)
                    log(f"step {step} (ep {step/spe:.2f}) val: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}) + f" eval {time.time()-te0:.0f}s")
                    np.save(out / f"val_probs_s{step}.npy", pv)
                if step >= total:
                    done = True
                    break
    if pv is not None:
        np.save(out / "val_probs.npy", pv)
        np.save(out / "val_rows.npy", np.where(va_mask)[0])
        (out / "metrics.json").write_text(json.dumps(met, indent=1))
    if a.save_adapter:
        model.save_pretrained(out / "adapter")
    if a.predict_test:
        t1 = time.time()
        pt = run_eval(te.assign(src_split="test"), "test", trans_te)
        np.save(out / "test_probs.npy", pt)
        log(f"test inference {time.time()-t1:.0f}s")
    log("DONE")


if __name__ == "__main__":
    main()
