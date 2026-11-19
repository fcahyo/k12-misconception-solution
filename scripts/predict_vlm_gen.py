"""Inference with a saved fold adapter of train_vlm_gen.py on any split (exact 74-way label scoring), sharded.

Usage: predict_vlm_gen.py --run R --fold K --split {test,unlabelled,train} [--rows ids.csv] [--shard-size 1000]
       [--transcription csv] [--out-name pred_test] [--model BASE_MODEL_DIR]
Writes runs/R/fold{K}/<out-name>/shard_XXXX.npz (probs float16 (n,74), trace_id); for the whole test split it also
writes runs/R/fold{K}/test_probs.npy.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from k12.data import RUNS, labels, load
from k12.gen import label_token_ids, score_labels

_spec = importlib.util.spec_from_file_location("tvg", str(Path(__file__).with_name("train_vlm_gen.py")))
tvg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tvg)

# Options of earlier, experimental versions of train_vlm_gen.py that change the inputs or the model. An adapter trained with
# any of them switched on is refused: this script reproduces only the configuration of the two models.
RETIRED = {"tax": "compact", "rag": 0, "teacher": None, "test_teacher": None, "qlora": 0, "lora_vision": 0, "task": "code",
           "cands": None, "finding_first": 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--rows", default=None)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--transcription", default=None)
    p.add_argument("--out-name", default=None)
    p.add_argument("--solutions", default=None, help="override the run's reference-solution csv; default = the one recorded in the run's args.json")
    p.add_argument("--model", default=None, help="base model directory; default = the path recorded in the run's args.json")
    a = p.parse_args()
    fdir = RUNS / a.run / (f"fold{a.fold}" if a.fold not in ("full", "-1") else "full")
    targs = argparse.Namespace(**json.loads((fdir / "args.json").read_text()))
    on = {k: getattr(targs, k) for k, v in RETIRED.items() if getattr(targs, k, v) != v}
    if on:
        sys.exit(f"error: {fdir / 'args.json'} was trained with retired options {on}; this script reproduces only the configuration of the two models")
    if a.solutions:
        targs.solutions = a.solutions
    if a.model:
        targs.model = a.model
    if getattr(targs, "solutions", None):
        s = pd.read_csv(targs.solutions, keep_default_na=False)
        tvg.SOLUTIONS = dict(zip(s.question_id, s.solution))
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dev = "cuda:0"
    proc = AutoProcessor.from_pretrained(targs.model)
    proc.image_processor.size = {"shortest_edge": targs.min_pixels, "longest_edge": targs.max_pixels}
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    base = AutoModelForImageTextToText.from_pretrained(targs.model, **kw)
    base.to(dev)
    model = PeftModel.from_pretrained(base, fdir / "adapter").eval()
    L = labels()
    lab_ids = label_token_ids(proc.tokenizer, L)
    df = load(a.split)
    if a.rows:
        keep = set(pd.read_csv(a.rows).trace_id)
        df = df[df.trace_id.isin(keep)].reset_index(drop=True)
    trans = None
    if a.transcription:
        t = pd.read_csv(a.transcription, keep_default_na=False)
        trans = dict(zip(t.trace_id, t.transcription))
    out = fdir / (a.out_name or f"pred_{a.split}")
    out.mkdir(exist_ok=True)
    for s0 in range(0, len(df), a.shard_size):
        fn = out / f"shard_{s0 // a.shard_size:04d}.npz"
        if fn.exists():
            continue
        sub = df.iloc[s0:s0 + a.shard_size].reset_index(drop=True)
        ds = tvg.Rows(sub, a.split, proc, targs, trans, train=False)
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=6, prefetch_factor=4)
        P = np.zeros((len(sub), len(L)), dtype=np.float32)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for b in dl:
                P[b["row"]] = torch.softmax(score_labels(model, b["enc_p"], lab_ids, dev).float(), -1).cpu().numpy()
        np.savez(fn, probs=P.astype(np.float16), trace_id=sub.trace_id.values)
        print(f"{fn.name}: {len(sub)} rows {time.time()-t0:.0f}s", flush=True)
    if a.split == "test" and not a.rows:
        parts = [np.load(f, allow_pickle=True) for f in sorted(out.glob("shard_*.npz"))]
        P = np.concatenate([x["probs"] for x in parts]).astype(np.float32)
        ids = np.concatenate([x["trace_id"] for x in parts])
        assert (ids == load("test").trace_id.values).all()
        np.save(fdir / "test_probs.npy", P)
        print("wrote", fdir / "test_probs.npy")
    print("DONE")


if __name__ == "__main__":
    main()
