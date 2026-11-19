"""First reranker's output (zeroshot_diagnose.py --cands shards, column `ranked`) -> a 74-way probability component for the stack.

Usage: rerank_to_probs.py <shard_dir> <split> <run_id> --cands data/rerank/<tag>_cands.csv --stack-P data/rerank/<tag>_P_<split>.npy
          [--decay 0.42]
Per row: the K candidates get p_r ∝ decay^(r-1) by the reranker's rank (candidates it did not rank go after the ranked ones, in
stack order), scaled to the stack's top-K mass; codes outside the top-K keep the stack's probabilities. Unparsed rows and test
rows missing from the shards copy the stack row (no information); train rows missing from the shards stay NaN (not covered).
Writes runs/<run_id>/oof_partial.npy (train; NaN outside the covered rows) or test.npy (test; rows without candidates = uniform).
"""
import argparse
import glob

import numpy as np
import pandas as pd

from k12.data import RUNS, labels, load


def main():
    a = argparse.ArgumentParser()
    a.add_argument("zs_dir"); a.add_argument("split"); a.add_argument("run_id")
    a.add_argument("--cands", required=True)
    a.add_argument("--stack-P", required=True)
    a.add_argument("--decay", type=float, default=0.42)
    args = a.parse_args()
    L = labels(); idx = {c: i for i, c in enumerate(L)}
    df = load(args.split)
    SP = np.load(args.stack_P)
    cands = pd.read_csv(args.cands, keep_default_na=False)
    cands = cands[cands.split == args.split]
    cmap = dict(zip(cands.trace_id, cands.codes.str.split()))
    sh = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob(f"{args.zs_dir}/shard_*.csv"))])
    rk = dict(zip(sh.trace_id, sh.ranked))
    P = np.full((len(df), len(L)), np.nan, dtype=np.float32)
    n_cov = n_parsed = 0
    for i, t in enumerate(df.trace_id.values):
        if t not in cmap or np.isnan(SP[i]).any():
            continue
        if t not in rk and args.split == "train":
            continue  # train rows the run did not cover stay NaN
        n_cov += t in rk
        row = SP[i].copy()
        codes = cmap[t]
        got = [c for c in str(rk.get(t, "")).split() if c in codes]
        got = list(dict.fromkeys(got))
        if got:
            n_parsed += 1
            order = got + [c for c in codes if c not in got]
            mass = row[[idx[c] for c in codes]].sum()
            pr = args.decay ** np.arange(len(order)); pr = pr / pr.sum() * mass
            for c, p in zip(order, pr):
                row[idx[c]] = p
        P[i] = row
    print(f"{args.split}: covered {n_cov}/{len(cmap)}, parsed {n_parsed}")
    d = RUNS / args.run_id; d.mkdir(parents=True, exist_ok=True)
    if args.split == "train":
        np.save(d / "oof_partial.npy", P)
    else:
        np.save(d / "test.npy", np.nan_to_num(P, nan=1.0 / len(L)))
    print("wrote", d)


if __name__ == "__main__":
    main()
