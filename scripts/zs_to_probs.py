"""Convert a zero-shot teacher's top-3 codes into rank-based pseudo-probabilities for the stack.

Usage: zs_to_probs.py <zs_out_dir> <split> <run_id> [--p 0.6 0.25 0.1]
Reads the teacher's shard CSVs (column `top3`). A row's codes get the probabilities --p in rank order and the other codes share
the rest evenly; a row with fewer than three valid codes is renormalised, and a row without any valid code gets the train prior.
  train: writes runs/<run_id>/oof_partial.npy (rows the teacher did not cover = NaN) and prints the OOF metrics; when every
         scored row is covered it also saves the run with k12.runs.save_run (oof.npy, config.json, metrics.json).
  test:  writes runs/<run_id>/test.npy (rows not covered = uniform).
"""
import argparse
import glob

import numpy as np
import pandas as pd

from k12.data import RUNS, labels, load
from k12.metric import report
from k12.runs import save_run, train_targets


def main():
    a = argparse.ArgumentParser()
    a.add_argument("zs_dir")
    a.add_argument("split")
    a.add_argument("run_id")
    a.add_argument("--p", nargs=3, type=float, default=[0.6, 0.25, 0.1])
    args = a.parse_args()
    L = labels()
    idx = {c: i for i, c in enumerate(L)}
    K = len(L)
    df = load(args.split)
    tr = load("train")
    prior = np.bincount(tr.root_code.map(idx), minlength=K).astype(float)
    prior /= prior.sum()
    zs = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob(f"{args.zs_dir}/shard_*.csv"))])
    zs = dict(zip(zs.trace_id, zs.top3))
    P = np.full((len(df), K), np.nan, dtype=np.float32)
    rest = 1 - sum(args.p)
    n_cov = n_parsed = 0
    for i, t in enumerate(df.trace_id):
        if t not in zs:
            continue
        n_cov += 1
        codes = [c for c in str(zs[t]).split() if c in idx][:3]
        if not codes:
            P[i] = prior
            continue
        n_parsed += 1
        row = np.full(K, rest / (K - len(codes)))
        for r, c in enumerate(codes):
            row[idx[c]] = args.p[r]
        if len(codes) < 3:  # renormalise unused rank mass
            row = row / row.sum()
        P[i] = row
    print(f"covered {n_cov}/{len(df)}, parsed {n_parsed}")
    d = RUNS / args.run_id
    d.mkdir(parents=True, exist_ok=True)
    if args.split == "train":
        np.save(d / "oof_partial.npy", P)
        y, m = train_targets()
        mm = m & ~np.isnan(P).any(1)
        print(report(P[mm], y[mm]))
        if (~np.isnan(P).any(1))[m].all():
            save_run(args.run_id, P, None, {"zs_dir": args.zs_dir, "p": args.p})
    else:
        np.save(d / "test.npy", np.nan_to_num(P, nan=1.0 / K))
        print("wrote", d / "test.npy")


if __name__ == "__main__":
    main()
