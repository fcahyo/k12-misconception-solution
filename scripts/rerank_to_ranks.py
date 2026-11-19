"""Second reranker's output (zeroshot_diagnose.py --cands shards, column `ranked`) -> a rank-coded component for fit_stack.py.

Usage: rerank_to_ranks.py <shard_dir> <split> <run_id> --cands data/rerank/<tag>_cands.csv   -> runs/<run_id>/rank.npy
rank.npy is an (n, K) array: 0 = not a candidate, r = the code's rank 1..K among the row's candidates in the reranker's order
(candidates it did not rank follow in stack order). Rows the reranker did not parse are all 0. Rows without a candidate list
and train rows the run did not cover are NaN; test rows it did not cover are all 0.
fit_stack.py reads one or more such arrays (samples) as a component NAME=rank:PATH[+PATH...].
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
    args = a.parse_args()
    L = labels(); idx = {c: i for i, c in enumerate(L)}
    df = load(args.split)
    cands = pd.read_csv(args.cands, keep_default_na=False)
    cands = cands[cands.split == args.split]
    cmap = dict(zip(cands.trace_id, cands.codes.str.split()))
    sh = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob(f"{args.zs_dir}/shard_*.csv"))])
    rk = dict(zip(sh.trace_id, sh.ranked))
    R = np.full((len(df), len(L)), np.nan, dtype=np.float32)
    n_cov = n_parsed = 0
    for i, t in enumerate(df.trace_id.values):
        if t not in cmap:
            continue
        if t not in rk:
            if args.split == "test":
                R[i] = 0
            continue
        n_cov += 1
        R[i] = 0
        codes = cmap[t]
        got = list(dict.fromkeys(c for c in str(rk[t]).split() if c in codes))
        if not got:
            continue  # unparsed: no indicators
        n_parsed += 1
        for r, c in enumerate(got + [c for c in codes if c not in got], 1):
            R[i, idx[c]] = r
    d = RUNS / args.run_id; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "rank.npy", R)
    print(f"{args.split}: covered {n_cov}/{len(cmap)}, parsed {n_parsed}; wrote {d / 'rank.npy'}")


if __name__ == "__main__":
    main()
