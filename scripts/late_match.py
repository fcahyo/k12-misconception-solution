"""Share matching for the final files: shift the predicted code shares of the test set to the code shares of train.

Usage: late_match.py test STACK.npz RAW.npy OUT.csv [--rare2 0|1] [--min-count 5]
  RAW.npy: the final stack's test probabilities (written next to its CSV by fit_stack.py --predict).
  STACK.npz: the stack fit that produced RAW.npy. The command line names it, but the matching does not read it.
  Writes OUT.csv and OUT.npy (float32 matched probabilities) and prints a one-line JSON summary with the md5 of OUT.csv.
Matching = per-code offsets on the log-probabilities (iterative proportional fitting) such that the mean probability of each
code equals its share among all train rows. Codes with at least --min-count train rows get their own offset and the others
share one; with --rare2 1 the filler code NOT-01 also gets its own offset instead of sharing the rare codes' offset.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from k12.data import labels, load
from k12.metric import top3_strings

L = labels(); K = len(L); FILLERS = [L.index("NOT-01")]


def ipf(P, cnt, split=(), min_count=5, iters=400):
    """Offsets d (K,) so that mean softmax(log P + d) matches the shares of the counts cnt per group: one group per code with
    >= min_count rows, one per code in split, one shared group for all other codes. Returns (matched probabilities, d)."""
    own = [c for c in range(K) if cnt[c] >= min_count]; rare = [c for c in range(K) if cnt[c] < min_count and c not in split]
    groups = [[c] for c in own] + [[c] for c in split] + ([rare] if rare else [])
    share = np.array([cnt[g].sum() for g in groups]) / cnt.sum(); share = np.clip(share, 0.5 / cnt.sum(), None); share = share / share.sum()
    logp = np.log(np.clip(P, 1e-9, 1)); d = np.zeros(K)
    for _ in range(iters):
        z = logp + d; z -= z.max(1, keepdims=True); Q = np.exp(z); Q /= Q.sum(1, keepdims=True)
        m = np.array([Q[:, g].sum(1).mean() for g in groups]); step = np.log(share / np.clip(m, 1e-12, None))
        for g, s in zip(groups, step):
            d[g] += 0.7 * s
        if np.abs(step).max() < 1e-6:
            break
    z = logp + d; z -= z.max(1, keepdims=True); Q = np.exp(z)
    return Q / Q.sum(1, keepdims=True), d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["test"])
    p.add_argument("stack", help="the stack fit (.npz) that produced RAW.npy; not read")
    p.add_argument("raw_npy")
    p.add_argument("out_csv")
    p.add_argument("--rare2", type=int, default=0, help="1 = the filler code NOT-01 gets its own offset")
    p.add_argument("--min-count", type=int, default=5)
    a = p.parse_args()
    Pt = np.load(a.raw_npy).astype(np.float64)
    te, tr = load("test"), load("train")
    idx = {c: i for i, c in enumerate(L)}
    cnt = np.bincount(np.array([idx[c] for c in tr.root_code if c in idx]), minlength=K).astype(float)
    Q, _ = ipf(Pt, cnt, split=FILLERS if a.rare2 else (), min_count=a.min_count)
    sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": top3_strings(Q, L)})
    assert len(sub) == len(te) and sub.trace_id.is_unique and all(len(x.split()) == 3 for x in sub.root_code)
    sub.to_csv(a.out_csv, index=False)
    np.save(a.out_csv.replace(".csv", ".npy"), Q.astype(np.float32))
    first_none = float((sub.root_code.str.split().str[0] == "NONE").mean())
    md5 = hashlib.md5(Path(a.out_csv).read_bytes()).hexdigest()
    print(json.dumps(dict(out=a.out_csv, rare2=bool(a.rare2), min_count=a.min_count, rows=len(sub), first_NONE=first_none, md5=md5)))


if __name__ == "__main__":
    main()
