"""Candidate lists (the round-0 stack's top-K codes) for the first reranker: train = cross-fitted stack, test = refit stack.

Usage: make_rerank_cands.py STACK.npz --test-runs RUN ... [--k 8] --out data/rerank/<tag>
  STACK.npz from fit_blend.py --save (rows, P = cross-fitted probabilities, names, w, b = all-rows refit weights).
  --test-runs: the run ids of the stack components in the order of `names`; the refit weights are applied to their test
  probabilities (k12.runs.test_probs).
Writes <out>_cands.csv (trace_id, split, fold, codes = top-K in stack order, topk_mass) and <out>_P_train.npy (n_train x K_labels,
NaN outside the stack rows) / <out>_P_test.npy (refit stack on test).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from k12.data import labels, load
from k12.runs import test_probs


def main():
    a = argparse.ArgumentParser()
    a.add_argument("stack")
    a.add_argument("--test-runs", nargs="+", required=True)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--out", required=True)
    args = a.parse_args()
    z = np.load(args.stack, allow_pickle=True)
    L = np.array(labels()); K = len(L)
    tr, te = load("train"), load("test")
    names, w, b = list(z["names"]), z["w"], float(z["b"])
    assert len(args.test_runs) == len(names), (args.test_runs, names)
    Ptr = np.full((len(tr), K), np.nan, dtype=np.float32)
    Ptr[z["rows"]] = z["P"]
    logp = 0
    for r, wi in zip(args.test_runs, w):
        p, _ = test_probs(r)
        logp = logp + wi * np.log(np.clip(p, 1e-6, 1))
    logp[:, 0] += b
    Pte = np.exp(logp - logp.max(1, keepdims=True)); Pte /= Pte.sum(1, keepdims=True)
    rows = []
    for split, P, df, fo in (("train", Ptr[z["rows"]], tr.iloc[z["rows"]], z["fold"]), ("test", Pte, te, np.full(len(te), -1))):
        top = np.argsort(-P, 1)[:, :args.k]
        for i, (tid, t) in enumerate(zip(df.trace_id.values, top)):
            rows.append({"trace_id": tid, "split": split, "fold": int(fo[i]), "codes": " ".join(L[t]), "topk_mass": float(P[i, t].sum())})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    c = pd.DataFrame(rows)
    c.to_csv(f"{out}_cands.csv", index=False)
    np.save(f"{out}_P_train.npy", Ptr); np.save(f"{out}_P_test.npy", Pte.astype(np.float32))
    yt = dict(zip(tr.trace_id, tr.root_code))
    ct = c[c.split == "train"]
    rec = np.mean([yt[t] in s.split() for t, s in zip(ct.trace_id, ct.codes)])
    print(f"wrote {out}_cands.csv: train {len(ct)} rows (gold in top-{args.k} {rec:.3f}), test {int((c.split == 'test').sum())} rows; "
          f"mean top-{args.k} mass train {ct.topk_mass.mean():.3f} test {c[c.split == 'test'].topk_mass.mean():.3f}")


if __name__ == "__main__":
    main()
