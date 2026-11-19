"""Candidate lists (the top-K codes) for the second reranker, from the round-1 stack (fit_stack.py) or its share-matched form.

Usage: make_cands_from_stack.py STACK.npz TEST.npy [--k 6] --out data/rerank/<tag>
  STACK.npz: rows, fold and P (cross-fitted train probabilities), from fit_stack.py --save or marginal_match.py oof --save;
  TEST.npy: the corresponding test probabilities, written next to its CSV by fit_stack.py --predict or marginal_match.py test.
Writes <out>_cands.csv (trace_id, split, fold, codes = the top-K codes by probability, topk_mass = their total probability),
<out>_P_train.npy (n_train x 74, NaN outside the stack rows) and <out>_P_test.npy.
"""
import argparse

import numpy as np
import pandas as pd

from k12.data import labels, load


def main():
    a = argparse.ArgumentParser()
    a.add_argument("stack")
    a.add_argument("test_npy")
    a.add_argument("--k", type=int, default=6)
    a.add_argument("--out", required=True)
    args = a.parse_args()
    L = labels(); tr, te = load("train"), load("test")
    z = np.load(args.stack, allow_pickle=True); Pt = np.load(args.test_npy)
    Ptr = np.full((len(tr), len(L)), np.nan, np.float32); Ptr[z["rows"]] = z["P"]
    recs = []
    for i, f in zip(z["rows"], z["fold"]):
        o = np.argsort(-Ptr[i])[:args.k]
        recs.append((tr.trace_id.values[i], "train", int(f), " ".join(L[j] for j in o), float(Ptr[i, o].sum())))
    for i in range(len(te)):
        o = np.argsort(-Pt[i])[:args.k]
        recs.append((te.trace_id.values[i], "test", -1, " ".join(L[j] for j in o), float(Pt[i, o].sum())))
    d = pd.DataFrame(recs, columns=["trace_id", "split", "fold", "codes", "topk_mass"])
    d.to_csv(args.out + "_cands.csv", index=False)
    np.save(args.out + "_P_train.npy", Ptr)
    np.save(args.out + "_P_test.npy", Pt.astype(np.float32))
    print("wrote", args.out + "_cands.csv", len(d))


if __name__ == "__main__":
    main()
