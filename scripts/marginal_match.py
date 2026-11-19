"""Share matching: shift the predicted code shares of a target set to known code shares (used on the round-1 stack, before its
top-6 candidate lists are built with make_cands_from_stack.py).

  marginal_match.py oof STACK.npz --pooled 1 --save OUT.npz [--min-count 5]
      the cross-fitted train probabilities in STACK.npz (fit_stack.py --save) are matched once, over all its rows, to the gold
      code counts of those rows: the train-side analogue of matching the whole test set at once. OUT.npz has the fields
      rows, fold, y, s, P, names (as STACK.npz, with the matched P and its per-row MAP@3 s).
  marginal_match.py test SUB.npy --out outputs/X.csv [--min-count 5]
      the test probabilities SUB.npy (written next to its CSV by fit_stack.py --predict) are matched to the code shares
      of all train rows; writes outputs/X.csv and outputs/X.npy.
Matching = per-code offsets on the log-probabilities, found by iterative proportional fitting, such that the mean probability
of each code equals its reference share; codes with fewer than --min-count reference rows share one offset matched to their
pooled share. Only the target rows' predictions and the reference code counts are used, never a row's own label.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from k12.data import labels, load
from k12.metric import row_scores, top3_strings


def ipf(P, y_ref, K, min_count=5, iters=300):
    """Offsets d (K,) so that mean softmax(log P + d) matches the reference shares per group: one group per code with
    >= min_count reference rows, one shared group for all other codes. Returns (matched probabilities, d, number of groups)."""
    cnt = np.bincount(y_ref, minlength=K).astype(float)
    own = np.where(cnt >= min_count)[0]
    rare = np.where(cnt < min_count)[0]
    groups = [[c] for c in own] + ([list(rare)] if len(rare) else [])
    share = np.array([cnt[g].sum() for g in groups]) / cnt.sum()
    share = np.clip(share, 0.5 / cnt.sum(), None); share = share / share.sum()
    logp = np.log(np.clip(P, 1e-9, 1)); d = np.zeros(K)
    for _ in range(iters):
        z = logp + d; z -= z.max(1, keepdims=True); Q = np.exp(z); Q /= Q.sum(1, keepdims=True)
        m = np.array([Q[:, g].sum(1).mean() for g in groups])
        step = np.log(share / np.clip(m, 1e-12, None))
        for g, s in zip(groups, step):
            d[g] += 0.7 * s
        if np.abs(step).max() < 1e-6:
            break
    z = logp + d; z -= z.max(1, keepdims=True); Q = np.exp(z)
    return Q / Q.sum(1, keepdims=True), d, len(groups)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["oof", "test"])
    a.add_argument("src")
    a.add_argument("--save", default=None)
    a.add_argument("--out", default=None)
    a.add_argument("--min-count", type=int, default=5)
    a.add_argument("--pooled", type=int, default=0, help="oof mode, required: 1 = match once over all OOF rows to their own "
                   "code counts (the analogue of matching the whole test set)")
    args = a.parse_args()
    L = labels(); K = len(L)
    if args.mode == "oof":
        if args.pooled != 1:
            a.error("oof mode needs --pooled 1 (one match over all OOF rows)")
        z = np.load(args.src, allow_pickle=True)
        P, y, f = z["P"].astype(np.float64), z["y"], z["fold"]
        P2, d, ng = ipf(P, y, K, args.min_count)
        s = row_scores(P2, y)
        print(f"rows {len(y)}: base {z['s'].mean():.4f} -> marginal-matched {s.mean():.4f}; per fold "
              f"{[round(float(s[f == k].mean()), 4) for k in sorted(set(f.tolist()))]}; groups {ng}; offsets range [{d.min():+.2f}, {d.max():+.2f}]")
        if args.save:
            np.savez(args.save, rows=z["rows"], fold=f, y=y, s=s, P=P2.astype(np.float32), names=z["names"])
            print("saved", args.save)
        return
    P = np.load(args.src).astype(np.float64)
    te, tr = load("test"), load("train")
    idx = {c: i for i, c in enumerate(L)}
    y_ref = np.array([idx[c] for c in tr.root_code if c in idx])  # ALL train rows, incl. the singleton codes that no fold scores
    P2, d, ng = ipf(P, y_ref, K, args.min_count)
    top = sorted(range(K), key=lambda c: -abs(d[c]))[:8]
    print(f"test rows {len(P)}; groups {ng}; largest offsets { {L[c]: round(float(d[c]), 2) for c in top} }")
    sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": top3_strings(P2, L)})
    assert len(sub) == len(te) and sub.trace_id.is_unique and all(len(x.split()) == 3 for x in sub.root_code)
    t1 = sub.root_code.str.split().str[0]
    print(f"pred NONE rate (top1): {(t1 == 'NONE').mean():.3f} (train {np.mean(tr.root_code == 'NONE'):.3f})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False); np.save(str(Path(args.out).with_suffix(".npy")), P2.astype(np.float32))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
