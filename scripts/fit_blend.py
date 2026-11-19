"""Constrained log-linear blend of component probabilities (the round-0 stack, whose top-8 codes feed the first reranker).

Usage:
  fit_blend.py --c NAME=run:RUN | NAME=npy:PATH ... --ref "NAME:W,NAME:W,..." [--folds 0 1 2 3 4] [--bias 1] [--l2 0.0]
               [--save OUT.npz]
Components: run:<RUN> -> the run's out-of-fold probabilities (k12.runs.oof_probs); npy:<PATH> -> an (n_train, 74) array
(NaN = row not covered). Rows: those of the requested folds that every component covers.
Fit: minimise the mean NLL of softmax(sum_i w_i log p_i + b [code == NONE]) + l2 * sum_i w_i^2 over 0 <= w_i <= 5
(L-BFGS-B, analytic gradient); with --bias 1 the NONE bias b in [-5, 5] is free, otherwise b = 0.
Report: the reference blend (fixed weights --ref, no bias) per fold; the cross-fitted blend (fold k scored with weights fitted
on the other folds) paired against the reference, with per-fold deltas; the refit on all rows (in-sample, optimistic).
--save writes rows, fold, y, s (cross-fitted per-row MAP@3), P (cross-fitted probabilities), s_ref (the reference's per-row
MAP@3), names, and w, b of the all-rows refit (read by make_rerank_cands.py).
"""
import argparse

import numpy as np
from scipy.optimize import minimize

from k12.data import folds, labels, load
from k12.metric import paired_bootstrap, row_scores
from k12.runs import oof_probs, train_targets


def logits(X, w, b):
    z = np.tensordot(w, X, axes=1)
    z[:, 0] += b
    return z


def softmax(z):
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def fit(X, y, bias=True, l2=0.0):
    """X: (M, n, K) log-probs; returns (w, b)."""
    M, n, _ = X.shape
    Xy = X[:, np.arange(n), y]  # (M, n)

    def f(th):
        w, b = th[:M], (th[M] if bias else 0.0)
        P = softmax(logits(X, w, b))
        lse_term = -(np.log(np.clip(P[np.arange(n), y], 1e-12, 1))).mean()
        gw = (np.einsum("mnk,nk->m", X, P) - Xy.sum(1)) / n + 2 * l2 * w
        g = list(gw)
        if bias:
            g.append(P[:, 0].mean() - np.mean(y == 0))
        return lse_term + l2 * np.sum(w ** 2), np.array(g)

    th0 = np.r_[np.full(M, 1.0 / M), [0.0] if bias else []]
    bounds = [(0, 5)] * M + ([(-5, 5)] if bias else [])
    r = minimize(f, th0, jac=True, method="L-BFGS-B", bounds=bounds)
    return r.x[:M], (r.x[M] if bias else 0.0)


def parse_w(s):
    return {k: float(v) for k, v in (p.split(":") for p in s.split(","))}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--c", nargs="+", required=True, help="NAME=run:RUN | NAME=npy:PATH")
    a.add_argument("--ref", required=True, help='fixed reference weights "name:w,name:w" (no bias)')
    a.add_argument("--folds", nargs="*", type=int, default=[0, 1])
    a.add_argument("--bias", type=int, default=1)
    a.add_argument("--l2", type=float, default=0.0)
    a.add_argument("--save", default=None, help="npz: rows, fold, y, cross-fitted per-row scores s and probabilities P, refit w and b")
    args = a.parse_args()
    L = labels(); K = len(L); n = len(load("train")); fo = folds().fold.values
    assert L[0] == "NONE"
    y, m = train_targets()
    names, comps = [], []
    for spec in args.c:
        name, src = spec.split("=", 1)
        kind, val = src.split(":", 1)
        P = oof_probs(val, n, K) if kind == "run" else np.load(val)
        names.append(name); comps.append(np.log(np.clip(P, 1e-6, 1)))
    ok = m & np.isin(fo, args.folds)
    for c in comps:
        ok &= ~np.isnan(c).any(1)
    X = np.stack([c[ok] for c in comps]); yy = y[ok]; f = fo[ok]
    print(f"rows {ok.sum()} folds {args.folds} (per fold {[int((f == k).sum()) for k in args.folds]}); components {names}")
    ref_w = parse_w(args.ref)
    wr = np.array([ref_w.get(nm, 0.0) for nm in names])
    s_ref = row_scores(softmax(logits(X, wr, 0.0)), yy)
    print(f"reference {args.ref}: {s_ref.mean():.4f} per fold {[round(float(s_ref[f == k].mean()), 4) for k in args.folds]}")
    s_cf = np.zeros(len(yy)); P_cf = np.zeros((len(yy), K), dtype=np.float32)
    for k in args.folds:
        tr, te = f != k, f == k
        w, b = fit(X[:, tr], yy[tr], bool(args.bias), args.l2)
        P_te = softmax(logits(X[:, te], w, b)); P_cf[te] = P_te
        s_cf[te] = row_scores(P_te, yy[te])
        print(f"  fit on folds != {k}: w { {nm: round(float(x), 3) for nm, x in zip(names, w / max(w.sum(), 1e-9))} } scale {w.sum():.2f} bias {b:+.2f}"
              f" -> fold {k} {s_cf[te].mean():.4f} (ref {s_ref[te].mean():.4f})")
    d = paired_bootstrap(s_cf, s_ref)
    per = [round(float(s_cf[f == k].mean() - s_ref[f == k].mean()), 4) for k in args.folds]
    print(f"CROSS-FITTED {s_cf.mean():.4f} vs ref {d['delta']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
          f"P={d['p_a_better']:.2f} per-fold {per}")
    w, b = fit(X, yy, bool(args.bias), args.l2)
    s_in = row_scores(softmax(logits(X, w, b)), yy)
    print(f"in-sample refit (optimistic) {s_in.mean():.4f}; weights {' '.join(f'{x:.4f}' for x in w)} (order {names}), "
          f"NONE bias {b:.3f}")
    if args.save:
        np.savez(args.save, rows=np.where(ok)[0], fold=f, y=yy, s=s_cf, P=P_cf, s_ref=s_ref, names=np.array(names), w=w, b=b)
        print("saved", args.save)


if __name__ == "__main__":
    main()
