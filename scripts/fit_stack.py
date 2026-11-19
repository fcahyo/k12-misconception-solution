"""Cross-fitted log-linear stack over the components' per-code scores (the round-1 stack and the two final stacks).

Fit:
  fit_stack.py --c NAME=run:RUN | NAME=npy:PATH | NAME=rank:PATH[+PATH...] ... [--folds 0 1 2 3 4] [--code-bias N]
               [--rank-shape NAME ...] --save OUT.npz
Predict (test):
  fit_stack.py --predict OUT.npz --test NAME=RUN | NAME=rank:PATH[+PATH...] ... --out outputs/X.csv

Logit(c) = sum_i w_i log p_i(c) + b [c = NONE] + sum_j b_j [c = j]
  w_i in [0, 5] per component; the NONE bias b and the per-code biases b_j in [-5, 5]. With --code-bias N, every non-NONE code
  with at least N gold rows among the training rows of a fit gets its own bias b_j.
  --rank-shape NAME ...: these components (teachers whose probabilities only encode a top-3 ranking, see zs_to_probs.py) get
  three free scores in [0, 10], for the codes they rank first, second and third, instead of w_i log p_i(c).
  rank:PATH[+PATH...]: a rank-coded component (rerank_to_ranks.py; one array per sample, 0 = not a candidate) gets five free
  scores in [0, 10], for ranks 1, 2, 3, 4 and 5+, each applied to the mean over its samples of the rank-bin indicator.
Components: run:RUN -> the run's out-of-fold probabilities (k12.runs.oof_probs); npy:PATH -> an (n_train, 74) array (NaN = row
not covered). Rows: those of the requested folds that every component covers.
Fit = mean NLL (L-BFGS-B, analytic gradient), cross-fitted by fold (fold k scored with parameters fitted on the other folds),
then refit on all rows. OUT.npz holds rows, fold, y, s (cross-fitted per-row MAP@3), P (cross-fitted probabilities), names,
theta (the all-rows refit), feat_names, spec (the feature specification, JSON) and opts.
--predict applies theta to the test components, whose run probabilities are the arithmetic mean of the run's fold models
(k12.runs.test_probs), and writes outputs/X.csv with outputs/X.npy (float32 probabilities).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from k12.data import folds, labels, load
from k12.metric import row_scores, top3_strings
from k12.runs import oof_probs, test_probs, train_targets


def softmax(z):
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


RANK_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]  # rank-coded components (kind "rank:"): one free score per bin


def load_ranks(val):
    """'PATH[+PATH2...]' -> (S, n, K) rank arrays of S samples (0 = not a candidate)."""
    return np.stack([np.load(p) for p in val.split("+")])


def design(X, names, bias_codes, rank_shape, ranks):
    """Features (Q, n, K), bounds and feature names: each component (one weight, or its rank / rank-bin scores), the NONE
    indicator, then the per-code biases. The '@0' suffix is part of the saved feature names (the fit has one row segment)."""
    M, n, K = X.shape
    F, bnd, fn = [], [], []
    for i in range(M):
        if names[i] in ranks:  # rank-coded component: mean over its samples of the bin indicators
            for lo, hi in RANK_BINS:
                e = np.mean([(Rk >= lo) & (Rk <= hi) for Rk in ranks[names[i]]], 0).astype(np.float32)
                F.append(e); bnd.append((0, 10)); fn.append(f"{names[i]}[r{lo}{'+' if hi > lo else ''}]@0")
        elif names[i] in rank_shape:  # three free scores for the component's rank 1 / 2 / 3 instead of w * log p
            order = np.argsort(-X[i], 1)[:, :3]
            top = np.sort(X[i], 1)[:, ::-1]
            for r in range(3):
                e = np.zeros((n, K), np.float32)
                valid = top[:, r] > top[:, 3] + 1e-6  # a ranked code stands out from the uniform rest
                e[np.arange(n)[valid], order[valid, r]] = 1.0
                F.append(e); bnd.append((0, 10)); fn.append(f"{names[i]}[rank{r + 1}]@0")
        else:
            F.append(X[i]); bnd.append((0, 5)); fn.append(f"{names[i]}@0")
    e = np.zeros((n, K), np.float32); e[:, 0] = 1.0
    F.append(e); bnd.append((-5, 5)); fn.append("NONE@0")
    for c in bias_codes:  # per-code bias (the NONE bias is the feature above)
        e = np.zeros((n, K), np.float32); e[:, int(c)] = 1.0
        F.append(e); bnd.append((-5, 5)); fn.append(f"bias[{int(c)}]")
    return np.stack(F), bnd, fn


def fit(F, y, bnd):
    Q, n, _ = F.shape
    Fy = F[:, np.arange(n), y]

    def f(th):
        P = softmax(np.tensordot(th, F, axes=1))
        nll = -np.log(np.clip(P[np.arange(n), y], 1e-12, 1)).mean()
        g = (np.einsum("qnk,nk->q", F, P) - Fy.sum(1)) / n
        return nll, g

    th0 = np.array([0.15 if lo == 0 else 0.0 for lo, _ in bnd])
    return minimize(f, th0, jac=True, method="L-BFGS-B", bounds=bnd).x


def feature_label(name, L):
    """Feature name for printing: bias[c] shows the code instead of its index."""
    return f"bias[{L[int(name[5:-1])]}]" if name.startswith("bias[") else name


def predict(args):
    z = np.load(args.predict, allow_pickle=True)
    spec = json.loads(str(z["spec"]))
    # The fit-file spec also records three options this code does not implement (component agreement, row segments, per-row
    # flags); they must be off.
    assert spec["agree"] == 0 and spec["seg"] == "none" and not spec["flags"], f"unsupported options in {args.predict}: {spec}"
    names, theta = list(z["names"]), z["theta"]
    L = labels(); te, tr = load("test"), load("train")
    runs = dict(s.split("=", 1) for s in args.test)
    assert set(runs) == set(names), (sorted(runs), names)
    comps, ranks = [], {}
    for nm in names:
        if runs[nm].startswith("rank:"):
            ranks[nm] = load_ranks(runs[nm][5:])
            assert not np.isnan(ranks[nm]).any(), runs[nm]
            print(f"{nm}: {runs[nm]} ({len(ranks[nm])} sample(s), rank-coded)")
            comps.append(np.zeros(ranks[nm].shape[1:], np.float32))
            continue
        p, nmod = test_probs(runs[nm])
        print(f"{nm}: {runs[nm]} ({nmod} model(s))")
        comps.append(np.log(np.clip(p, 1e-6, 1)))
    X = np.stack(comps).astype(np.float32)
    F, _, fn = design(X, names, spec["bias_codes"], spec["rank_shape"], ranks)
    assert fn == list(z["feat_names"]), (fn, list(z["feat_names"]))
    P = softmax(np.tensordot(theta, F, axes=1))
    sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": top3_strings(P, L)})
    top1 = sub.root_code.str.split().str[0]
    print(f"pred NONE rate (top1): {(top1 == 'NONE').mean():.3f}  (train NONE rate {np.mean(tr.root_code == 'NONE'):.3f}); expected NONE mass {P[:, 0].mean():.3f}")
    assert len(sub) == len(te) and sub.trace_id.is_unique and set(sub.trace_id) == set(te.trace_id)
    assert all(len(x.split()) == 3 and set(x.split()) <= set(L) for x in sub.root_code)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    np.save(str(Path(args.out).with_suffix(".npy")), P.astype(np.float32))
    print("wrote", args.out)


def fit_and_save(args):
    L = labels(); K = len(L); n = len(load("train")); fo = folds().fold.values
    y, m = train_targets()
    names, comps, rank_all = [], [], {}
    for spec in args.c:
        name, src = spec.split("=", 1)
        kind, val = src.split(":", 1)
        if kind == "rank":  # rank-coded: keep the ranks aside, a zero placeholder in X (NaN where any sample is not covered)
            Rk = load_ranks(val)
            rank_all[name] = Rk
            P = np.where(np.isnan(Rk).any(0), np.nan, 1.0)
        else:
            P = oof_probs(val, n, K) if kind == "run" else np.load(val)
        names.append(name); comps.append(np.log(np.clip(P, 1e-6, 1)))
    ok = m & np.isin(fo, args.folds)
    for c in comps:
        ok &= ~np.isnan(c).any(1)
    X = np.stack([c[ok] for c in comps]).astype(np.float32); yy = y[ok]; f = fo[ok]
    ranks = {nm: Rk[:, ok] for nm, Rk in rank_all.items()}
    s_cf = np.zeros(len(yy)); P_cf = np.zeros((len(yy), K), np.float32)
    for k in args.folds:
        trm, tem = f != k, f == k
        bc = [c for c in range(1, K) if (yy[trm] == c).sum() >= args.code_bias] if args.code_bias else []
        F, bnd, fnames = design(X, names, bc, args.rank_shape, ranks)
        th = fit(F[:, trm], yy[trm], bnd)
        Pk = softmax(np.tensordot(th, F[:, tem], axes=1)); P_cf[tem] = Pk
        s_cf[tem] = row_scores(Pk, yy[tem])
        print(f"  fold {k}: {s_cf[tem].mean():.4f}  params { {a_: round(float(b_), 3) for a_, b_ in zip(fnames, th)} }")
    per = [round(float(s_cf[f == k].mean()), 4) for k in args.folds]
    print(f"rows {ok.sum()} components {names}: CROSS-FITTED {s_cf.mean():.4f} per fold {per}")
    bc_all = [c for c in range(1, K) if (yy == c).sum() >= args.code_bias] if args.code_bias else []
    F, bnd, fnames = design(X, names, bc_all, args.rank_shape, ranks)  # all-rows refit (for --predict)
    theta = fit(F, yy, bnd)
    s_in = row_scores(softmax(np.tensordot(theta, F, axes=1)), yy)
    print(f"in-sample refit (optimistic) {s_in.mean():.4f}; params { {feature_label(a_, L): round(float(b_), 3) for a_, b_ in zip(fnames, theta)} }")
    if args.save:
        # agree / seg / flags / flag_rates: the options of the fit-file format that this code keeps off (see predict)
        spec = {"agree": 0, "seg": "none", "rank_shape": list(args.rank_shape), "bias_codes": [int(c) for c in bc_all],
                "flags": {}, "flag_rates": {}}
        np.savez(args.save, rows=np.where(ok)[0], fold=f, y=yy, s=s_cf, P=P_cf, names=np.array(names), theta=theta,
                 feat_names=np.array(fnames), spec=json.dumps(spec), opts=np.array(["agree=0", "seg=none"]))
        print("saved", args.save)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--c", nargs="+", default=[], help="NAME=run:RUN | NAME=npy:PATH | NAME=rank:PATH[+PATH...] (train side)")
    a.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    a.add_argument("--rank-shape", nargs="*", default=[], help="components whose probabilities only encode a top-3 ranking (zs_to_probs.py "
                   "teachers): free scores for their ranks 1-3")
    a.add_argument("--code-bias", type=int, default=0, help="N > 0: one free bias per non-NONE code with >= N gold rows in the training rows of the fit")
    a.add_argument("--save", default=None)
    a.add_argument("--predict", default=None, help="npz from --save: apply the all-rows refit to the test components")
    a.add_argument("--test", nargs="*", default=[], help="NAME=RUN | NAME=rank:PATH[+PATH...] (test side, one per component)")
    a.add_argument("--out", default=None)
    args = a.parse_args()
    if args.predict:
        return predict(args)
    fit_and_save(args)


if __name__ == "__main__":
    main()
