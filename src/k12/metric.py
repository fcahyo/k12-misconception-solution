"""MAP@3 (competition definition), diagnostics and paired bootstrap on OOF probability arrays."""
from __future__ import annotations

import numpy as np


def map3_lists(true: list[str], preds: list[list[str]]) -> np.ndarray:
    """Per-row score: 1/(rank) of first occurrence of the true code in the first 3 predictions; repeats skipped."""
    out = np.zeros(len(true))
    for i, (t, p) in enumerate(zip(true, preds)):
        seen = []
        for c in p:
            if c in seen:
                continue
            seen.append(c)
            if len(seen) > 3:
                break
            if c == t:
                out[i] = 1.0 / len(seen)
                break
    return out


def row_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """probs: (n, L) scores; y: (n,) int label index. Per-row MAP@3 using top-3 by score."""
    top = np.argsort(-probs, axis=1)[:, :3]
    s = np.zeros(len(y))
    for r in range(3):
        s += (top[:, r] == y) / (r + 1)
    return s


def bootstrap_ci(scores: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    b = scores[rng.integers(0, n, (n_boot, n))].mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def paired_bootstrap(sa: np.ndarray, sb: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired comparison of per-row scores of model A vs B on the same rows."""
    d = sa - sb
    rng = np.random.default_rng(seed)
    n = len(d)
    b = d[rng.integers(0, n, (n_boot, n))].mean(1)
    return {
        "delta": float(d.mean()),
        "ci95": (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))),
        "p_a_better": float((b > 0).mean()),
    }


def report(probs: np.ndarray, y: np.ndarray, none_idx: int = 0) -> dict:
    s = row_scores(probs, y)
    lo, hi = bootstrap_ci(s)
    top1 = probs.argmax(1)
    is_none = y == none_idx
    pred_none = top1 == none_idx
    tp = (pred_none & is_none).sum()
    return {
        "n": int(len(y)),
        "map3": float(s.mean()),
        "ci95": (lo, hi),
        "top1_acc": float((top1 == y).mean()),
        "map3_nonNONE": float(s[~is_none].mean()) if (~is_none).any() else float("nan"),
        "map3_NONE": float(s[is_none].mean()) if is_none.any() else float("nan"),
        "none_precision": float(tp / max(pred_none.sum(), 1)),
        "none_recall": float(tp / max(is_none.sum(), 1)),
        "pred_none_rate": float(pred_none.mean()),
    }


def top3_strings(probs: np.ndarray, labels: list[str]) -> list[str]:
    top = np.argsort(-probs, axis=1)[:, :3]
    return [" ".join(labels[j] for j in row) for row in top]
