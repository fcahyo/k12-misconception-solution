"""Run directories runs/<id>/: reading a run's probabilities for the stack, and writing a run (save_run).

A run holds per-fold model outputs or run-level arrays (74 = the label space of data/labels.json):
  fold*/val_rows.npy, fold*/val_probs.npy   each fold model's validation rows and their probabilities
  fold*/test_probs.npy, full/test_probs.npy  each fold model's (and an all-rows model's) test probabilities
  oof.npy, oof_partial.npy                   (1500, 74) out-of-fold probabilities in train.csv row order (NaN = not covered)
  test.npy                                   (3498, 74) probabilities in test.csv row order
save_run writes oof.npy, test.npy, predictions.csv, config.json and metrics.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from .data import RUNS, folds, labels, load
from .metric import report, top3_strings


def train_targets() -> tuple[np.ndarray, np.ndarray]:
    """Returns (y index array for train rows, mask of rows that are scored in OOF)."""
    L = labels()
    idx = {c: i for i, c in enumerate(L)}
    tr = load("train")
    f = folds()
    assert (f.trace_id.values == tr.trace_id.values).all()
    y = tr.root_code.map(idx).values.astype(int)
    return y, (f.fold.values >= 0)


def evaluate_oof(oof: np.ndarray) -> dict:
    y, m = train_targets()
    return report(oof[m], y[m])


def oof_probs(run: str, n: int, K: int) -> np.ndarray:
    """(n, K) out-of-fold probabilities of a run, in train.csv row order.

    Taken from the fold models' validation outputs when the run has them (rows that no fold model validated stay NaN),
    otherwise from runs/<run>/oof.npy, otherwise from runs/<run>/oof_partial.npy.
    """
    fold_files = sorted(glob.glob(str(RUNS / run / "fold*" / "val_probs.npy")))
    if fold_files:
        P = np.full((n, K), np.nan)
        for f in fold_files:
            P[np.load(Path(f).with_name("val_rows.npy"))] = np.load(f)
        return P
    if (RUNS / run / "oof.npy").exists():
        return np.load(RUNS / run / "oof.npy")
    return np.load(RUNS / run / "oof_partial.npy")


def test_probs(run: str) -> tuple[np.ndarray, int]:
    """Test probabilities of a run and the number of models behind them.

    The arithmetic mean over the run's models (fold*/test_probs.npy in sorted order, then full/test_probs.npy) when it has
    them, otherwise runs/<run>/test.npy.
    """
    model_files = sorted(glob.glob(str(RUNS / run / "fold*" / "test_probs.npy"))) + sorted(
        glob.glob(str(RUNS / run / "full" / "test_probs.npy"))
    )
    if model_files:
        return np.mean([np.load(f) for f in model_files], 0), len(model_files)
    path = RUNS / run / "test.npy"
    assert path.exists(), f"no test probabilities for run {run}"
    return np.load(path), 1


def save_run(run_id: str, oof: np.ndarray | None, test: np.ndarray | None, config: dict) -> dict:
    """Write runs/<run_id>/: oof.npy, test.npy with predictions.csv (top-3 codes), config.json and metrics.json (OOF metrics)."""
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if oof is not None:
        np.save(d / "oof.npy", oof.astype(np.float32))
        metrics = evaluate_oof(oof)
    if test is not None:
        np.save(d / "test.npy", test.astype(np.float32))
        te = load("test")
        import pandas as pd

        pd.DataFrame({"trace_id": te.trace_id, "root_code": top3_strings(test, labels())}).to_csv(
            d / "predictions.csv", index=False
        )
    (d / "config.json").write_text(json.dumps(config, indent=1, default=str))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics
