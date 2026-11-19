"""Refuse test-side outputs that were computed for another test set (README, 'Running on a new test set').

A stage is skipped when its output exists, so outputs computed for one data/raw/test.csv would otherwise be reused for the rows of
another. A test set is identified by its trace_ids in file order; .test_set records the one that the test-side outputs in this
checkout belong to. Saved outputs without that record (archive 2) belong to the competition's test set. Nothing is deleted.
Usage: PYTHONPATH=src python scripts/check_test_set.py
"""
import hashlib
import sys

import pandas as pd

from k12.data import RAW, ROOT

COMPETITION = "045ee3e601136b1a3019f9834e0ba487"  # md5 of the competition test.csv's trace_ids, one per line, in file order
TEST_SIDE = ["data/trans/*_test.csv", "data/solutions/*.csv", "data/rerank/*", "outputs/*", "runs/read_*/test",
             "runs/*/fold*/test_probs.npy", "runs/*/fold*/pred_test*", "runs/*/test.npy", "runs/*_test/rank.npy",
             "runs/zsnt_pred/*_test", "runs/zst_pred/*test*"]
RECORD = ROOT / ".test_set"


def main():
    ids = pd.read_csv(RAW / "test.csv", usecols=["trace_id"], dtype=str, keep_default_na=False).trace_id
    current = hashlib.md5("\n".join(ids).encode()).hexdigest()
    found = sorted({str(p.relative_to(ROOT)) for pattern in TEST_SIDE for p in ROOT.glob(pattern)})
    belongs_to = RECORD.read_text().split()[0] if RECORD.exists() else COMPETITION
    if found and belongs_to != current:
        print("error: the test-side outputs below were computed for another test set than data/raw/test.csv. A stage is skipped "
              "when its output exists, so they would be reused for the wrong rows. For a new test set, start from a fresh checkout "
              "with archive 1 only (README, 'Running on a new test set'), or move these away first:\n  " + "\n  ".join(found),
              file=sys.stderr)
        return 1
    RECORD.write_text(current + "\n")
    print(f"test set: {len(ids)} rows" + ("; the saved test-side outputs belong to it" if found else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
