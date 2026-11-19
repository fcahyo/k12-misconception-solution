"""Validate a prediction file: the columns of the competition's sample_submission.csv, one row per row of data/raw/test.csv
(which may be a new test set) and three distinct valid codes per row; optionally compare it with a reference file.
Usage: check_predictions.py NEW.csv [REF.csv]"""
import sys

import pandas as pd

from k12.data import RAW, labels, load

new = pd.read_csv(sys.argv[1], keep_default_na=False); te = load("test"); L = set(labels())
cols = list(pd.read_csv(RAW / "sample_submission.csv", nrows=0).columns)
assert list(new.columns) == cols, (list(new.columns), cols)
assert len(new) == len(te) and new.trace_id.is_unique and set(new.trace_id) == set(te.trace_id), "ids differ from data/raw/test.csv"
col = new.columns[1]; parts = new[col].str.split()
assert all(len(p) == 3 and len(set(p)) == 3 and set(p) <= L for p in parts), "each row needs 3 distinct valid codes"
t1 = parts.str[0]
print(f"{sys.argv[1]}: OK — {len(new)} rows, 3 distinct valid codes each; first answer NONE {(t1 == 'NONE').mean():.3f}; distinct first answers {t1.nunique()}")
if len(sys.argv) > 2:
    ref = pd.read_csv(sys.argv[2], keep_default_na=False).set_index("trace_id")[col].reindex(new.trace_id).str.split()
    r1 = ref.str[0].values
    print(f"vs {sys.argv[2]}: same first answer {(t1.values == r1).mean():.3f}; same top-3 set {sum(set(a) == set(b) for a, b in zip(parts, ref)) / len(new):.3f}; identical string {(new[col].values == ref.str.join(' ').values).mean():.3f}; ref first answer NONE {(r1 == 'NONE').mean():.3f}")
