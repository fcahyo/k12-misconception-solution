"""Merge reader shard CSVs (trace_id, transcription) into data/trans/<tag>_<split>.csv; report coverage.

Usage: merge_shards.py <shard_dir> <tag> <split>
"""
import glob
import sys

import pandas as pd

from k12.data import DATA, load


def main():
    d, tag, split = sys.argv[1:4]
    fs = sorted(glob.glob(f"{d}/shard_*.csv"))
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in fs], ignore_index=True).drop_duplicates("trace_id")
    ids = set(load(split).trace_id)
    out = DATA / "trans"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"{tag}_{split}.csv", index=False)
    print(f"{len(df)} rows; coverage of {split}: {len(ids & set(df.trace_id))}/{len(ids)}; empty: {(df.transcription.str.len()==0).sum()}")


if __name__ == "__main__":
    main()
