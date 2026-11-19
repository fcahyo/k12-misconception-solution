"""Check the competition data layout and write the two fixed files every later stage reads.

Expected layout (repository root):
  data/raw/{train,test,unlabelled,taxonomy_v1,sample_submission}.csv
  data/images/{train,test,unlabelled,questions}/   one JPEG per row, named as in the CSVs' image columns
Writes data/labels.json (NONE + the taxonomy codes that occur in train, in taxonomy order) and data/folds.csv (5-fold
StratifiedKFold on root_code, seed 42; the 13 rows whose code occurs once get fold -1 and always stay in training).

Usage: PYTHONPATH=src python scripts/prepare_data.py [--check]
  --check  run every check, write nothing (the drivers call this when the two files exist already)
"""
import hashlib
import sys

from k12.data import DATA, build_folds, build_labels

REQUIRED = [f"raw/{f}.csv" for f in ("train", "test", "unlabelled", "taxonomy_v1", "sample_submission")] + \
           [f"images/{d}" for d in ("train", "test", "unlabelled", "questions")]
# The Kaggle download holds two copies of train/test/unlabelled.csv. The prompts include the topic name, and the solution was
# built with the copies at the top level of the download (English topic names). The copies under competition_data/ carry
# Indonesian topic names and would change every prompt.
INDONESIAN = {"train.csv": "ab31c7521bdaa58c83a82e6093adbdf3", "test.csv": "4f5c2ccc4ef1a5bdecc57ccc4fc981ae",
              "unlabelled.csv": "93d2009dc9e50c64906943f04dff1904"}


def main():
    missing = [f"data/{p}" for p in REQUIRED if not (DATA / p).exists()]
    if missing:
        print("missing competition files (see the README, 'Data'):\n  " + "\n  ".join(missing))
        return 1
    # vLLM loads a local image only if its real path (links resolved) lies inside data/ (--allowed-local-media-path), so the images
    # must sit inside data/: linking data/ itself is fine, linking a folder below it to somewhere else is not.
    data_real = DATA.resolve()
    for d in ("train", "test", "unlabelled", "questions"):
        first = next((p for p in sorted((DATA / "images" / d).iterdir()) if p.is_file()), None)
        if first is None:
            print(f"data/images/{d} holds no image files (README, 'Data')")
            return 1
        if data_real not in first.resolve().parents:
            print(f"data/images/{d} resolves to {first.resolve().parent}, outside data/: vLLM only loads images whose real path is "
                  "inside data/. Move the images inside data/ (or make data/ itself the link) (README, 'Data')")
            return 1
    wrong = [f for f, md5 in INDONESIAN.items() if hashlib.md5((DATA / "raw" / f).read_bytes()).hexdigest() == md5]
    if wrong:
        print(f"data/raw/{', data/raw/'.join(wrong)}: these are the copies with Indonesian topic names (competition_data/ in the "
              "Kaggle download); use the files at the top level of the download instead (README, 'Data')")
        return 1
    if "--check" in sys.argv[1:]:
        print("data layout OK")
        return 0
    labels = build_labels()
    folds = build_folds()
    print(f"wrote data/labels.json ({len(labels)} labels) and data/folds.csv ({(folds.fold >= 0).sum()} scored rows, "
          f"{(folds.fold < 0).sum()} always in training)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
