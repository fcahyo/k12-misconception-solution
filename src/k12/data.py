"""Paths, CSV loading, fixed label space and CV folds."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"
IMG = DATA / "images"  # images/{train,test,unlabelled,questions}
RUNS = ROOT / "runs"

LABELS_FILE = DATA / "labels.json"
FOLDS_FILE = DATA / "folds.csv"
N_FOLDS = 5
FOLD_SEED = 42


def load(split: str) -> pd.DataFrame:
    """split in {train, test, unlabelled}: the competition CSV from data/raw."""
    df = pd.read_csv(RAW / f"{split}.csv", keep_default_na=False, na_values=[""])
    for c in ("question_image_file",):
        if c in df:
            df[c] = df[c].where(df[c].notna(), None)
    if "transcription_steps" in df:  # strip annotator judgments that name the error
        df["transcription_raw"] = df["transcription_steps"]
        df["transcription_steps"] = df["transcription_steps"].map(clean_transcription)
    return df


def taxonomy() -> pd.DataFrame:
    return pd.read_csv(RAW / "taxonomy_v1.csv")


def build_labels() -> list[str]:
    """Label space = NONE + taxonomy codes that occur in train (host: every test answer code has >=1 train example).
    Order: NONE first, then taxonomy file order. Written once to data/labels.json."""
    tr = load("train")
    tx = taxonomy()
    present = set(tr.root_code)
    labels = ["NONE"] + [c for c in tx.id if c in present]
    missing = present - set(labels)
    assert not missing, missing
    LABELS_FILE.write_text(json.dumps(labels, indent=1))
    return labels


def labels() -> list[str]:
    return json.loads(LABELS_FILE.read_text())


def build_folds() -> pd.DataFrame:
    """StratifiedKFold over rows on root_code (questions shared across folds, mirroring test).
    Singleton codes get fold=-1: always in training, never scored (test codes always have >=1 train example)."""
    from sklearn.model_selection import StratifiedKFold

    tr = load("train")
    vc = tr.root_code.value_counts()
    single = tr.root_code.map(vc).eq(1)
    folds = np.full(len(tr), -1, dtype=int)
    idx = np.where(~single)[0]
    y = tr.root_code.values[idx]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, (_, va) in enumerate(skf.split(idx, y)):
            folds[idx[va]] = k
    out = pd.DataFrame({"trace_id": tr.trace_id, "fold": folds, "root_code": tr.root_code})
    # every label in a val fold must appear in that fold's train side
    for k in range(N_FOLDS):
        va = set(out.root_code[out.fold == k])
        trn = set(out.root_code[out.fold != k])
        assert va <= trn, (k, va - trn)
    out.to_csv(FOLDS_FILE, index=False)
    return out


def folds() -> pd.DataFrame:
    return pd.read_csv(FOLDS_FILE)


def image_path(split: str, image_file: str) -> Path:
    return IMG / split / image_file


def question_image_path(name) -> Path | None:
    if not isinstance(name, str) or not name.strip():
        return None  # NaN / None / empty -> no diagram
    return IMG / "questions" / name


def image_url(path, max_pixels: int = 1024 * 1024, min_pixels: int = 0) -> str:
    """URL for a vLLM chat message: file:// if within [min_pixels, max_pixels], else a resized JPEG data URL
    (downscaled above max_pixels; upscaled below min_pixels, e.g. to give small handwriting more vision tokens)."""
    import base64
    import io

    from PIL import Image

    p = str(path)
    with Image.open(p) as im:
        w, h = im.size
        if min_pixels and w * h < min_pixels:
            s = (min_pixels / (w * h)) ** 0.5
        elif w * h <= max_pixels:
            return "file://" + p
        else:
            s = (max_pixels / (w * h)) ** 0.5
        im = im.convert("RGB").resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_JUDGE_WORDS = re.compile(r"\b(correct(ly)?|incorrect(ly)?|wrong|should|mistake|error|misconcep\w*|answered|aswered)\b", re.I)
_OBSERVE = re.compile(r"self-correct|correct(ed|ion)\b|overwrit|crossed|strike|scribble", re.I)


def _judgment(text: str) -> bool:
    return bool(_JUDGE_WORDS.search(text)) and not _OBSERVE.search(text)


def clean_transcription(t) -> str:
    """Remove annotator judgments/leaks from a gold transcription; keep observational notes ([crossed out] ...)."""
    if not isinstance(t, str):
        return ""
    t = re.sub(r"<\s*miscon.*?(</\s*miscon[^>]*>|$)", " ", t, flags=re.S | re.I)  # xml-like annotation blocks
    lines = []
    for ln in t.split("\n"):
        if re.search(r"(category\s*=|severity\s*=|root\s*=)", ln, re.I):
            continue
        is_line = re.match(r"\s*L\d+\s*\|", ln) is not None
        if not is_line and _judgment(ln):  # free-standing annotator sentence
            continue
        ln = re.sub(r"\[[^\]]*\]", lambda m: "" if _judgment(m.group(0)) else m.group(0), ln).rstrip()
        if re.fullmatch(r"\s*L\d+\s*\|\s*", ln):
            continue
        lines.append(ln)
    return "\n".join(l for l in lines if l.strip())
