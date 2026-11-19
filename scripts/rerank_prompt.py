"""Candidate-conditioned prompt pieces for the listwise reranker (used by zeroshot_diagnose.py --cands).

Candidates file (make_rerank_cands.py / make_cands_from_stack.py): trace_id, codes = the stack's top-K codes. The prompt lists them
in ALPHABETICAL order (NONE first) so the stack's order does not anchor the reranker. Per candidate: taxonomy name + description,
the convention rules that mention the code, and up to N annotated examples of that code from OTHER folds (training rows) or any
training row (test rows), preferring the same question, then the same topic: transcription excerpt + one-line annotator note.
The convention rules are fold-aware: training rows of folds 0-1 and all test rows use the sheet written from folds 2-4; training
rows of folds 2-4 use that rule list re-checked against folds 0-1 alone (only the choice of code pairs comes from folds 2-4).
"""
import json
import re
import zlib

import pandas as pd

from k12.data import folds, load, taxonomy

SHEETS = {"f234": "conventions/sheet_folds234.txt", "f01": "conventions/sheet_folds01.txt"}

RR_GUIDE = (
    "Rank the candidate codes for this student's working.\n"
    "1. Work out the correct solution path.\n"
    "2. Read the student's steps, find the errors, and decide which one is the ROOT (the major error the rest of the working "
    "follows from; NONE if there is no major error).\n"
    "3. Compare the candidates using their definitions, the conventions and the examples; the annotators' root code is very "
    "likely one of them.\n"
    "Finish with exactly one line: 'ROOT: <code>, <code>, ...' listing ALL {k} candidate codes, most likely first."
)


def _rules(path):
    out = {}
    for line in open(path).read().splitlines():
        line = line.strip()
        if not re.match(r"^\d+\.", line):
            continue
        for c in set(re.findall(r"\b[A-Z][A-Za-z]{1,3}-\d\d\b", line)) | ({"NONE"} if "NONE" in line else set()):
            out.setdefault(c, []).append(re.sub(r"^\d+\.\s*", "", line))
    return out


def _note(r, n=160):
    try:
        ms = json.loads(r.misconceptions)
        txt = next((m.get("text", "") for m in ms if m.get("root")), "")
    except Exception:
        txt = ""
    return " ".join(str(txt).split())[:n]


def make_cand_block(split, cands_csv, n_examples=1, use_rules=True, note_chars=160):
    """Returns (block(row) -> candidate text, {trace_id: [codes]})."""
    cands = pd.read_csv(cands_csv, keep_default_na=False)
    cmap = dict(zip(cands.trace_id, cands.codes.str.split()))
    tx = taxonomy().set_index("id")
    rules = {k: _rules(p) for k, p in SHEETS.items()} if use_rules else {}
    tr = load("train")
    fo = dict(zip(folds().trace_id, folds().fold))
    tr = tr.assign(fold=tr.trace_id.map(fo))
    tr = tr[tr.fold >= 0]
    by_code = {c: g for c, g in tr.groupby("root_code")}

    def example(code, r, k):
        out = []
        g = by_code.get(code)
        if g is None or code == "NONE":
            return out
        g = g[g.trace_id != r.trace_id]
        if split == "train":
            g = g[g.fold != k]
        if g.empty:
            return out
        pri = (g.question_id != r.question_id).astype(int) * 2 + (g.topic_name != r.topic_name).astype(int)
        g = g.assign(_p=pri.values, _h=[zlib.crc32((r.trace_id + t).encode()) for t in g.trace_id]).sort_values(["_p", "_h"])
        for _, e in g.head(n_examples).iterrows():
            ex = " ".join(str(e.transcription_steps).split())[:300]
            out.append(f"    Example ({'same question' if e._p == 0 else 'same topic' if e._p == 1 else 'other question'}): "
                       f"{ex} → note: {_note(e, note_chars)}")
        return out

    def block(r):
        codes = cmap.get(r.trace_id)
        if not codes:
            raise KeyError(f"no candidates for {r.trace_id}")
        k = fo.get(r.trace_id, -1) if split == "train" else -1
        sheet = "f01" if (split == "train" and k in (2, 3, 4)) else "f234"
        order = sorted(codes, key=lambda x: (x != "NONE", x))
        lines = [f"Candidate root codes for this student ({len(codes)}; alphabetical order, not a ranking):"]
        for c in order:
            if c == "NONE":
                lines.append("- NONE: no major misconception (correct, or only minor slips)")
            else:
                lines.append(f"- {c}: {tx.loc[c, 'name']} — {tx.loc[c, 'description']}")
            for rule in (rules.get(sheet, {}).get(c, []) if use_rules else []):
                lines.append(f"    Convention: {rule}")
            lines += example(c, r, k)
        return "\n".join(lines)

    return block, cmap
