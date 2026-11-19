"""Prompt construction shared by all models (VLM classifier, reader, text classifier).

The taxonomy block is a fixed shared prefix (prefix-cache friendly); per-row content comes after it.
"""
from __future__ import annotations

from functools import lru_cache

from .data import labels, taxonomy

SYSTEM = (
    "You are an experienced Indonesian mathematics teacher. You read photos of students' handwritten "
    "working (Bahasa Indonesia or English, LaTeX-style maths) and diagnose the ROOT misconception: the "
    "major error that the rest of the working follows from. Minor slips that do not change the reasoning "
    "or result are not roots. If the working has no major misconception, the answer is NONE."
)


@lru_cache(maxsize=4)
def taxonomy_block(style: str = "compact") -> str:
    """compact: 'CODE: name'; full: 'CODE: name — description'. Only codes in the label space."""
    tx = taxonomy().set_index("id")
    L = labels()
    lines = ["NONE: no major misconception (correct, or only minor slips)"]
    for c in L[1:]:
        r = tx.loc[c]
        if style == "full":
            lines.append(f"{c}: {r['name']} — {r['description']}")
        else:
            lines.append(f"{c}: {r['name']}")
    return "Misconception codes:\n" + "\n".join(lines)


def question_block(row) -> str:
    parts = [
        f"Grade: {row['grade']}",
        f"Topic: {row['topic_name']}",
        f"Question: {row['question_text']}",
        f"Answer key: {row['correct_answer']}",
    ]
    return "\n".join(parts)


def classify_messages(row, *, tax_style="compact", transcription: str | None = None,
                      examples: list[tuple[str, str]] | None = None, with_image=True,
                      with_question_image=True, image_path=None, question_image_path=None,
                      solution: str | None = None) -> list[dict]:
    """Chat messages for the classifier. The model's last prompt token is used for the label head."""
    content = []
    content.append({"type": "text", "text": taxonomy_block(tax_style) + "\n\n" + question_block(row)})
    if solution:
        content.append({"type": "text", "text": "\n\nReference solution (teacher):\n" + solution.strip()})
    if with_question_image and question_image_path is not None:
        content.append({"type": "text", "text": "\nQuestion diagram:"})
        content.append({"type": "image", "image": str(question_image_path)})
    if examples:
        ex = "\n\n".join(f"Example (same question, other student):\n{t}\nRoot: {c}" for t, c in examples)
        content.append({"type": "text", "text": "\n\n" + ex})
    if with_image and image_path is not None:
        content.append({"type": "text", "text": "\n\nStudent's working (photo):"})
        content.append({"type": "image", "image": str(image_path)})
    if transcription is not None:
        content.append({"type": "text", "text": "\n\nTranscription of the student's working:\n" + transcription})
    content.append({"type": "text", "text": "\n\nWhat is the root misconception code?"})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


READER_INSTR = (
    "Transcribe the student's handwritten working line by line, verbatim, as 'L1 | ...', 'L2 | ...'. "
    "Keep the student's own notation, numbers and mistakes exactly; do not correct anything and do not solve. "
    "Use PLAIN-TEXT maths, NOT LaTeX: write fractions as a/b, roots as √5 or ∛a, powers as x² or 5^(x+3), "
    "multiplication as ×, minus as −, and units as written. Write crossed-out text as [crossed out: ...] and "
    "unreadable parts as [unreadable]. Do not copy the printed question unless the student wrote it.\n\n"
    "Format examples (other students):\n"
    "L1 | 8a − 3a = 11 + 4\nL2 | 5a = 15\nL3 | a = 15/5 = 3\n\n"
    "L1 | f(2a−1) = (2a−1)² + 2(2a−1) − 3\nL2 | = 4a² + 1 + 4a − 2 − 3"
)


def reader_messages(row, image_path, question_image_path=None) -> list[dict]:
    content = [{"type": "text", "text": question_block(row) + "\n\n" + READER_INSTR + "\n\nPhoto:"},
               {"type": "image", "image": str(image_path)}]
    return [{"role": "user", "content": content}]


READER_INSTR_V3Q = (
    "Transcribe the student's handwritten working line by line, as 'L1 | ...', 'L2 | ...'. "
    "Transcribe exactly what is written, including errors and crossed-out text marked [crossed out: ...]. "
    "Never correct, complete or compute: copy wrong numbers and wrong steps exactly as written, and stop where the "
    "student stopped, even if the working is wrong or unfinished. Mark unreadable spans [illegible]. "
    "Use PLAIN-TEXT maths, NOT LaTeX: write fractions as a/b, roots as √5 or ∛a, powers as x² or 5^(x+3), "
    "multiplication as ×, minus as −, and units as written. Do not copy the printed question unless the student wrote it.\n\n"
    "Format examples (other students):\n"
    "L1 | 8a − 3a = 11 + 4\nL2 | 5a = 15\nL3 | a = 15/5 = 3\n\n"
    "L1 | f(2a−1) = (2a−1)² + 2(2a−1) − 3\nL2 | = 4a² + 1 + 4a − 2 − 3"
)


def reader_messages_v3q(row, image_path) -> list[dict]:
    """Reader v3q: the printed question as context, but no answer key and no reference solution
    (with the key in the prompt, the v2 reader tended to 'correct' wrong answers toward it)."""
    ctx = f"Grade: {row['grade']}\nTopic: {row['topic_name']}\nPrinted question (context only): {row['question_text']}"
    content = [{"type": "text", "text": ctx + "\n\n" + READER_INSTR_V3Q + "\n\nPhoto:"},
               {"type": "image", "image": str(image_path)}]
    return [{"role": "user", "content": content}]
