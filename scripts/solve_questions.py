"""Reference solutions per question with a VLM (vLLM), verified against the answer key.

For each unique question (train ∪ test ∪ unlabelled): sample up to N solutions (thinking on), keep the first
whose final answer matches the key (normalised string / numeric match), else keep the first and flag it.
Output: data/solutions/<tag>.csv  (question_id, solution, final, key, match, n_match)
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd

from k12.data import DATA, image_url, load, question_image_path

PROMPT = (
    "Solve this school maths question. Write a SHORT worked solution (at most 8 numbered steps, plain text, "
    "simple math notation), in the language of the question, then a last line 'Final answer: <answer>'.\n\n"
    "Grade: {grade}\nTopic: {topic}\nQuestion: {q}"
)


def delatex(t: str) -> str:
    for _ in range(3):
        t = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", t)
    t = t.replace("\\times", "*").replace("\\cdot", "*")
    return t


def norm_ans(s):
    s = delatex(str(s)).lower().strip()
    s = s.replace("−", "-").replace("–", "-").replace("$", "").replace("\\dfrac", "\\frac").replace(" ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\,", "").replace("\\", "")
    s = s.rstrip(".")
    return s


def num(s):
    s = norm_ans(s).replace(",", ".").replace("(", "").replace(")", "")
    m = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
    if len(m) != 1:
        return None
    try:
        if "/" in m[0]:
            a, b = m[0].split("/")
            return float(a) / float(b)
        return float(m[0])
    except Exception:
        return None


def matches(pred, key):
    a, b = norm_ans(pred), norm_ans(key)
    if not b:
        return False
    if a == b:
        return True
    x, y = num(pred), num(key)
    if x is not None and y is not None:
        return abs(x - y) < 1e-6 * max(1, abs(y))
    # substring only for non-numeric keys (e.g. expressions / options), never for numbers
    return y is None and len(b) >= 3 and b in a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--tries", type=int, default=4)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--quant", default=None)
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--only-unmatched", default=None, help="previous solutions csv: re-solve only rows with match==False")
    p.add_argument("--max-chars", type=int, default=1500)
    p.add_argument("--only-questions", default=None, help="csv(question_id): solve only these questions; default = all")
    a = p.parse_args()
    from vllm import LLM, SamplingParams

    qs = pd.concat([load(s) for s in ["train", "test", "unlabelled"]]).drop_duplicates("question_id")
    qs = qs[["question_id", "grade", "topic_name", "question_text", "question_image_file", "correct_answer"]].reset_index(drop=True)
    if a.only_questions:
        qs = qs[qs.question_id.isin(set(pd.read_csv(a.only_questions).question_id))].reset_index(drop=True)
    prev = None
    if a.only_unmatched:
        prev = pd.read_csv(a.only_unmatched, keep_default_na=False)
        bad = set(prev.question_id[(prev.match.astype(str) != "True") | (prev.solution.str.len() > a.max_chars)])
        qs = qs[qs.question_id.isin(bad)].reset_index(drop=True)
        print("re-solving", len(qs), "questions")
    kw = dict(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs, max_model_len=12288,
              limit_mm_per_prompt={"image": 1, "video": 0}, allowed_local_media_path=str(DATA))
    if a.quant:
        kw["quantization"] = a.quant
    llm = LLM(**kw)
    convs = []
    for _, r in qs.iterrows():
        content = [{"type": "text", "text": PROMPT.format(grade=r.grade, topic=r.topic_name, q=r.question_text)}]
        qi = question_image_path(r.question_image_file) if isinstance(r.question_image_file, str) else None
        if qi is not None and qi.exists():
            content.append({"type": "image_url", "image_url": {"url": image_url(qi)}})
        convs.append([{"role": "user", "content": content}])
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, n=a.tries, seed=0)
    t0 = time.time()
    res = llm.chat(convs, sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    rows = []
    for (_, r), o in zip(qs.iterrows(), res):
        cands = []
        for c in o.outputs:
            txt = c.text
            if "</think>" in txt:
                txt = txt.split("</think>", 1)[1]
            txt = txt.strip()
            m = re.findall(r"(?im)^\s*final answer\s*[:：]\s*(.+)$", txt)
            fin = m[-1].strip() if m else ""
            cands.append((txt, fin, matches(fin, r.correct_answer)))
        ok = [c for c in cands if c[2] and len(c[0]) <= a.max_chars]
        best = ok[0] if ok else min(cands, key=lambda c: (not c[2], len(c[0])))
        rows.append({"question_id": r.question_id, "solution": best[0], "final": best[1], "key": r.correct_answer,
                     "match": best[2], "n_match": sum(c[2] for c in cands)})
    out = DATA / "solutions"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    if prev is not None:  # merge: keep previous rows for questions not re-solved, or where the new one is not better
        prev = prev.set_index("question_id")
        for _, r in df.iterrows():
            old = prev.loc[r.question_id]
            if r.match or not (str(old.match) == "True"):
                prev.loc[r.question_id, ["solution", "final", "key", "match", "n_match"]] = [r.solution, r.final, r.key, r.match, r.n_match]
        df = prev.reset_index()
    df["solution"] = df.solution.astype(str).str.slice(0, a.max_chars)
    df.to_csv(out / f"{a.tag}.csv", index=False)
    print(f"{len(df)} questions, key-matched {df.match.mean():.3f}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
