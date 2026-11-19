# Misconception Detection in K-12 Student Math Workings: read, reason, rerank

**TL;DR.** A 27B vision-language model reads each handwritten working, reasoning models diagnose it zero-shot, a stack combines
their answers, and Qwen3.8-Flash-Next reranks the stack's short list of codes with annotated examples (diagram in the README). Two
models were developed, differing only in two settings of the last step: **Model 1 (Kaggle entry 58446106) reaches CV 0.8274
(public 0.83050), Model 2 (entry 58446518) CV 0.8245 (public 0.83033)**. All models are open-weight and run on one 4 × L40S
machine; no test row was labelled by hand. A *code* is a misconception (74 labels: NONE and the 73 taxonomy codes seen in train);
*CV* is MAP@3 in 5-fold cross-validation on 1,487 training rows.

## Approach

- **Reader** (Qwen3.8-27B) transcribes each working with and without the answer key (with it, it sometimes "corrected" wrong
  answers). **Solver** (27B, thinking): per question, the first solution that matches the answer key, else the first attempt
  (15%).
- **Students** (two Qwen3.5-9B LoRA models) read the transcription or the photo and score all 74 labels. **Teachers** (two 27B
  with thinking, one Flash-Next) see the photo, a transcription, the solution and the full taxonomy, and name three codes; a
  no-thinking 27B pass feeds stacks 0 and 1 only.
- **Stacks** are small models fitted on the training labels. **Rerankers** (Flash-Next; round 2 at xhigh effort, two samples)
  order a stack's short list, seeing each candidate's definition, the conventions that mention it and annotated examples (one,
  then up to five). The conventions are short hand-written rules from folds 2–4's notes, used for folds 0–1 and test; folds 2–4
  get them re-checked on folds 0–1 (code pairs still from folds 2–4).
- **Final stack:** learned weights on the students' and reranker 1's log-probabilities, a learned score per teacher and
  reranker-2 rank, a NONE bias and per-code biases. **Share matching** then shifts each code's average predicted share over the
  test set to its share in train. **Model 1 / Model 2** give per-code biases to codes with at least 25 / 40 training rows
  (13 / 7 codes) and own shifts to labels with at least 5 / 2 rows (51 / 61; the rest share one, and NOT-01, the root of one
  training row but often ranked second or third, always gets its own).

## Validation and results

| step, in order | CV | Δ | public |
|---|---|---|---|
| 2 students + 4 zero-shot teachers, stacked | 0.7911 | | 0.79716 ¹ |
| + reranker 1 | 0.8009 | +0.0099 | 0.80650 ¹ |
| + per-code biases | 0.8089 | +0.0080 | 0.81316 |
| teachers entered by rank instead of probability | 0.8127 | +0.0038 | 0.81616 |
| + reranker 2, one example per code | 0.8198 | +0.0071 | 0.82633 |
| + its second sample | 0.8211 | +0.0013 | 0.82416 |
| five examples per candidate | 0.8245 | +0.0034 | 0.83016 |
| without the no-thinking teacher ² | **0.8274** | +0.0029 | 0.83066 |
| **Model 1 / Model 2** | 0.8274 / 0.8245 | | **0.83050 / 0.83033** |

CV without share matching, public with it, except ¹. ² Found after the fact; chosen inside the folds, worth about +0.0015.

Every input to a stack is out-of-fold; each stack is fitted on four folds and applied to the fifth. As a CV figure is uncertain
by about ±0.016, a change was kept when resampling the same rows gave it at least a 90% chance of helping and at least 3 of 5
folds improved; five examples per candidate, reranker 2's second sample and the two model thresholds were kept below that bar.
**Share matching is the one choice informed by the public leaderboard:** neutral on CV, it added 0.0025–0.0075 public in six
pairs of otherwise identical uploads.

## What worked, what did not, and the unlabelled set

- **Reasoning beats fine-tuning:** alone, the zero-shot 27B teachers (0.764–0.766) beat the LoRA students (0.699).
- **Comparing neighbours:** reranking adds 0.0217 CV, over half of all gains after the first stack; reranker 2 is the strongest
  part (0.798 alone).
- **Code preferences:** models over-predict some codes (CAL-01) and under-predict others (FRM-01); biases and share matching help.
- **The rest is describing the fault, not finding it:** 59% of the 0.8127 stack's loss came from rows whose gold code never
  appears on the question's other training rows; given the row's annotator note, the model names the code on 80% of them.
- **Did not work:** more fine-tuning, more compute per row (Flash-Next at full effort, extra samples, a 27B reranker, another
  model family) and more stack features; of 109 recombinations, only dropping the no-thinking teacher helped.
- **The unlabelled set:** neither model uses `unlabelled.csv`; no attempt to use it was kept. Attempts included pseudo-labels
  from the text student to train a 27B student (+0.016 alone, +0.0014 in the stack), the no-thinking teacher's top three over all
  50,000
  rows as a per-question prior (+0.0015, chance level) and pool statistics (answer twins, final-answer frequency, question
  difficulty).

## Notes for verification

- **Test set as a batch:** share matching and reranker 2's candidate lists look at the other test rows' predictions, without test
  labels or human input; every other step reads one row at a time. Flagged for the hosts to judge.
- **Reproducibility:** from the saved outputs, both prediction files and the five stack fits rebuild bit for bit on CPU (checked
  19 November). The model stages sample, so a rerun repeats the procedure, not the tokens; expect it within about 0.01, more
  likely below. `run_inference.sh` ran end to end on a 10-row synthetic test set, not on the full one.
- **Compute and models:** 4 × L40S (48 GB), 32 cores, 125 GB RAM; about 100 GPU-hours to predict, an estimated 150 from scratch.
  Qwen3.5-9B and Qwen3.8-27B (Apache-2.0), Qwen3.8-Flash-Next-NVFP4 (NVIDIA Open Model License); Step-3.7-Flash-NVFP4 (Apache-2.0)
  in development only; no external datasets. Links and base-model licences: README.
