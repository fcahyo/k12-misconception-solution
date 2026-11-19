#!/usr/bin/env bash
# Every stage command of the pipeline, defined once; sourced by run_inference.sh and run_training.sh.
#
# Environment:
#   M9, M27, MFN   model directories of Qwen3.5-9B, Qwen3.8-27B and Qwen3.8-Flash-Next-NVFP4 (only the stages that load them need them)
#   GPUS           space-separated GPU ids (default "0 1 2 3"); Flash-Next runs tensor-parallel over all of them
#   VENV_VLLM      python of the vLLM environment   (default .venv-vllm/bin/python)
#   VENV_TRAIN     python of the training/CPU environment (default .venv-train/bin/python)
#   ROWS           optional csv with a trace_id column: a smoke run of the first test-side model stages on these rows only
#                  (run it in a separate copy of the repository: its outputs are not valid for a full run)
#   CPU_ONLY=1     refuse every model stage (rebuild from saved stage outputs; README, path (c))
#   DRY=1          print the commands instead of running them
# A stage is skipped when its output already exists, so a run can be resumed, and saved outputs can stand in for a stage.
# Ctrl-C or kill stops every running stage job, including the vLLM processes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
VPY="${VENV_VLLM:-$ROOT/.venv-vllm/bin/python}"
TPY="${VENV_TRAIN:-$ROOT/.venv-train/bin/python}"
# The environments' bin/ directories go on PATH: vLLM compiles kernels at start-up by calling `ninja` from its environment.
export PATH="$(dirname "$VPY"):$(dirname "$TPY"):$PATH"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
TP=${#GPU_LIST[@]}
LOG="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG"
# BLAS thread counts of the original run; with them the stack fits are reproduced bit for bit (other counts change the fitted
# weights in the sixth decimal, which in the checks changed no ranking and neither prediction file).
THREADS_EARLY="${THREADS_EARLY:-32}"
THREADS_FINAL="${THREADS_FINAL:-2}"
FINALS="58446106:25:5 58446518:40:2"   # Model 1, Model 2 (named by their Kaggle entry) : per-code bias threshold : share-matching own-offset threshold

say() { echo "[$(date '+%F %H:%M:%S')] $*"; }
dry() { [ "${DRY:-0}" = 1 ]; }
have() { ! dry && [ -e "$1" ]; }
# need STAGE VAR...: a model stage is about to run; its model variables must name directories, and CPU_ONLY=1 forbids it.
need() {
  local stage=$1 v; shift
  if [ "${CPU_ONLY:-0}" = 1 ]; then
    echo "error: CPU_ONLY=1, but the $stage stage has to run (its saved outputs are missing; README, path (c))" >&2; exit 1
  fi
  for v in "$@"; do [ -n "${!v:-}" ] || { echo "error: the $stage stage needs $v, the model directory (README, 'Models')" >&2; exit 2; }; done
}
cpu() { local t=$1; shift; if dry; then echo "OMP_NUM_THREADS=$t $TPY $*"; else OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t "$TPY" "$@"; fi; }
rows_for() { if [ "$1" = test ] && [ -n "${ROWS:-}" ]; then echo "--rows $ROWS"; fi; }

# ---------------------------------------------------------------- checks before any stage
for v in M9 M27 MFN; do  # a model variable that is set must name a directory (a typo would otherwise surface hours later)
  if [ -n "${!v:-}" ] && ! dry && [ ! -d "${!v}" ]; then echo "error: $v=${!v} is not a directory (README, 'Models')" >&2; exit 2; fi
done
if [ -n "${ROWS:-}" ]; then
  dry || echo "smoke run on the rows of $ROWS" > "$ROOT/.rows_run"
elif [ -e "$ROOT/.rows_run" ]; then
  echo "error: this checkout holds the outputs of a ROWS smoke run, which cover only a few test rows; run smoke tests in a" \
       "separate copy of the repository and start full runs from a checkout without .rows_run (README, 'Running the scripts')" >&2
  exit 1
fi

# ---------------------------------------------------------------- job control
# Every job runs in its own process group (setsid), so stopping the driver (Ctrl-C, kill) can stop the jobs and their vLLM
# processes as a whole; run_watchdog.sh stops its own process group when it is signalled.
CHILD_GROUPS=()
stop_jobs() {
  local g alive
  for g in "${CHILD_GROUPS[@]+"${CHILD_GROUPS[@]}"}"; do kill -TERM -- "-$g" 2>/dev/null || true; done
  for _ in $(seq 1 40); do
    alive=0
    for g in "${CHILD_GROUPS[@]+"${CHILD_GROUPS[@]}"}"; do kill -0 -- "-$g" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep 1
  done
  for g in "${CHILD_GROUPS[@]+"${CHILD_GROUPS[@]}"}"; do kill -KILL -- "-$g" 2>/dev/null || true; done
}
trap 'echo "stopping: ending the running stage jobs" >&2; stop_jobs; exit 130' INT TERM HUP

# One-GPU jobs ("name|command"), run in waves of one job per GPU; stops the run if any job fails.
gpu_jobs() {
  local n=${#GPU_LIST[@]} i=0 fail=0 job name cmd g pids=()
  for job in "$@"; do
    name=${job%%|*}; cmd=${job#*|}; g=${GPU_LIST[$((i % n))]}
    if dry; then echo "CUDA_VISIBLE_DEVICES=$g $cmd"; else
      say "start $name on GPU $g (log: $LOG/$name.log)"
      CUDA_VISIBLE_DEVICES=$g setsid bash -c "$cmd" > "$LOG/$name.log" 2>&1 &
      pids+=($!); CHILD_GROUPS+=($!)
    fi
    i=$((i + 1))
    if [ ${#pids[@]} -eq "$n" ]; then for p in "${pids[@]}"; do wait "$p" || fail=1; done; pids=(); fi
  done
  for p in "${pids[@]+"${pids[@]}"}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || { echo "error: a GPU job failed (logs in $LOG)" >&2; exit 1; }
}

# A Flash-Next pass on all GPUs, restarted by the watchdog if it stalls: flash OUT_DIR ARGS...
flash() {
  local out=$1; shift
  local cmd="bash scripts/run_watchdog.sh $out 3000 3 -- $VPY scripts/zeroshot_diagnose.py --model $MFN --tp $TP --input image --solutions data/solutions/q27_think_v2.csv --think 1 --temperature 1.0 --top-k 20 --gpu-mem 0.92 --kv-fp8 1 --out $out $*"
  if dry; then echo "CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_LIST[*]}") $cmd"; else
    say "start $(basename "$out") on GPUs ${GPU_LIST[*]} (log: $LOG/$(basename "$out").log)"
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_LIST[*]}")" setsid bash -c "$cmd" > "$LOG/$(basename "$out").log" 2>&1 &
    local p=$!; CHILD_GROUPS+=($p)
    wait "$p" || { echo "error: $out failed (log: $LOG/$(basename "$out").log)" >&2; exit 1; }
  fi
}

# ---------------------------------------------------------------- stages (split = train | test)
stage_reader() {  # line-by-line transcription, two prompt versions -> data/trans/zs27v{2,3q}_<split>.csv
  local s=$1
  if have "data/trans/zs27v2_$s.csv" && have "data/trans/zs27v3q_$s.csv"; then say "reader $s: present"; return; fi
  need "reader ($s)" M27
  local R="$VPY scripts/vllm_read.py --model $M27 --split $s --shard-size 500 --quant fp8 --gpu-mem 0.92 --max-model-len 6144 $(rows_for "$s")"
  gpu_jobs "read_v2_$s|$R --out runs/read_zs27v2/$s" "read_v3q_$s|$R --out runs/read_zs27v3q/$s --prompt v3q"
  cpu 4 scripts/merge_shards.py "runs/read_zs27v2/$s" zs27v2 "$s"
  cpu 4 scripts/merge_shards.py "runs/read_zs27v3q/$s" zs27v3q "$s"
}

stage_solutions() {  # one key-checked reference solution per question -> data/solutions/q27_think_{n2,v2}.csv
  if have data/solutions/q27_think_v2.csv; then say "solutions: present"; return; fi
  need "reference solutions" M27
  local q=""
  if [ -n "${ROWS:-}" ] && ! dry; then
    "$TPY" -c "import pandas as pd; from k12.data import load; t = load('test'); ids = set(pd.read_csv('$ROWS').trace_id); t[t.trace_id.isin(ids)][['question_id']].drop_duplicates().to_csv('$LOG/questions.csv', index=False)"
    q="--only-questions $LOG/questions.csv"
  fi
  local S="$VPY scripts/solve_questions.py --model $M27 --quant fp8 --gpu-mem 0.92"
  have data/solutions/q27_think_n2.csv || gpu_jobs "solve_n2|$S --tag q27_think_n2 --tries 2 --max-tokens 6144 $q"
  gpu_jobs "solve_v2|$S --tag q27_think_v2 --tries 4 --max-tokens 8192 --only-unmatched data/solutions/q27_think_n2.csv --max-chars 1500 $q"
}

stage_students_train() {  # two Qwen3.5-9B LoRA students, five folds each; every fold also predicts the test set
  local jobs=() k
  for k in 0 1 2 3 4; do
    have "runs/t9gen_zs_v1/fold$k/test_probs.npy" || jobs+=("t9_fold$k|$TPY scripts/train_vlm_gen.py --model $M9 --run t9gen_zs_v1 --fold $k --image 0 --qimg 1 --transcription data/trans/zs27v2_train.csv --test-transcription data/trans/zs27v2_test.csv --solutions data/solutions/q27_think_n2.csv --aux fac --aux-weight 0.3 --epochs 3 --lr 2e-4 --grad-accum 8 --eval-epochs 2 --predict-test 1 --save-adapter 1")
    have "runs/v9gen_img_v1/fold$k/test_probs.npy" || jobs+=("v9_fold$k|$TPY scripts/train_vlm_gen.py --model $M9 --run v9gen_img_v1 --fold $k --image 1 --qimg 1 --transcription none --solutions data/solutions/q27_think_n2.csv --aux fac,trans --aux-weight 0.3 --epochs 3 --lr 2e-4 --grad-accum 8 --eval-epochs 2 --predict-test 1 --save-adapter 1")
  done
  [ ${#jobs[@]} -eq 0 ] && { say "students: present"; return; }
  need "student training" M9
  gpu_jobs "${jobs[@]}"
}

stage_students_test() {  # the saved fold adapters on the test set -> runs/<student>/fold<k>/test_probs.npy
  local jobs=() k out=""
  [ -n "${ROWS:-}" ] && out="--out-name pred_test_rows"
  for k in 0 1 2 3 4; do
    { have "runs/t9gen_zs_v1/fold$k/test_probs.npy" && [ -z "${ROWS:-}" ]; } || jobs+=("t9_test_fold$k|$TPY scripts/predict_vlm_gen.py --model $M9 --run t9gen_zs_v1 --fold $k --split test --transcription data/trans/zs27v2_test.csv $(rows_for test) $out")
    { have "runs/v9gen_img_v1/fold$k/test_probs.npy" && [ -z "${ROWS:-}" ]; } || jobs+=("v9_test_fold$k|$TPY scripts/predict_vlm_gen.py --model $M9 --run v9gen_img_v1 --fold $k --split test $(rows_for test) $out")
  done
  [ ${#jobs[@]} -eq 0 ] && { say "students on test: present"; return; }
  need "students (test)" M9
  gpu_jobs "${jobs[@]}"
}

stage_teachers() {  # three Qwen3.8-27B zero-shot teachers -> runs/{zsnt_pred_v1, zst_train_v1 | zst_pred_v1, zst_v3q_all | zst_v3q_test}
  local s=$1 v2 v3 d2 d3
  if [ "$s" = train ]; then v2=zst_train_v1; v3=zst_v3q_all; d2=zst_train_v2; d3=zst_v3q_all; else v2=zst_pred_v1; v3=zst_v3q_test; d2=zst_test; d3=zst_v3q_test; fi
  local f; [ "$s" = train ] && f=oof_partial.npy || f=test.npy
  if have "runs/zsnt_pred_v1/$f" && have "runs/$v2/$f" && have "runs/$v3/$f"; then say "27B teachers $s: present"; return; fi
  need "27B teachers ($s)" M27
  local Z="$VPY scripts/zeroshot_diagnose.py --model $M27 --quant fp8 --kv-fp8 1 --solutions data/solutions/q27_think_v2.csv --split $s $(rows_for "$s")"
  gpu_jobs "zsnt_$s|$Z --input data/trans/zs27v2_$s.csv --think 0 --max-tokens 1024 --shard-size 500 --out runs/zsnt_pred/zsnt_$s" \
           "v2_$s|$Z --input image --trans-csv data/trans/zs27v2_$s.csv --think 1 --max-tokens 6144 --shard-size 250 --out runs/zst_pred/$d2" \
           "v3_$s|$Z --input image --trans-csv data/trans/zs27v3q_$s.csv --think 1 --max-tokens 6144 --seed 0 --shard-size 250 --force-rc 0 --out runs/zst_pred/$d3"
  cpu 4 scripts/zs_to_probs.py "runs/zsnt_pred/zsnt_$s" "$s" zsnt_pred_v1
  cpu 4 scripts/zs_to_probs.py "runs/zst_pred/$d2" "$s" "$v2"
  cpu 4 scripts/zs_to_probs.py "runs/zst_pred/$d3" "$s" "$v3"
}

stage_flash_teacher() {  # Flash-Next zero-shot teacher, medium reasoning effort -> runs/d_fnext_{all|test}
  local s=$1 id; [ "$s" = train ] && id=d_fnext_all || id=d_fnext_test
  local f; [ "$s" = train ] && f=oof_partial.npy || f=test.npy
  if have "runs/$id/$f"; then say "Flash-Next teacher $s: present"; return; fi
  need "Flash-Next teacher ($s)" MFN
  flash "runs/zst_pred/$id" --split "$s" --trans-csv "data/trans/zs27v3q_$s.csv" --reasoning-effort medium --max-tokens 12288 --max-model-len 20480 --seed 0 --shard-size 300 $(rows_for "$s")
  cpu 4 scripts/zs_to_probs.py "runs/zst_pred/$id" "$s" "$id"
}

stage_round0_fit() {  # round-0 stack of the two students and the four teachers (cross-fitted) -> fits/round0.npz
  if have fits/round0.npz; then say "round-0 fit: present"; return; fi
  mkdir -p fits
  cpu "$THREADS_EARLY" scripts/fit_blend.py --c t9=run:t9gen_zs_v1 v9=run:v9gen_img_v1 zsnt=run:zsnt_pred_v1 v2=npy:runs/zst_train_v1/oof.npy \
      v3=npy:runs/zst_v3q_all/oof_partial.npy fn=npy:runs/d_fnext_all/oof_partial.npy \
      --ref t9:.2125,v9:.2125,zsnt:.075,v2:.166667,v3:.166667,fn:.166667 --folds 0 1 2 3 4 --save fits/round0.npz
}

stage_top8() {  # top-8 candidate codes of the round-0 stack (train: cross-fitted, test: refit) -> data/rerank/top8_*
  if have data/rerank/top8_cands.csv; then say "top-8 candidates: present"; return; fi
  cpu "$THREADS_EARLY" scripts/make_rerank_cands.py fits/round0.npz --test-runs t9gen_zs_v1 v9gen_img_v1 zsnt_pred_v1 zst_pred_v1 zst_v3q_test d_fnext_test \
      --k 8 --out data/rerank/top8
}

stage_rerank1() {  # first-round listwise reranker over the top 8 (Flash-Next, medium effort) -> runs/rrfn_{all|test}
  local s=$1 id d; if [ "$s" = train ]; then id=rrfn_all; d=rrfn_f05; else id=rrfn_test; d=rrfn_test; fi
  local f; [ "$s" = train ] && f=oof_partial.npy || f=test.npy
  if have "runs/$id/$f"; then say "first reranker $s: present"; return; fi
  need "first reranker ($s)" MFN
  flash "runs/zst_pred/$d" --split "$s" --trans-csv "data/trans/zs27v3q_$s.csv" --cands data/rerank/top8_cands.csv --reasoning-effort medium \
      --max-tokens 4096 --max-model-len 16384 --seed 0 --shard-size 300 --save-text 1 $(rows_for "$s")
  cpu 4 scripts/rerank_to_probs.py "runs/zst_pred/$d" "$s" "$id" --cands data/rerank/top8_cands.csv --stack-P "data/rerank/top8_P_$s.npy"
}

stage_round1_fit() {  # round-1 stack (+ first reranker, per-code biases, rank-coded teachers) and its train rows share-matched
  if have fits/round1_matched.npz; then say "round-1 fit: present"; return; fi
  mkdir -p fits
  cpu "$THREADS_EARLY" scripts/fit_stack.py --c t9=run:t9gen_zs_v1 v9=run:v9gen_img_v1 zsnt=run:zsnt_pred_v1 v2=npy:runs/zst_train_v1/oof.npy \
      v3=npy:runs/zst_v3q_all/oof_partial.npy fn=npy:runs/d_fnext_all/oof_partial.npy rrfn=npy:runs/rrfn_all/oof_partial.npy \
      --folds 0 1 2 3 4 --code-bias 25 --rank-shape v2 v3 fn --save fits/round1.npz
  cpu "$THREADS_EARLY" scripts/marginal_match.py oof fits/round1.npz --pooled 1 --save fits/round1_matched.npz
}

stage_top6() {  # round-1 stack on test, share-matched over the test set; top-6 candidates -> data/rerank/top6_*
  if have data/rerank/top6_cands.csv; then say "top-6 candidates: present"; return; fi
  mkdir -p outputs
  cpu "$THREADS_EARLY" scripts/fit_stack.py --predict fits/round1.npz --test t9=t9gen_zs_v1 v9=v9gen_img_v1 zsnt=zsnt_pred_v1 v2=zst_pred_v1 \
      v3=zst_v3q_test fn=d_fnext_test rrfn=rrfn_test --out outputs/round1_test.csv
  cpu "$THREADS_EARLY" scripts/marginal_match.py test outputs/round1_test.npy --out outputs/round1_test_matched.csv
  cpu "$THREADS_EARLY" scripts/make_cands_from_stack.py fits/round1_matched.npz outputs/round1_test_matched.npy --k 6 --out data/rerank/top6
}

stage_rerank2() {  # second-round reranker over the top 6 (Flash-Next, full effort, five examples per candidate), two samples
  local s=$1 seed id d
  for seed in 0 1; do
    if [ "$s" = train ]; then id=$([ $seed = 0 ] && echo rr2x5_all || echo rr2x5_s1_all); d=$([ $seed = 0 ] && echo rr2x5_f05 || echo rr2x5_s1_f05)
    else id=$([ $seed = 0 ] && echo rr2x5_test || echo rr2x5_s1_test); d=$id; fi
    if have "runs/$id/rank.npy"; then say "second reranker $s seed $seed: present"; continue; fi
    need "second reranker ($s, seed $seed)" MFN
    flash "runs/zst_pred/$d" --split "$s" --trans-csv "data/trans/zs27v3q_$s.csv" --cands data/rerank/top6_cands.csv --cand-examples 5 \
        --cand-note-chars 400 --reasoning-effort xhigh --max-tokens 12288 --max-model-len 32768 --seed $seed --shard-size 150 --save-text 1 $(rows_for "$s")
    cpu 4 scripts/rerank_to_ranks.py "runs/zst_pred/$d" "$s" "$id" --cands data/rerank/top6_cands.csv
  done
}

FINAL_C="t9=run:t9gen_zs_v1 v9=run:v9gen_img_v1 v2=npy:runs/zst_train_v1/oof.npy v3=npy:runs/zst_v3q_all/oof_partial.npy fn=npy:runs/d_fnext_all/oof_partial.npy rrfn=npy:runs/rrfn_all/oof_partial.npy rr2x=rank:runs/rr2x5_all/rank.npy+runs/rr2x5_s1_all/rank.npy"
FINAL_T="t9=t9gen_zs_v1 v9=v9gen_img_v1 v2=zst_pred_v1 v3=zst_v3q_test fn=d_fnext_test rrfn=rrfn_test rr2x=rank:runs/rr2x5_test/rank.npy+runs/rr2x5_s1_test/rank.npy"

stage_final_fits() {  # the two final stacks (they differ only in the per-code bias threshold) -> fits/final_<id>.npz
  local spec id cb mc
  mkdir -p fits
  for spec in $FINALS; do IFS=: read -r id cb mc <<< "$spec"
    if have "fits/final_$id.npz"; then say "final fit $id: present"; continue; fi
    cpu "$THREADS_FINAL" scripts/fit_stack.py --c $FINAL_C --folds 0 1 2 3 4 --code-bias "$cb" --rank-shape v2 v3 fn --save "fits/final_$id.npz"
  done
}

stage_predictions() {  # final stacks on test + final share matching -> outputs/predictions_<id>.csv
  local spec id cb mc
  mkdir -p outputs
  for spec in $FINALS; do IFS=: read -r id cb mc <<< "$spec"
    cpu "$THREADS_FINAL" scripts/fit_stack.py --predict "fits/final_$id.npz" --test $FINAL_T --out "outputs/final_${id}_raw.csv"
    cpu "$THREADS_FINAL" scripts/late_match.py test "fits/final_$id.npz" "outputs/final_${id}_raw.npy" "outputs/predictions_$id.csv" --rare2 1 --min-count "$mc"
    cpu 4 scripts/check_predictions.py "outputs/predictions_$id.csv"
  done
}
