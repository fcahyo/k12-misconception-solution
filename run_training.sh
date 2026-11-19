#!/usr/bin/env bash
# Full training from scratch: every stage on the training rows (out-of-fold) and on the test set, the student fine-tuning,
# all stack fits, and finally outputs/predictions_58446106.csv and outputs/predictions_58446518.csv.
# Needs only the competition data (README, "Data") and the three base models, in a clean checkout: trained artefacts that are
# already present would otherwise be reused as if this run had produced them. Environment variables: see scripts/stages.sh.
source "$(dirname "$0")/scripts/stages.sh"

if [ -n "${ROWS:-}" ]; then echo "error: ROWS is for smoke runs of run_inference.sh" >&2; exit 1; fi
if [ ! -e .training_run ] && { compgen -G "fits/*.npz" > /dev/null || compgen -G "runs/*/fold*/adapter" > /dev/null; }; then
  echo "error: this checkout already holds trained artefacts (fits/ or student adapters under runs/); training from scratch" \
       "needs a clean checkout (README, 'Commands')" >&2
  exit 1
fi
dry || date '+%F %H:%M:%S' > .training_run   # marks the artefacts below as this run's own, so an interrupted run can resume

cpu 4 scripts/prepare_data.py
cpu 4 scripts/check_test_set.py   # refuses test-side outputs of another test set (README, 'Running on a new test set')
stage_reader train
stage_reader test
stage_solutions
stage_students_train
stage_teachers train
stage_teachers test
stage_flash_teacher train
stage_flash_teacher test
stage_round0_fit
stage_top8
stage_rerank1 train
stage_rerank1 test
stage_round1_fit
stage_top6
stage_rerank2 train
stage_rerank2 test
stage_final_fits
stage_predictions
say "done: outputs/predictions_58446106.csv and outputs/predictions_58446518.csv"
