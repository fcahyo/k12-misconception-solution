#!/usr/bin/env bash
# Inference from the trained weights: the test set -> outputs/predictions_58446106.csv and outputs/predictions_58446518.csv.
# Needs the competition data (README, "Data") and the trained artefacts: the ten student adapters under runs/ and the five
# stack fits under fits/ (README, "Trained artefacts"). Environment variables: see scripts/stages.sh (M9, M27, MFN, GPUS, ...).
source "$(dirname "$0")/scripts/stages.sh"

if [ -e data/labels.json ] && [ -e data/folds.csv ]; then cpu 4 scripts/prepare_data.py --check; else cpu 4 scripts/prepare_data.py; fi
cpu 4 scripts/check_test_set.py   # refuses test-side outputs of another test set (README, 'Running on a new test set')
missing=()
for f in fits/round0.npz fits/round1.npz fits/round1_matched.npz fits/final_58446106.npz fits/final_58446518.npz; do
  [ -e "$f" ] || missing+=("$f")
done
for r in t9gen_zs_v1 v9gen_img_v1; do for k in 0 1 2 3 4; do for f in adapter args.json; do
  [ -e "runs/$r/fold$k/$f" ] || missing+=("runs/$r/fold$k/$f")
done; done; done
if [ ${#missing[@]} -gt 0 ]; then
  echo "missing trained artefacts (README, 'Trained artefacts'):"; printf '  %s\n' "${missing[@]}"
  if dry; then echo "(dry run: a real run would stop here)"; else exit 1; fi
fi

stage_reader test
stage_solutions
stage_students_test
stage_teachers test
stage_flash_teacher test
if [ -n "${ROWS:-}" ]; then
  say "smoke run: the model stages up to the Flash-Next teacher ran on the rows of $ROWS. The candidate lists and every later stage need all test rows, so the run stops here."
  exit 0
fi
stage_top8
stage_rerank1 test
stage_top6
stage_rerank2 test
stage_predictions
say "done: outputs/predictions_58446106.csv and outputs/predictions_58446518.csv"
