#!/usr/bin/env bash
# Run on the login node. Strict mode is confined to this child shell.
set -euo pipefail
cd "${GREED_REPOSITORY:?}"
: "${GREED_50K_ROOT:?}" "${GREED_100K_ROOT:?}"
record="$GREED_50K_ROOT/metadata/next_stage_submission.env"
if [[ -e "$record" || -e "$GREED_100K_ROOT" ]]; then
  echo "Existing submission record or 100k directory found; inspect before resubmitting: $record" >&2
  exit 1
fi
test -f "$GREED_50K_ROOT/results/production/ged_labels_train.csv"
test -f "$GREED_50K_ROOT/checkpoints/greed_3d_50k/pairs_50000/seed_161803/latest_checkpoint.pt"
test -f "$GREED_50K_ROOT/checkpoints/greed_3d_50k/pairs_50000/seed_271828/latest_checkpoint.pt"
printf 'export GREED_50K_ROOT=%q\nexport GREED_100K_ROOT=%q\n' "$GREED_50K_ROOT" "$GREED_100K_ROOT" > "$record"
printf 'export GREED_REPOSITORY=%q\n' "$GREED_REPOSITORY" >> "$record"
if [[ -n "${GREED_ENV:-}" ]]; then
  printf 'export GREED_ENV=%q\n' "$GREED_ENV" >> "$record"
fi
printf 'SUBMISSION_COMMIT=%q\n' "$(git rev-parse HEAD)" >> "$record"
echo "SUBMISSION_RECORD=$record"
submit() {
  local key="$1" job
  shift
  job=$(sbatch --parsable "$@")
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]]
  printf '%s=%q\n' "$key" "$job" >> "$record"
  printf '%s=%s\n' "$key" "$job"
  printf -v "$key" '%s' "$job"
}
submit EXTEND_50K_JOB_ID scripts/jean_zay/extend_50k.sbatch
submit EXTEND_REPORT_JOB_ID --dependency="afterok:$EXTEND_50K_JOB_ID" scripts/jean_zay/report_extended_50k.sbatch
submit TEST_INVENTORY_JOB_ID scripts/jean_zay/test_patient_inventory.sbatch
submit PREPARE_100K_JOB_ID scripts/jean_zay/prepare_100k.sbatch
submit VERIFY_100K_JOB_ID --time=01:00:00 --dependency="afterok:$PREPARE_100K_JOB_ID" scripts/jean_zay/verify_environment.sbatch
submit GED_100K_JOB_ID --dependency="afterok:$VERIFY_100K_JOB_ID" scripts/jean_zay/production_100k.sbatch
submit FINALIZE_100K_JOB_ID --dependency="afterok:$GED_100K_JOB_ID" scripts/jean_zay/finalize_100k.sbatch
