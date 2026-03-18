#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
FAIL_DIR="${FAIL_DIR:-$BASE_DIR/FAIL}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$BASE_DIR/run_individual.sbatch}"
LOG_DIR="${LOG_DIR:-$BASE_DIR/slurm_logs_individual}"
PY_SCRIPT="${PY_SCRIPT:-run_individual.py}"
CALCULATOR="${CALCULATOR:-MACEOFF}"
MODE="${MODE:-auto}"
CSV_FILE="${CSV_FILE:-$BASE_DIR/entire_data.csv}"
TIMEOUT_SEC="${TIMEOUT_SEC:-14400}"
MAX_RELAX_STEPS="${MAX_RELAX_STEPS:-5000}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
WALLTIME="${WALLTIME:-24:00:00}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$FAIL_DIR" ]]; then
  echo "ERROR: missing FAIL directory: $FAIL_DIR"
  exit 2
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch script: $SBATCH_SCRIPT"
  exit 2
fi

if [[ ! -f "$BASE_DIR/$PY_SCRIPT" && ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: could not find PY_SCRIPT=$PY_SCRIPT"
  exit 2
fi

collect_targets() {
  if (( $# > 0 )); then
    printf '%s\n' "$@"
    return
  fi

  if [[ -n "${TARGET_REFS:-}" ]]; then
    for ref in $TARGET_REFS; do
      printf '%s\n' "$ref"
    done
    return
  fi

  find "$FAIL_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
}

mapfile -t RAW_TARGETS < <(collect_targets "$@")

if (( ${#RAW_TARGETS[@]} == 0 )); then
  echo "ERROR: no target systems provided."
  echo "Pass refcodes as arguments, or set TARGET_REFS=\"FANTIX ACAMIA\"."
  exit 2
fi

echo "BASE_DIR=$BASE_DIR"
echo "FAIL_DIR=$FAIL_DIR"
echo "SBATCH_SCRIPT=$SBATCH_SCRIPT"
echo "PY_SCRIPT=$PY_SCRIPT"
echo "CALCULATOR=$CALCULATOR"
echo "MODE=$MODE"
echo "CSV_FILE=$CSV_FILE"
echo "TIMEOUT_SEC=$TIMEOUT_SEC"
echo "MAX_RELAX_STEPS=$MAX_RELAX_STEPS"
echo "CPUS_PER_TASK=$CPUS_PER_TASK"
echo "MEMORY=$MEMORY"
echo "WALLTIME=$WALLTIME"

submitted=0
for raw_ref in "${RAW_TARGETS[@]}"; do
  ref="$(printf '%s' "$raw_ref" | tr -cd '[:alnum:]')"
  if [[ -z "$ref" ]]; then
    continue
  fi

  system_dir="$FAIL_DIR/$ref"
  if [[ ! -d "$system_dir" ]]; then
    echo "SKIP $ref: missing directory $system_dir"
    continue
  fi

  if [[ ! -f "$system_dir/$ref.cif" ]]; then
    echo "SKIP $ref: missing original CIF $system_dir/$ref.cif"
    continue
  fi

  job_name="$(printf 'mlip_%s_%s' "$(printf '%s' "$CALCULATOR" | tr '[:upper:]' '[:lower:]')" "$ref")"
  log_tag="$ref"

  jobid=$(
    sbatch --parsable \
      --partition=Orion,Apus \
      --job-name="$job_name" \
      --nodes=1 \
      --ntasks-per-node=1 \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEMORY" \
      --time="$WALLTIME" \
      --hint=nomultithread \
      --open-mode=append \
      --output="$LOG_DIR/%x.out" \
      --error="$LOG_DIR/%x.err" \
      --export=ALL,BASE_DIR="$BASE_DIR",REFCODE="$ref",PY_SCRIPT="$PY_SCRIPT",CALCULATOR="$CALCULATOR",MODE="$MODE",CSV_FILE="$CSV_FILE",TIMEOUT_SEC="$TIMEOUT_SEC",MAX_RELAX_STEPS="$MAX_RELAX_STEPS",LOG_TAG="$log_tag" \
      "$SBATCH_SCRIPT"
  )

  echo "Submitted $job_name refcode=$ref jobid=$jobid"
  submitted=$((submitted + 1))
done

echo "Done. Submitted $submitted job(s)."
