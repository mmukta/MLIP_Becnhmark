#!/bin/sh -l
#SBATCH --partition=Orion,Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread

set -eu

BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$BASE_DIR"

if [ -z "${REFCODE:-}" ]; then
  echo "ERROR: REFCODE is not set."
  exit 2
fi

PY_SCRIPT="${PY_SCRIPT:-run_individual.py}"
CALCULATOR="${CALCULATOR:-MACEOFF}"
MODE="${MODE:-auto}"
CSV_FILE="${CSV_FILE:-$BASE_DIR/entire_data.csv}"
DB_FILE="${DB_FILE:-$BASE_DIR/HEM.db}"
FF_STYLE="${FF_STYLE:-none}"
TIMEOUT_SEC="${TIMEOUT_SEC:-14400}"
MAX_RELAX_STEPS="${MAX_RELAX_STEPS:-5000}"
LOG_TAG="${LOG_TAG:-}"

if [ ! -f "$DB_FILE" ]; then
  echo "ERROR: DB_FILE not found: $DB_FILE"
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TBB_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_MAIN_FREE=1

echo "BASE_DIR=$BASE_DIR"
echo "PY_SCRIPT=$PY_SCRIPT"
echo "REFCODE=$REFCODE"
echo "CALCULATOR=$CALCULATOR"
echo "MODE=$MODE"
echo "CSV_FILE=$CSV_FILE"
echo "DB_FILE=$DB_FILE"
echo "FF_STYLE=$FF_STYLE"
echo "TIMEOUT_SEC=$TIMEOUT_SEC"
echo "MAX_RELAX_STEPS=$MAX_RELAX_STEPS"
echo "LOG_TAG=${LOG_TAG:-'(none)'}"

set -- python "$PY_SCRIPT" "$REFCODE" \
  --calculator "$CALCULATOR" \
  --mode "$MODE" \
  --csv-file "$CSV_FILE" \
  --db-file "$DB_FILE" \
  --ff-style "$FF_STYLE" \
  --timeout-sec "$TIMEOUT_SEC" \
  --max-relax-steps "$MAX_RELAX_STEPS"

if [ -n "$LOG_TAG" ]; then
  set -- "$@" --log-tag "$LOG_TAG"
fi

srun --cpu-bind=none "$@"
