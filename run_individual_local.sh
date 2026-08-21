#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

TARGET_METHOD="${1:-${TARGET_METHOD:-}}"
PARALLEL_JOBS="${2:-${PARALLEL_JOBS:-6}}"

PY_SCRIPT="${PY_SCRIPT:-$BASE_DIR/run_individual.py}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODE="${MODE:-auto}"
DB_FILE="${DB_FILE:-$BASE_DIR/HEM.db}"
METHODS_CSV="${METHODS_CSV:-$BASE_DIR/t.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ML_FMAX="${ML_FMAX:-0.01}"
FF_STYLE="${FF_STYLE:-none}"
TIMEOUT_SEC="${TIMEOUT_SEC:-64400}"
MAX_RELAX_STEPS="${MAX_RELAX_STEPS:-5000}"
LOG_TAG="${LOG_TAG:-}"

if [[ -z "$TARGET_METHOD" ]]; then
  echo "Usage: $0 <METHOD> [PARALLEL_JOBS]"
  echo "Example: $0 MACE 6"
  exit 2
fi

TARGET_METHOD="$(echo "$TARGET_METHOD" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
case "$TARGET_METHOD" in
  MACE|MACEOFF|UMA|ORB|ORB_V3|GAFF|GAFF_MACE|GAFF_MACEOFF|GAFF_UMA|GAFF_ORB|GAFF_ORB_V3) ;;
  *)
    echo "Invalid METHOD: $TARGET_METHOD"
    echo "Choose one of: MACE, MACEOFF, UMA, ORB, ORB_V3, GAFF, GAFF_MACE, GAFF_MACEOFF, GAFF_UMA, GAFF_ORB, GAFF_ORB_V3"
    exit 2
    ;;
esac

if [[ -z "$OUTPUT_DIR" ]]; then
  case "$TARGET_METHOD" in
    MACE|GAFF_MACE)
      OUTPUT_DIR="$BASE_DIR/output_mace2"
      ;;
    MACEOFF|GAFF_MACEOFF)
      OUTPUT_DIR="$BASE_DIR/output_maceoff"
      ;;
    *)
      OUTPUT_DIR="$BASE_DIR/output_$(echo "$TARGET_METHOD" | tr '[:upper:]' '[:lower:]')"
      ;;
  esac
fi

if [[ ! -f "$METHODS_CSV" ]]; then
  echo "ERROR: METHODS_CSV not found: $METHODS_CSV"
  exit 2
fi
if [[ ! -f "$DB_FILE" ]]; then
  echo "ERROR: DB_FILE not found: $DB_FILE"
  exit 2
fi
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: PY_SCRIPT not found: $PY_SCRIPT"
  exit 2
fi
if [[ -z "$PYTHON_BIN" ]] || [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 2
fi
if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [[ "$PARALLEL_JOBS" -lt 1 ]]; then
  echo "ERROR: PARALLEL_JOBS must be a positive integer (got: $PARALLEL_JOBS)"
  exit 2
fi
if ! [[ "$ML_FMAX" =~ ^[0-9]*\.?[0-9]+$ ]]; then
  echo "WARNING: invalid ML_FMAX='$ML_FMAX'; using default 0.05"
  ML_FMAX="0.05"
fi

mkdir -p "$OUTPUT_DIR/single_runs"

echo "BASE_DIR=$BASE_DIR"
echo "TARGET_METHOD=$TARGET_METHOD"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "PARALLEL_JOBS=$PARALLEL_JOBS"
echo "METHODS_CSV=$METHODS_CSV"
echo "DB_FILE=$DB_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ML_FMAX=$ML_FMAX"
echo "FF_STYLE=$FF_STYLE"
echo "TIMEOUT_SEC=$TIMEOUT_SEC"
echo "MAX_RELAX_STEPS=$MAX_RELAX_STEPS"
echo "LOG_TAG=${LOG_TAG:-'(none)'}"

TMP_REFCODES="$(mktemp)"
TMP_PENDING="$(mktemp)"
trap 'rm -f "$TMP_REFCODES" "$TMP_PENDING"' EXIT

"$PYTHON_BIN" - "$METHODS_CSV" "$TARGET_METHOD" > "$TMP_REFCODES" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
target_method = sys.argv[2].strip().upper()
seen = set()

with open(csv_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        refcode = (row.get("refcode") or "").strip()
        methods_raw = (row.get("methods") or "").strip()
        if not refcode or not methods_raw:
            continue
        methods = {m.strip().upper() for m in methods_raw.split(",") if m.strip()}
        if target_method in methods and refcode not in seen:
            print(refcode)
            seen.add(refcode)
PY

if [[ ! -s "$TMP_REFCODES" ]]; then
  echo "No refcodes found for method $TARGET_METHOD in $METHODS_CSV"
  exit 0
fi

NUM_REFCODES="$(wc -l < "$TMP_REFCODES" | tr -d ' ')"
echo "Resolved $NUM_REFCODES refcodes for method $TARGET_METHOD"

OUTPUT_LABEL="$("$PYTHON_BIN" - "$ML_FMAX" <<'PY'
import sys
v = float(sys.argv[1])
print(f"fmax{int(round(v*100)):02d}")
PY
)"
echo "OUTPUT_LABEL=$OUTPUT_LABEL"

"$PYTHON_BIN" - "$TMP_REFCODES" "$OUTPUT_DIR" "$TARGET_METHOD" "$OUTPUT_LABEL" > "$TMP_PENDING" <<'PY'
import pathlib
import sys

ref_path, out_dir, method, output_label = sys.argv[1:]
method = method.strip().upper()
out_dir = pathlib.Path(out_dir)

for line in pathlib.Path(ref_path).read_text(encoding="utf-8").splitlines():
    ref = line.strip()
    if not ref:
        continue
    ref_dir = out_dir / ref
    done_markers = sorted(ref_dir.glob(f"OK{output_label}_{method}.json"))
    done_markers += sorted(ref_dir.glob(f"TIMEOUT{output_label}_{method}.json"))
    if done_markers:
        print(f"SKIP\t{ref}\t{done_markers[0]}")
    else:
        print(f"RUN\t{ref}")
PY

awk -F '\t' '$1=="SKIP"{print "[SKIP] " $2 " (" ENVIRON["TARGET_METHOD"] ") -> " $3}' "$TMP_PENDING"
awk -F '\t' '$1=="RUN"{print $2}' "$TMP_PENDING" > "${TMP_PENDING}.run"
mv "${TMP_PENDING}.run" "$TMP_PENDING"

NUM_PENDING="$(wc -l < "$TMP_PENDING" | tr -d ' ')"
echo "Pending refcodes after skip check: $NUM_PENDING"
if [[ "$NUM_PENDING" -eq 0 ]]; then
  echo "All matching refcodes are already done for $TARGET_METHOD/$OUTPUT_LABEL"
  exit 0
fi

run_one_refcode() {
  local ref="$1"
  local ml_fmax="${ML_FMAX:-0.05}"
  if ! [[ "$ml_fmax" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    ml_fmax="0.05"
  fi
  local -a cmd
  cmd=("$PYTHON_BIN" "$PY_SCRIPT" "$ref"
    --calculator "$TARGET_METHOD"
    --mode "$MODE"
    --db-file "$DB_FILE"
    --ff-style "$FF_STYLE"
    --ml-fmax "$ml_fmax"
    --log-dir "$OUTPUT_DIR/single_runs"
    --timeout-sec "$TIMEOUT_SEC"
    --max-relax-steps "$MAX_RELAX_STEPS")
  if [[ -n "$LOG_TAG" ]]; then
    cmd+=(--log-tag "$LOG_TAG")
  fi
  echo "[$(date '+%F %T')] START $ref ($TARGET_METHOD)"
  OUTPUT_DIR="$OUTPUT_DIR" "${cmd[@]}"
  rc=$?
  echo "[$(date '+%F %T')] END   $ref ($TARGET_METHOD) (exit=$rc)"
  return "$rc"
}

export PYTHON_BIN PY_SCRIPT TARGET_METHOD MODE DB_FILE OUTPUT_DIR FF_STYLE TIMEOUT_SEC MAX_RELAX_STEPS LOG_TAG
export -f run_one_refcode

xargs -P "$PARALLEL_JOBS" -I {} bash -c 'run_one_refcode "$@"' _ {} < "$TMP_PENDING"
