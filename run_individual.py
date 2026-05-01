#!/usr/bin/env python3
"""Run or resume a single relaxation target without marker dependencies."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import signal
import time
import traceback
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FAIL_DIR = BASE_DIR / "FAIL"
CSV_FILE = os.environ.get("CSV_FILE", str(BASE_DIR / "entire_data.csv"))
DEFAULT_LOG_DIR = str(BASE_DIR / "single_runs")
DEFAULT_DB_FILE = os.environ.get("DB_FILE", str(BASE_DIR / "HEM.db"))

ID_COL = "ccdc_id"
SMILES_COL = "CHIRAL SMILES"
DEFAULT_CALCULATOR = os.environ.get("CALCULATOR", "MACEOFF").strip().upper() or "MACEOFF"
FF_STYLE_BY_CALCULATOR = {
    "MACE": "openff",
    "MACEOFF": "openff",
    "UMA": "openff",
    "GAFF_MACE": "gaff",
    "GAFF_MACEOFF": "gaff",
    "GAFF_UMA": "gaff",
}
ML_CALCULATOR_BY_CALCULATOR = {
    "MACE": "MACE",
    "MACEOFF": "MACEOFF",
    "UMA": "UMA",
    "GAFF_MACE": "MACE",
    "GAFF_MACEOFF": "MACEOFF",
    "GAFF_UMA": "UMA",
}


def calculator_suffix(calculator: str) -> str:
    value = calculator.strip().upper()
    if value.startswith("GAFF_"):
        return value.lower()
    if value == "MACEOFF":
        return "maceoff"
    if value == "MACE":
        return "mace"
    if value == "UMA":
        return "uma"
    return value.lower()


def clean_refcode(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    ref = raw.split()[-1].strip()
    return re.sub(r"[^A-Za-z0-9]", "", ref)


def ref_out_dir(refcode: str) -> str:
    return str(FAIL_DIR / refcode)


def original_cif_path(refcode: str) -> str:
    return str(FAIL_DIR / refcode / f"{refcode}.cif")


def relaxed_cif_path(refcode: str, calculator: str) -> str:
    return os.path.join(ref_out_dir(refcode), f"{refcode}_relaxed_{calculator_suffix(calculator)}.cif")


def timeout_partial_cif_path(refcode: str, calculator: str) -> str:
    suffix = calculator_suffix(calculator)
    return os.path.join(ref_out_dir(refcode), f"{refcode}_relaxed_{suffix}_timeout.cif")


def fail_snapshot_cif_path(refcode: str, calculator: str) -> str:
    suffix = calculator_suffix(calculator)
    return os.path.join(ref_out_dir(refcode), f"{refcode}_relaxed_{suffix}_fail.cif")


def prepend_smiles_to_cif(cif_path: str, refcode: str, smiles: str):
    smiles = (smiles or "").strip()
    if not smiles:
        return

    header = (
        f"smiles: {smiles}\n"
        f"# Refcode: {refcode}\n"
        f"# SourceCSV: {os.path.basename(str(CSV_FILE))}\n"
    )

    with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
        body = f.read()

    if body.startswith("smiles:"):
        return

    with open(cif_path, "w", encoding="utf-8") as f:
        f.write(header + body)


def save_structure_snapshot(snapshot_path: str, refcode: str, smiles: str, structure) -> str:
    if structure is None:
        return ""
    try:
        os.makedirs(ref_out_dir(refcode), exist_ok=True)
        structure.to_file(snapshot_path)
        prepend_smiles_to_cif(snapshot_path, refcode, smiles)
        return snapshot_path
    except Exception:
        return ""


def write_json_atomic(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def load_smiles_map(csv_path: str) -> dict[str, str]:
    smiles_map: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames or []]

        if ID_COL not in reader.fieldnames:
            raise ValueError(f"Missing '{ID_COL}' column. Columns={reader.fieldnames}")
        if SMILES_COL not in reader.fieldnames:
            raise ValueError(f"Missing '{SMILES_COL}' column. Columns={reader.fieldnames}")

        for row in reader:
            row = {(k or "").strip(): v for k, v in row.items()}
            ref = clean_refcode(row.get(ID_COL, ""))
            if not ref or ref in smiles_map:
                continue
            smiles_map[ref] = (row.get(SMILES_COL, "") or "").strip()
    return smiles_map


def resolve_seed_cif(refcode: str, calculator: str, mode: str, explicit_seed: str | None) -> str:
    if explicit_seed:
        path = Path(explicit_seed)
        return str(path if path.is_absolute() else Path.cwd() / path)

    original = original_cif_path(refcode)
    timeout_seed = timeout_partial_cif_path(refcode, calculator)
    fail_seed = fail_snapshot_cif_path(refcode, calculator)

    if mode == "original":
        return original

    if mode == "resume":
        if os.path.exists(timeout_seed):
            return timeout_seed
        if os.path.exists(fail_seed):
            return fail_seed
        raise FileNotFoundError(
            f"No resume snapshot found for {refcode}. Checked {timeout_seed} and {fail_seed}"
        )

    if os.path.exists(timeout_seed):
        return timeout_seed
    return original


def resolve_db_file(raw_db_file: str) -> str:
    path = Path(raw_db_file).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def load_smiles_from_db(db_file: str, refcode: str) -> str:
    try:
        from pyxtal.db import database

        db = database(db_file)
        row = db.get_row(refcode)
        return str(getattr(row, "mol_smi", "") or "").strip()
    except Exception:
        return ""


def load_charmm_info_from_db(db_file: str, refcode: str) -> dict | None:
    try:
        from pyxtal.db import database

        db = database(db_file)
        row = db.get_row(refcode)
        raw = getattr(row, "charmm_info", None)
        if raw is None:
            return None
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return None
        if {"prm", "rtf", "atom_info"}.issubset(data):
            return data
    except Exception:
        return None
    return None


def worker_relax_task(
    refcode: str,
    smiles: str,
    calculator: str,
    seed_cif: str,
    db_file: str,
    ff_style: str,
    ff_charge_method: str,
    ff_max_steps: int,
    ff_fmax: float,
    ml_fmax: float,
    ff_nproc: int,
    max_relax_steps: int,
    result_q: mp.Queue,
):
    t0 = time.time()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["TBB_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"

    c = None
    seed_label = seed_cif
    relaxation_started = False

    def _handle_terminate(signum, frame):
        out_cif = save_structure_snapshot(timeout_partial_cif_path(refcode, calculator), refcode, smiles, c)
        raise SystemExit(128 + int(signum) + (1 if out_cif else 0))

    signal.signal(signal.SIGTERM, _handle_terminate)
    signal.signal(signal.SIGINT, _handle_terminate)

    try:
        import torch

        torch.set_num_threads(1)

        from relax_lib import load_relaxation_structure, run_ff_then_ml_relax

        os.makedirs(ref_out_dir(refcode), exist_ok=True)

        if db_file:
            if not os.path.exists(db_file):
                raise FileNotFoundError(f"DB file not found: {db_file}")
            c = load_relaxation_structure(db_file=db_file, code=refcode, smiles=smiles)
            seed_label = f"{db_file}:{refcode}"
        else:
            if not os.path.exists(seed_cif):
                raise FileNotFoundError(f"CIF not found: {seed_cif}")
            c = load_relaxation_structure(cif=seed_cif, smiles=smiles)
            seed_label = seed_cif

        charmm_info = load_charmm_info_from_db(db_file, refcode) if db_file else None
        ml_calculator = ML_CALCULATOR_BY_CALCULATOR[calculator]

        relaxation_started = True
        result = run_ff_then_ml_relax(
            c,
            refcode=refcode,
            out_dir=ref_out_dir(refcode),
            ff_style=ff_style,
            ml_calculator=ml_calculator,
            ff_charge_method=ff_charge_method,
            ff_fmax=ff_fmax,
            ml_fmax=ml_fmax,
            ff_max_steps=ff_max_steps,
            ml_max_steps=max_relax_steps,
            ff_nproc=ff_nproc,
            charmm_info=charmm_info,
        )

        out_cif = str(result.get("out_cif", ""))
        if out_cif:
            prepend_smiles_to_cif(out_cif, refcode, smiles)
        ff_cif = str(result.get("ff_cif", ""))
        if ff_cif:
            prepend_smiles_to_cif(ff_cif, refcode, smiles)

        if result.get("status") == "OK":
            res = {
                "status": "OK",
                "refcode": refcode,
                "energy": float(result.get("energy", 0.0)),
                "converged": True,
                "reason": "",
                "steps": int(result.get("ml_steps", 0)),
                "max_steps": int(max_relax_steps),
                "out_cif": out_cif,
                "ff_cif": ff_cif,
                "seed_cif": seed_label,
                "seconds": time.time() - t0,
                "relaxation_started": True,
                "calculator": calculator,
                "ml_calculator": ml_calculator,
                "ff_style": ff_style,
                "ff_converged": bool(result.get("ff_converged", False)),
                "ff_steps": int(result.get("ff_steps", 0)),
                "ff_seconds": float(result.get("ff_seconds", 0.0)),
                "ml_seconds": float(result.get("ml_seconds", 0.0)),
            }
        else:
            res = {
                "status": "TIMEOUT",
                "refcode": refcode,
                "energy": float(result.get("energy", 0.0)),
                "converged": False,
                "reason": f"Reached max relaxation steps ({max_relax_steps})",
                "steps": int(result.get("ml_steps", 0)),
                "max_steps": int(max_relax_steps),
                "out_cif": out_cif,
                "ff_cif": ff_cif,
                "seed_cif": seed_label,
                "seconds": time.time() - t0,
                "relaxation_started": True,
                "calculator": calculator,
                "ml_calculator": ml_calculator,
                "ff_style": ff_style,
                "ff_converged": bool(result.get("ff_converged", False)),
                "ff_steps": int(result.get("ff_steps", 0)),
                "ff_seconds": float(result.get("ff_seconds", 0.0)),
                "ml_seconds": float(result.get("ml_seconds", 0.0)),
            }
        result_q.put(res)

    except Exception as e:
        error_details = traceback.format_exc()
        res = {
            "status": "FAIL",
            "refcode": refcode,
            "reason": str(e),
            "exception_type": type(e).__name__,
            "traceback": error_details,
            "error_details": error_details,
            "seconds": time.time() - t0,
            "seed_cif": seed_label,
            "out_cif": "",
            "relaxation_started": relaxation_started,
            "calculator": calculator,
            "ff_style": ff_style,
        }
        result_q.put(res)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume one target from the FAIL/<refcode>/ folder."
    )
    parser.add_argument("refcode", help="Target refcode, for example FANTIX")
    parser.add_argument(
        "--calculator",
        choices=["MACE", "MACEOFF", "UMA", "GAFF_MACE", "GAFF_MACEOFF", "GAFF_UMA"],
        default=DEFAULT_CALCULATOR,
        help="Calculator to use for the run.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "original", "resume"],
        default="auto",
        help="Choose FAIL/<refcode>/<refcode>.cif, resume snapshot, or auto-pick snapshot if available.",
    )
    parser.add_argument(
        "--seed-cif",
        default="",
        help="Explicit CIF path to use as the input seed. Overrides --mode.",
    )
    parser.add_argument(
        "--csv-file",
        default=CSV_FILE,
        help="CSV file used as a fallback to resolve SMILES when --db-file has no mol_smi.",
    )
    parser.add_argument(
        "--db-file",
        default=DEFAULT_DB_FILE,
        help="pyxtal database used to load the original refcode structure. Use '' to load from --seed-cif instead.",
    )
    parser.add_argument(
        "--ff-style",
        choices=["auto", "openff", "gaff"],
        default=os.environ.get("FF_STYLE", "auto"),
        help="CHARMM force-field style for the position-only pre-relaxation.",
    )
    parser.add_argument(
        "--ff-charge-method",
        default=os.environ.get("FF_CHARGE_METHOD", "am1bcc"),
        help="Charge method passed to pyxtal.get_forcefield for CHARMM setup.",
    )
    parser.add_argument(
        "--ff-max-steps",
        type=int,
        default=int(os.environ.get("FF_MAX_RELAX_STEPS", "5000")),
        help="Maximum CHARMM position-only relaxation steps.",
    )
    parser.add_argument(
        "--ff-fmax",
        type=float,
        default=float(os.environ.get("FF_FMAX", "0.05")),
        help="CHARMM pre-relaxation force target.",
    )
    parser.add_argument(
        "--ml-fmax",
        type=float,
        default=float(os.environ.get("ML_FMAX", "0.05")),
        help="MACE/MACEOFF/UMA relaxation force target.",
    )
    parser.add_argument(
        "--ff-nproc",
        type=int,
        default=int(os.environ.get("FF_NPROC", "1")),
        help="Compatibility option for CHARMM/force-field setup.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(os.environ.get("TIMEOUT_SEC", str(240 * 60))),
        help="Wall-clock timeout in seconds for this one run.",
    )
    parser.add_argument(
        "--max-relax-steps",
        type=int,
        default=int(os.environ.get("MAX_RELAX_STEPS", "5000")),
        help="Maximum FIRE relaxation steps.",
    )
    parser.add_argument(
        "--log-tag",
        default="",
        help="Optional suffix to include in the single-run report filename.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for single-run JSON/JSONL reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refcode = clean_refcode(args.refcode)
    if not refcode:
        raise ValueError("Refcode is empty after cleaning.")

    calculator = args.calculator.strip().upper()
    db_file = resolve_db_file(args.db_file) if str(args.db_file or "").strip() else ""
    ff_style = args.ff_style.strip().lower()
    if ff_style == "auto":
        ff_style = FF_STYLE_BY_CALCULATOR[calculator]

    os.makedirs(FAIL_DIR, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(ref_out_dir(refcode), exist_ok=True)

    smiles = load_smiles_from_db(db_file, refcode) if db_file else ""
    if not smiles:
        smiles_map = load_smiles_map(args.csv_file)
        smiles = smiles_map.get(refcode, "")
    if not smiles:
        raise ValueError(f"SMILES not found for refcode {refcode} in {db_file or args.csv_file}")

    seed_cif = ""
    if db_file and not args.seed_cif:
        if not os.path.exists(db_file):
            raise FileNotFoundError(f"DB file not found: {db_file}")
        seed_cif = f"{db_file}:{refcode}"
    else:
        seed_cif = resolve_seed_cif(refcode, calculator, args.mode, args.seed_cif or None)
    if not (db_file and not args.seed_cif) and not os.path.exists(seed_cif):
        raise FileNotFoundError(f"Input seed CIF not found: {seed_cif}")

    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", (args.log_tag or "").strip())
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{refcode}_{calculator_suffix(calculator)}_{timestamp}"
    if tag:
        stem += f"_{tag}"

    report_jsonl = os.path.join(args.log_dir, f"{stem}.jsonl")
    result_json = os.path.join(args.log_dir, f"{stem}.json")

    result_q: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=worker_relax_task,
        args=(
            refcode,
            smiles,
            calculator,
            "" if db_file and not args.seed_cif else seed_cif,
            db_file if not args.seed_cif else "",
            ff_style,
            args.ff_charge_method,
            int(args.ff_max_steps),
            float(args.ff_fmax),
            float(args.ml_fmax),
            int(args.ff_nproc),
            int(args.max_relax_steps),
            result_q,
        ),
        daemon=True,
    )
    proc.start()

    t_start = time.time()
    timed_out = False
    result = None

    while True:
        try:
            result = result_q.get(timeout=0.2)
            break
        except Exception:
            if not proc.is_alive():
                break
            if (time.time() - t_start) > int(args.timeout_sec):
                timed_out = True
                break

    if timed_out:
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        except Exception:
            pass

        partial_cif = timeout_partial_cif_path(refcode, calculator)
        has_partial = os.path.exists(partial_cif) and os.path.getsize(partial_cif) > 0
        result = {
            "status": "TIMEOUT",
            "refcode": refcode,
            "reason": f"Exceeded {int(args.timeout_sec)} seconds",
            "seconds": time.time() - t_start,
            "out_cif": partial_cif if has_partial else "",
            "seed_cif": seed_cif,
            "relaxation_started": True,
            "calculator": calculator,
            "ff_style": ff_style,
        }
    elif result is None:
        result = {
            "status": "FAIL",
            "refcode": refcode,
            "reason": "Process exited without result",
            "seconds": time.time() - t_start,
            "out_cif": "",
            "seed_cif": seed_cif,
            "relaxation_started": False,
            "calculator": calculator,
            "ff_style": ff_style,
        }

    try:
        proc.join(timeout=0)
    except Exception:
        pass

    result["smiles"] = smiles
    result["requested_mode"] = args.mode
    result["resolved_seed_cif"] = seed_cif
    result["db_file"] = db_file
    result["ff_style"] = ff_style
    result["ff_max_steps"] = int(args.ff_max_steps)
    result["ff_fmax"] = float(args.ff_fmax)
    result["ml_fmax"] = float(args.ml_fmax)
    result["max_relax_steps"] = int(args.max_relax_steps)
    result["timeout_sec"] = int(args.timeout_sec)

    write_json_atomic(result_json, result)
    append_jsonl(report_jsonl, result)

    print(f"refcode={refcode}")
    print(f"calculator={calculator}")
    print(f"mode={args.mode}")
    print(f"db_file={db_file}")
    print(f"ff_style={ff_style}")
    print(f"seed_cif={seed_cif}")
    print(f"status={result.get('status', 'FAIL')}")
    print(f"reason={result.get('reason', '')}")
    if result.get("error_details"):
        print("error_details=")
        print(result["error_details"])
    print(f"out_cif={result.get('out_cif', '')}")
    print(f"result_json={result_json}")
    print(f"report_jsonl={report_jsonl}")
    return 0 if result.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
