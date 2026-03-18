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

ID_COL = "ccdc_id"
SMILES_COL = "CHIRAL SMILES"
DEFAULT_CALCULATOR = os.environ.get("CALCULATOR", "MACEOFF").strip().upper() or "MACEOFF"


def calculator_suffix(calculator: str) -> str:
    value = calculator.strip().upper()
    if value == "MACEOFF":
        return "maceoff"
    if value == "MACE":
        return "mace"
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
    if os.path.exists(fail_seed):
        return fail_seed
    return original


def worker_relax_task(
    refcode: str,
    smiles: str,
    calculator: str,
    seed_cif: str,
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
    cif_in = seed_cif
    relaxation_started = False

    def _handle_terminate(signum, frame):
        out_cif = save_structure_snapshot(timeout_partial_cif_path(refcode, calculator), refcode, smiles, c)
        raise SystemExit(128 + int(signum) + (1 if out_cif else 0))

    signal.signal(signal.SIGTERM, _handle_terminate)
    signal.signal(signal.SIGINT, _handle_terminate)

    try:
        import torch

        torch.set_num_threads(1)

        from rdkit import Chem
        from pyxtal import pyxtal
        from relax_lib import ASE_optimizer

        os.makedirs(ref_out_dir(refcode), exist_ok=True)

        if not os.path.exists(cif_in):
            raise FileNotFoundError(f"CIF not found: {cif_in}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed SMILES: {smiles}")

        c = pyxtal(molecular=True)
        c.from_seed(cif_in, molecules=[smiles + ".smi"])

        calc = ASE_optimizer(
            c,
            calculator=calculator,
            opt_lat=True,
            logfile=os.path.join(ref_out_dir(refcode), f"fire_{calculator_suffix(calculator)}.log"),
        )
        relaxation_started = True
        calc.run(fmax_target=0.05, max_steps=max_relax_steps)

        if bool(calc.optimized):
            out_cif = relaxed_cif_path(refcode, calculator)
            c.to_file(out_cif)
            prepend_smiles_to_cif(out_cif, refcode, smiles)
            res = {
                "status": "OK",
                "refcode": refcode,
                "energy": float(c.energy),
                "converged": True,
                "reason": "",
                "steps": int(getattr(calc, "nsteps", 0)),
                "max_steps": int(max_relax_steps),
                "out_cif": out_cif,
                "seed_cif": cif_in,
                "seconds": time.time() - t0,
                "relaxation_started": True,
                "calculator": calculator,
            }
        else:
            out_cif = save_structure_snapshot(timeout_partial_cif_path(refcode, calculator), refcode, smiles, c)
            res = {
                "status": "TIMEOUT",
                "refcode": refcode,
                "energy": float(c.energy),
                "converged": False,
                "reason": f"Reached max relaxation steps ({max_relax_steps})",
                "steps": int(getattr(calc, "nsteps", 0)),
                "max_steps": int(max_relax_steps),
                "out_cif": out_cif,
                "seed_cif": cif_in,
                "seconds": time.time() - t0,
                "relaxation_started": True,
                "calculator": calculator,
            }
        result_q.put(res)

    except Exception as e:
        fail_snapshot = ""
        if relaxation_started:
            fail_snapshot = save_structure_snapshot(fail_snapshot_cif_path(refcode, calculator), refcode, smiles, c)
        res = {
            "status": "FAIL",
            "refcode": refcode,
            "reason": str(e),
            "traceback": traceback.format_exc(),
            "seconds": time.time() - t0,
            "seed_cif": cif_in,
            "out_cif": fail_snapshot,
            "relaxation_started": relaxation_started,
            "calculator": calculator,
        }
        result_q.put(res)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume one target from the FAIL/<refcode>/ folder."
    )
    parser.add_argument("refcode", help="Target refcode, for example FANTIX")
    parser.add_argument(
        "--calculator",
        choices=["MACE", "MACEOFF"],
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
        help="CSV file used to resolve SMILES.",
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

    os.makedirs(FAIL_DIR, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(ref_out_dir(refcode), exist_ok=True)

    smiles_map = load_smiles_map(args.csv_file)
    smiles = smiles_map.get(refcode, "")
    if not smiles:
        raise ValueError(f"SMILES not found for refcode {refcode} in {args.csv_file}")

    seed_cif = resolve_seed_cif(refcode, calculator, args.mode, args.seed_cif or None)
    if not os.path.exists(seed_cif):
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
        args=(refcode, smiles, calculator, seed_cif, int(args.max_relax_steps), result_q),
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
        }

    try:
        proc.join(timeout=0)
    except Exception:
        pass

    result["smiles"] = smiles
    result["requested_mode"] = args.mode
    result["resolved_seed_cif"] = seed_cif
    result["max_relax_steps"] = int(args.max_relax_steps)
    result["timeout_sec"] = int(args.timeout_sec)

    write_json_atomic(result_json, result)
    append_jsonl(report_jsonl, result)

    print(f"refcode={refcode}")
    print(f"calculator={calculator}")
    print(f"mode={args.mode}")
    print(f"seed_cif={seed_cif}")
    print(f"status={result.get('status', 'FAIL')}")
    print(f"reason={result.get('reason', '')}")
    print(f"out_cif={result.get('out_cif', '')}")
    print(f"result_json={result_json}")
    print(f"report_jsonl={report_jsonl}")
    return 0 if result.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
