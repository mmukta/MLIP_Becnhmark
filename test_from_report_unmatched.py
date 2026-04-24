import argparse
import csv
from pathlib import Path

from pyxtal import pyxtal


def load_smiles_map(csv_path: Path) -> dict[str, str]:
    smiles_map: dict[str, str] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            refcode = (row.get("ccdc_id") or "").strip()
            smiles = (row.get("CHIRAL SMILES") or "").strip()
            if refcode and smiles and refcode not in smiles_map:
                smiles_map[refcode] = smiles

    return smiles_map


def resolve_csv_file(base_dir: Path, explicit_csv: str | None) -> Path:
    if explicit_csv:
        csv_path = Path(explicit_csv)
        return csv_path if csv_path.is_absolute() else (Path.cwd() / csv_path).resolve()

    cleaned_csv = base_dir / "entire_data_cleaned.csv"
    if cleaned_csv.exists():
        return cleaned_csv.resolve()
    return (base_dir / "entire_data.csv").resolve()


def resolve_cif_dir(base_dir: Path, explicit_cif_dir: str | None) -> Path:
    if explicit_cif_dir:
        cif_dir = Path(explicit_cif_dir)
        return cif_dir if cif_dir.is_absolute() else (Path.cwd() / cif_dir).resolve()
    return (base_dir / "ccdc_cifs").resolve()


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run pyxtal.from_seed one-by-one for all CIFs in ccdc_cifs."
    )
    parser.add_argument(
        "--csv-file",
        help="CSV containing ccdc_id and CHIRAL SMILES columns. Defaults to entire_data_cleaned.csv, then entire_data.csv.",
    )
    parser.add_argument(
        "--cif-dir",
        help="Directory containing input CIF files. Defaults to ./ccdc_cifs next to this script.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of CIFs to process (0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print generated calls, do not run pyxtal.",
    )
    args = parser.parse_args()

    csv_file = resolve_csv_file(script_dir, args.csv_file)
    cif_dir = resolve_cif_dir(script_dir, args.cif_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF directory not found: {cif_dir}")

    smiles_map = load_smiles_map(csv_file)
    cif_paths = sorted(cif_dir.glob("*.cif"))

    if args.limit > 0:
        cif_paths = cif_paths[: args.limit]

    print(f"CSV file: {csv_file}")
    print(f"CIF directory: {cif_dir}")
    print(f"Total CIFs selected: {len(cif_paths)}")

    ok = []
    failed = []
    skipped = []

    for cif_path in cif_paths:
        refcode = cif_path.stem
        smiles = smiles_map.get(refcode, "")
        print(f"\n({refcode}, {smiles or 'NO_SMILES'})")

        if not smiles:
            skipped.append((refcode, "missing_smiles"))
            print(f"[SKIP] missing SMILES for {refcode}")
            continue

        call_repr = f'xtal.from_seed("{cif_path}", ["{smiles}.smi"])'
        print(call_repr)

        if args.dry_run:
            continue

        try:
            xtal = pyxtal(molecular=True)
            xtal.from_seed(str(cif_path), [smiles + ".smi"])
            print("[OK]", refcode)
            ok.append(refcode)
        except Exception as exc:
            failed.append((refcode, str(exc)))
            print(f"[FAIL] {refcode}: {exc}")

    print("\nSummary")
    print("OK:", len(ok))
    print("FAIL:", len(failed))
    print("SKIP:", len(skipped))

    if failed:
        print("\nFailed refcodes:")
        for refcode, reason in failed:
            print(refcode, "|", reason)

    if skipped:
        print("\nSkipped refcodes:")
        for refcode, reason in skipped:
            print(refcode, "|", reason)


if __name__ == "__main__":
    main()
