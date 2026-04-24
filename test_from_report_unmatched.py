import argparse
import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from pyxtal import pyxtal
from pymatgen.core import Molecule, Structure


MATCH_FAILURE_TEXT = "This molecule cannot be matched to the reference"


def _build_pmg_molecule_without_hydrogen(smiles: str) -> Molecule:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES during retry: {smiles}")

    mol_no_h = Chem.RemoveHs(mol)
    mol_no_h = Chem.Mol(mol_no_h)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D

    if AllChem.EmbedMolecule(mol_no_h, params) != 0:
        raise ValueError(f"RDKit failed to embed heavy-atom conformer during retry: {smiles}")

    mmff_props = AllChem.MMFFGetMoleculeProperties(mol_no_h)
    if mmff_props is not None:
        AllChem.MMFFOptimizeMolecule(mol_no_h, mmffVariant="MMFF94")
    else:
        AllChem.UFFOptimizeMolecule(mol_no_h)

    conf = mol_no_h.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol_no_h.GetAtoms()]
    coords = [list(conf.GetAtomPosition(i)) for i in range(mol_no_h.GetNumAtoms())]
    return Molecule(symbols, coords)


def _is_reference_match_failure(exc: Exception) -> bool:
    return MATCH_FAILURE_TEXT in str(exc)


def _write_hydrogen_free_cif(cif: str) -> Path:
    source = Path(cif)
    structure = Structure.from_file(str(source))
    structure_no_h = structure.copy()
    structure_no_h.remove_species(["H"])

    tmp_path = source.with_name(source.stem + ".noH.tmp.cif")
    structure_no_h.to(filename=str(tmp_path))
    return tmp_path


def _ensure_structure_smiles(structure, smiles: str):
    smiles = (smiles or "").strip()
    if not smiles:
        return structure

    for mol in getattr(structure, "molecules", []):
        if getattr(mol, "smile", None) in (None, ""):
            mol.smile = smiles
    return structure


def load_seed_with_hydrogen_retry(cif: str, smiles: str):
    tmp_no_h_cif = None
    last_match_failure = None
    pmg_mol_no_h = _build_pmg_molecule_without_hydrogen(smiles)

    try:
        attempts = [
            (cif, [smiles + ".smi"]),
            (cif, [pmg_mol_no_h]),
        ]

        for attempt_cif, molecules in attempts:
            xtal = pyxtal(molecular=True)
            try:
                xtal.from_seed(attempt_cif, molecules=molecules)
                return _ensure_structure_smiles(xtal, smiles)
            except Exception as exc:
                if not _is_reference_match_failure(exc):
                    raise
                last_match_failure = exc

        tmp_no_h_cif = _write_hydrogen_free_cif(cif)
        xtal = pyxtal(molecular=True)
        try:
            xtal.from_seed(str(tmp_no_h_cif), molecules=[pmg_mol_no_h])
            return _ensure_structure_smiles(xtal, smiles)
        except Exception as exc:
            if not _is_reference_match_failure(exc):
                raise
            last_match_failure = exc
    finally:
        if tmp_no_h_cif is not None:
            try:
                if tmp_no_h_cif.exists():
                    tmp_no_h_cif.unlink()
            except Exception:
                pass

    if last_match_failure is not None:
        raise last_match_failure
    raise RuntimeError("from_seed retry sequence ended without a successful match or captured exception")


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
        help="Scan CIFs and SMILES without running pyxtal.",
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

        if not smiles:
            skipped.append((refcode, "missing_smiles"))
            print(f"[SKIP] missing SMILES for {refcode}")
            continue

        if args.dry_run:
            continue

        try:
            load_seed_with_hydrogen_retry(str(cif_path), smiles)
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
