import argparse
import csv
from pathlib import Path
from pyxtal import pyxtal
from pyxtal.msg import ReadSeedError
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def read_csv(csv_path):
    smiles_map = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            refcode = row.get("ccdc_id")
            smiles = row.get("CHIRAL SMILES")
            if refcode and smiles:
                smiles_map.append((refcode, smiles))
    return smiles_map


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run pyxtal.from_seed one-by-one for all CIFs in ccdc_cifs."
    )
    parser.add_argument(
        "--csv-file",
        default="entire_data.csv",
        help="CSV containing ccdc_id and CHIRAL SMILES columns. Defaults to entire_data_cleaned.csv, then entire_data.csv.",
    )
    parser.add_argument(
        "--cif-dir",
        default='./ccdc_cifs',
        help="Directory containing input CIF files. Defaults to ./ccdc_cifs next to this script.",
    )
    args = parser.parse_args()


    print(f"CSV file: {args.csv_file}")
    print(f"CIF directory: {args.cif_dir}")
    data = read_csv(args.csv_file)
    fail_csv_path = script_dir / "fail_read.csv"
    ok_csv_path = script_dir / "ok.csv"

    ok = []
    failed = []

    with fail_csv_path.open("w", encoding="utf-8", newline="") as fail_handle, ok_csv_path.open("w", encoding="utf-8", newline="") as ok_handle:
        fail_writer = csv.writer(fail_handle)
        ok_writer = csv.writer(ok_handle)
        fail_writer.writerow(["ccdc_id", "CHIRAL SMILES"])
        ok_writer.writerow(["ccdc_id", "CHIRAL SMILES"])
        fail_handle.flush()
        ok_handle.flush()

        count = 0
        for d in data:
            ref_code, smiles = d
            cif_path = Path(args.cif_dir) / f"{ref_code}.cif"
            if not cif_path.is_file():
                print(f"{count:6d} [SKIP] CIF not found for {ref_code}")
                continue
            count += 1
            xtal = pyxtal(molecular=True)
            if len(smiles) > 0:
                print(f"{count:6d} [TRY]", ref_code, smiles)
                try:
                    xtal.from_seed(str(cif_path), [smiles + ".smi"])
                    print(f"{count:6d} [OK]", ref_code, smiles)
                    ok.append(ref_code)
                    ok_writer.writerow([ref_code, smiles])
                    ok_handle.flush()
                except ReadSeedError:
                    try:
                        xtal.from_seed(str(cif_path), [smiles + ".smi"], add_H=True)
                        ok.append(ref_code)
                        ok_writer.writerow([ref_code, smiles])
                        ok_handle.flush()
                        print(f"{count:6d} [OK after ignoring H]", ref_code, smiles)
                    except Exception:
                        failed.append((ref_code, smiles))
                        fail_writer.writerow([ref_code, smiles])
                        fail_handle.flush()
                        print(f"{count:6d} [FAIL] {ref_code} {smiles}")
                #except Exception:
                #    print(f"{count:6d} [Other] {ref_code} {smiles}")
                #    import sys; sys.exit()

    print("\nSummary")
    print("OK:", len(ok))
    print("FAIL:", len(failed))
    print(f"ok.csv written: {ok_csv_path}")
    print(f"fail_read.csv written: {fail_csv_path}")

if __name__ == "__main__":
    main()
