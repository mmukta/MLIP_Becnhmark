"""
Database class
"""
from pyxtal.db import database, make_entry_from_pyxtal
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

db = database("HEM.db")
data = read_csv('ok.csv')
count = 0
for d in data:
    count += 1
    ref_code, smiles = d
    cif_path = f"ccdc_cifs/{ref_code}.cif"
    xtal = pyxtal(molecular=True)
    print(f"{count:6d} [TRY]", ref_code, smiles)
    try:
        xtal.from_seed(str(cif_path), [smiles + ".smi"])
        print(f"{count:6d} [OK]", ref_code, smiles)
    except ReadSeedError:
        xtal.from_seed(str(cif_path), [smiles + ".smi"], add_H=True)
    xtal.tag = {"smiles": smiles, "csd_code": ref_code, "ccdc_number": "N/A"}
    entry = make_entry_from_pyxtal(xtal)
    db.add(entry)
print("Total number of entries", len(db.codes))

# view structure
c = db.get_pyxtal("CAQLIP")
print(c)
