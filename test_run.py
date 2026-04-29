from pyxtal.db import database
import os
import json
from pyxtal.interface.charmm import CHARMM
from pymatgen.analysis.structure_matcher import StructureMatcher
from time import time

matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=5)

#code = 'RAYFEF'
#code = 'ABAFAO'
if not os.path.exists("Fails"): os.mkdir("Fails")
if not os.path.exists("TMP"): os.mkdir("TMP")


db = database("HEM.db")
codes = db.get_all_codes()[:10]
for code in codes:
    row = db.get_row(code)
    struc = db.get_pyxtal(code)
    if struc.has_special_site(): struc = struc.to_subgroup()
    struc0 = struc.copy()
    pmg0 = struc.to_pymatgen()
    pmg0.remove_species("H")

    t0 = time()
    if hasattr(row, 'charmm_info') and row.charmm_info is not None:
        #print(f"\nUsing cached CHARMM info for {code}")
        charmm_info = json.loads(row.charmm_info)
        atom_info = charmm_info['atom_info']
        with open('TMP/charmm.prm', 'w') as f: f.write(charmm_info['prm'])
        with open('TMP/charmm.rtf', 'w') as f: f.write(charmm_info['rtf'])
    else:
        print(f"\nGenerating new CHARMM info for {code}")
        ase_with_ff = struc.get_forcefield(ff_style="openff",
                                           code="charmm",
                                           chargemethod='am1bcc',
                                           )
        ase_with_ff.write_prm(open('TMP/charmm.prm', 'w'))
        ase_with_ff.write_rtf(open('TMP/charmm.rtf', 'w'))
        atom_info = ase_with_ff.get_atom_info()
        # read the prm file strings and save it to a variable
        with open('TMP/charmm.prm', 'r') as f: prm_str = f.read()
        with open('TMP/charmm.rtf', 'r') as f: rtf_str = f.read()
        charmm_info = {
            'prm': prm_str,
            'rtf': rtf_str,
            'atom_info': atom_info,
        }
        db.db.update(row.id, charmm_info=json.dumps(charmm_info))


    #print(f"Time taken for {code}: {time() - t0:.2f} seconds")

    # Relaxation with CHARMM without optimizating cell parameters
    calc = CHARMM(struc, atom_info=atom_info, prefix='charmm', steps=[5000], folder='TMP')
    calc.run()#clean=False)
    #print(calc.structure)
    xtal = calc.structure
    #calc.structure.to_file('opt.cif')

    # Relaxation with MACE/MACEOFF/UMA with optimizating cell parameters
    # Compute the 
    pmg1 = xtal.to_pymatgen()
    pmg1.remove_species("H")

    # Compare the two structures
    match = matcher.fit(pmg0, pmg1)
    if match:
        d1, d2 = matcher.get_rms_dist(pmg0, pmg1) # RMSD and maximum displacement
        strs = f"{row.id:<4d}{code:<8s} {struc.group.number:3d} Match, RMSD: {d1:.4f}, {d2:.4f}"
    else:
        strs = f"{row.id:<4d}{code:<8s} {struc.group.number:3d} No Match"
        struc0.to_file(f"Fails/{code}_raw.cif")
        xtal.to_file(f"Fails/{code}_opt.cif")
        #import sys; sys.exit(1)
    print(strs)