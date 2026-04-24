import torch
# --- Temporary fix for PyTorch 2.6 weights_only change ---
from torch.serialization import add_safe_globals
add_safe_globals([slice])   # allow 'slice' to be unpickled
# ----------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read, write
#from fairchem.core import pretrained_mlip, FAIRChemCalculator
import signal
from time import time
import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter
from ase.optimize.fire import FIRE
import logging
from pyxtal.optimize import WFS, DFS, QRS
from pyxtal import pyxtal
from pyxtal.util import get_pmg_dist
from pymatgen.core import Molecule, Structure
import os
import json
from pathlib import Path
from ase.data import atomic_numbers, covalent_radii
_cached_mace_mp = None
_cached_uma = None
MATCH_FAILURE_TEXT = "This molecule cannot be matched to the reference"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def _resolve_uma_checkpoint() -> str:
    candidates = []

    env_path = (os.environ.get("UMA_CKPT_PATH") or "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            Path("/users/mmukta/Github/PyXtal/pyxtal/interface/uma-s-1p1.pt"),
            Path("/scratch/mmukta/uma-s-1p1.pt"),
            Path.cwd() / "uma-s-1p1.pt",
        ]
    )

    for path in candidates:
        if path.exists():
            return str(path)

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "UMA checkpoint not found. Set UMA_CKPT_PATH or place uma-s-1p1.pt in a known location. "
        f"Checked: {checked}"
    )


def get_calculator(calculator):
    global _cached_mace_mp, _cached_uma

    if type(calculator) is str:
        if calculator == 'ANI':
            import torchani
            calc = torchani.models.ANI2x().ase()

        elif calculator == 'MACE':
            if _cached_mace_mp is None:
                from mace.calculators import mace_mp
                _cached_mace_mp = mace_mp(
                    model='small',
                    dispersion=True,
                    device='cpu'
                )
            calc = _cached_mace_mp

        elif calculator == 'MACEOFF':
            from mace.calculators import mace_off
            _cached_mace = mace_off(model='medium', device='cpu')
            calc = _cached_mace
            
        elif calculator == "UMA":
            if _cached_uma is None:
                from fairchem.core import pretrained_mlip, FAIRChemCalculator
                from fairchem.core.units.mlip_unit import load_predict_unit
                ckpt_path = _resolve_uma_checkpoint()
                # Load from a local checkpoint so batch runs stay offline and reproducible.
                torch.load(ckpt_path, map_location="cpu", weights_only=False)
                predictor = load_predict_unit(ckpt_path, device="cpu")
                _cached_uma = FAIRChemCalculator(predictor, task_name="omc")
            calc = _cached_uma
            
        else:
            raise ValueError(f"Unknown calculator: {calculator}")
            
    else:
        calc = calculator

    return calc


def _build_pmg_molecule_without_hydrogen(smiles):
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


def _is_reference_match_failure(exc):
    return MATCH_FAILURE_TEXT in str(exc)


def _write_hydrogen_free_cif(cif):
    source = Path(cif)
    structure = Structure.from_file(str(source))
    structure_no_h = structure.copy()
    structure_no_h.remove_species(["H"])

    tmp_path = source.with_name(source.stem + ".noH.tmp.cif")
    structure_no_h.to(filename=str(tmp_path))
    return tmp_path


def _ensure_structure_smiles(structure, smiles):
    smiles = (smiles or "").strip()
    if not smiles:
        return structure

    for mol in getattr(structure, "molecules", []):
        if getattr(mol, "smile", None) in (None, ""):
            mol.smile = smiles
    return structure


def load_seed_with_hydrogen_retry(cif, smiles):
    tmp_no_h_cif = None
    last_match_failure = None
    pmg_mol_no_h = _build_pmg_molecule_without_hydrogen(smiles)

    try:
        attempts = [
            ("default_smiles", cif, [smiles + ".smi"]),
            ("no_h_smiles", cif, [pmg_mol_no_h]),
        ]

        for _label, attempt_cif, molecules in attempts:
            c = pyxtal(molecular=True)
            try:
                c.from_seed(attempt_cif, molecules=molecules)
                return _ensure_structure_smiles(c, smiles)
            except Exception as exc:
                if not _is_reference_match_failure(exc):
                    raise
                last_match_failure = exc

        tmp_no_h_cif = _write_hydrogen_free_cif(cif)
        c = pyxtal(molecular=True)
        try:
            c.from_seed(str(tmp_no_h_cif), molecules=[pmg_mol_no_h])
            return _ensure_structure_smiles(c, smiles)
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


def update_structure_from_ase_atoms(
    structure,
    atoms,
    save_connectivity_failure_snapshot=None,
    save_connectivity_report=None,
):
    positions = atoms.get_scaled_positions()
    structure.lattice.set_matrix(atoms.get_cell())

    count = 0
    for _i, site in enumerate(structure.mol_sites):
        coords0, species = site._get_coords_and_species(first=True)
        coords1 = positions[count : count + len(site.molecule.mol)]
        for j, _coor in enumerate(coords1):
            diff = coords1[j] - coords0[j]
            diff -= np.round(diff)
            abs_diff = np.dot(diff, atoms.get_cell())
            if abs(np.linalg.norm(abs_diff)) < 2.0:
                coords1[j] = coords0[j] + diff
            else:
                print(coords1[j], coords1[j], np.linalg.norm(abs_diff))
                import sys
                sys.exit()
        try:
            site.update(coords1, structure.lattice)
        except ValueError as exc:
            if (
                "molecular connectivity changes" in str(exc)
                and save_connectivity_failure_snapshot is not None
                and save_connectivity_report is not None
            ):
                save_connectivity_failure_snapshot(atoms)
                save_connectivity_report(
                    site_index=_i,
                    site=site,
                    coords0=coords0,
                    coords1=coords1,
                    species=species,
                    cell=atoms.get_cell(),
                    error_text=str(exc),
                )
            raise
        count += len(site.molecule.mol) * site.wp.multiplicity

    structure.optimize_lattice()
    structure.energy = atoms.get_potential_energy()
    return structure


class FF_optimizer:
    """
    Perform a classical FF pre-relaxation for a molecular crystal using the
    same pyxtal structure update path as the MLP optimizer.

    Args:
        struc: pyxtal object
        ff_style (str): 'gaff' or 'openff'
        chargemethod (str): charge assignment method understood by pyocse
        opt_lat (bool): optimize lattice and positions when True
        logfile (str): FIRE log file
        workdir (str | None): scratch folder for LAMMPS inputs
        nproc (int): OMP threads for LAMMPS
    """

    def __init__(
        self,
        struc,
        ff_style="gaff",
        chargemethod="am1bcc",
        opt_lat=True,
        logfile="TICJUN/ff_TICJUN.log",
        workdir=None,
        nproc=1,
    ):
        self.structure = struc
        self.ff_style = str(ff_style).strip().lower()
        self.chargemethod = str(chargemethod).strip()
        self.opt_lat = opt_lat
        self.logfile = logfile
        self.workdir = workdir
        self.nproc = max(1, int(nproc))
        self.optimized = True
        self.cell = None
        self.cputime = 0
        self.nsteps = 0
        self.reached_max_steps = False

    def _resolve_workdir(self) -> Path:
        log_path = Path(self.logfile)
        default_workdir = log_path.parent if log_path.parent != Path("") else Path(".")
        workdir = Path(self.workdir) if self.workdir is not None else default_workdir / f"ff_{self.ff_style}"
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    def run(self, fmax_target=0.05, max_steps=2000):
        t0 = time()
        workdir = self._resolve_workdir()
        smiles_list = []
        for i, mol in enumerate(self.structure.molecules):
            smi = getattr(mol, "smile", None)
            if smi in (None, ""):
                raise ValueError(
                    f"Missing SMILES on structure.molecules[{i}] before FF setup; "
                    "ensure seed-loading fallback populates molecule.smile"
                )
            smiles_list.append(smi)

        from pyocse.parameters import ForceFieldParameters
        from pyxtal.interface.charmm import CHARMM

        params = ForceFieldParameters(
            smiles_list,
            style=self.ff_style,
            chargemethod=self.chargemethod,
            ncpu=self.nproc,
            verbose=False,
        )
        ase_with_ff = params.get_ase_charmm(params.params_init.copy())

        cwd = os.getcwd()
        prefix = "pyxtal_ff"
        os.chdir(workdir)
        try:
            ase_with_ff.write_charmmfiles(base=prefix)
        finally:
            os.chdir(cwd)

        atom_info = ase_with_ff.get_atom_info()
        steps = [max(1, int(max_steps // 2)), int(max_steps)] if self.opt_lat else [int(max_steps)]
        calc = CHARMM(
            self.structure,
            label="_",
            prefix=prefix,
            atom_info=atom_info,
            folder=str(workdir),
            steps=steps,
            exe=os.environ.get("CHARMM_EXE", "charmm"),
        )
        calc.run(clean=False)
        if calc.error:
            raise RuntimeError(f"CHARMM FF pre-relaxation failed for ff_style={self.ff_style}")

        self.structure = calc.structure
        self.cell = getattr(calc, "cell", None)
        self.nsteps = int(sum(steps))
        self.reached_max_steps = not bool(calc.optimized)
        self.optimized = bool(calc.optimized)

        self.cputime = time() - t0

class ASE_optimizer:
    """
    This is a ASE optimizer to perform oragnic crystal structure optimization.
    We assume that the geometry has been well optimized by classical FF.

    Args:
        struc: pyxtal object
        calculator (str): 'ANI', 'MACE'
        opt_lat (bool): to opt lattice or not
        log_file (str): output file
    """

    def __init__(self, struc, calculator="MACE", opt_lat=True, logfile="TICJUN/mace_TICJUN"):
        self.structure = struc
        self.calculator = get_calculator(calculator)
        self.opt_lat = opt_lat
        self.stress = None
        self.forces = None
        self.optimized = True
        self.positions = None
        self.cell = None
        self.cputime = 0
        self.logfile = logfile
        self.nsteps = 0
        self.reached_max_steps = False
        self.failure_snapshot_path = ""

    def _save_connectivity_failure_snapshot(self, atoms):
        base = Path(self.logfile).with_name(f"{Path(self.logfile).stem}_connectivity_failure")
        cif_path = base.with_suffix(".cif")
        xyz_path = base.with_suffix(".xyz")
        try:
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            write(str(cif_path), atoms, format="cif")
            write(str(xyz_path), atoms, format="xyz")
            self.failure_snapshot_path = str(xyz_path)
        except Exception:
            self.failure_snapshot_path = ""

    def _bond_map(self, coords_frac, species, cell, scale=1.15):
        coords_cart = np.dot(coords_frac, np.array(cell))
        bonds = {}
        for i in range(len(species)):
            for j in range(i + 1, len(species)):
                d = float(np.linalg.norm(coords_cart[i] - coords_cart[j]))
                ri = covalent_radii[atomic_numbers[species[i]]]
                rj = covalent_radii[atomic_numbers[species[j]]]
                cutoff = max(0.7, scale * (ri + rj))
                if d <= cutoff:
                    bonds[(i, j)] = d
        return bonds

    def _expected_bonds(self, site):
        mol = site.molecule.mol
        bonds = {}
        for b in mol.get_covalent_bonds(tol=0.25):
            i = mol.index(b.site1)
            j = mol.index(b.site2)
            if i > j:
                i, j = j, i
            ref_d = float(np.linalg.norm(mol.cart_coords[i] - mol.cart_coords[j]))
            bonds[(i, j)] = ref_d
        return bonds

    def _save_connectivity_report(self, site_index, site, coords0, coords1, species, cell, error_text):
        out_dir = Path(self.logfile).parent if Path(self.logfile).parent != Path("") else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "connectivity_change_report.json"
        txt_path = out_dir / "connectivity_change_report.txt"

        expected = self._expected_bonds(site)
        before = self._bond_map(coords0, species, cell)
        after = self._bond_map(coords1, species, cell)
        broken = sorted(set(before) - set(after))
        formed = sorted(set(after) - set(before))
        before_cart = np.dot(coords0, np.array(cell))
        after_cart = np.dot(coords1, np.array(cell))

        expected_distances = []
        broken_expected = []
        for (i, j), ref_d in sorted(expected.items()):
            before_d = float(np.linalg.norm(before_cart[i] - before_cart[j]))
            after_d = float(np.linalg.norm(after_cart[i] - after_cart[j]))
            ri = covalent_radii[atomic_numbers[species[i]]]
            rj = covalent_radii[atomic_numbers[species[j]]]
            cutoff = max(1.15 * (ri + rj), 1.20 * ref_d)
            is_broken = after_d > cutoff

            entry = {
                "atom_i": i + 1,
                "atom_i_element": species[i],
                "atom_j": j + 1,
                "atom_j_element": species[j],
                "template_distance_A": round(ref_d, 4),
                "distance_before_A": round(before_d, 4),
                "distance_after_A": round(after_d, 4),
                "break_cutoff_A": round(cutoff, 4),
                "is_broken_after_relax": is_broken,
            }
            expected_distances.append(entry)

            if is_broken:
                neigh = []
                for k in range(len(species)):
                    if k == j:
                        continue
                    d = float(np.linalg.norm(after_cart[j] - after_cart[k]))
                    neigh.append((d, k))
                neigh.sort(key=lambda x: x[0])
                nearest = [
                    {
                        "atom": k + 1,
                        "element": species[k],
                        "distance_A": round(d, 4),
                    }
                    for d, k in neigh[:3]
                ]
                item = dict(entry)
                item["nearest_neighbors_of_atom_j_after"] = nearest
                broken_expected.append(item)

        def item(pair, dist_before=None, dist_after=None):
            i, j = pair
            return {
                "atom_i": i + 1,
                "atom_i_element": species[i],
                "atom_j": j + 1,
                "atom_j_element": species[j],
                "distance_before_A": None if dist_before is None else round(float(dist_before), 4),
                "distance_after_A": None if dist_after is None else round(float(dist_after), 4),
            }

        report = {
            "site_index": site_index,
            "error": error_text,
            "n_atoms_in_molecule": len(species),
            "n_bonds_before": len(before),
            "n_bonds_after": len(after),
            "n_broken_bonds": len(broken),
            "n_formed_bonds": len(formed),
            "n_expected_bonds": len(expected),
            "n_expected_bonds_broken_after_relax": len(broken_expected),
            "broken_bonds": [item(p, dist_before=before[p], dist_after=after.get(p)) for p in broken],
            "formed_bonds": [item(p, dist_before=before.get(p), dist_after=after[p]) for p in formed],
            "broken_expected_bonds": broken_expected,
            "expected_bond_distances": expected_distances,
        }

        with open(json_path, "w") as f:
            json.dump(_json_safe(report), f, indent=2)

        with open(txt_path, "w") as f:
            f.write(f"Connectivity failure at site index {site_index}\n")
            f.write(f"Error: {error_text}\n")
            f.write(f"Bonds before: {len(before)}, after: {len(after)}\n")
            f.write(f"Broken bonds: {len(broken)}, formed bonds: {len(formed)}\n\n")
            f.write(f"Expected bonds from template: {len(expected)}\n")
            f.write(f"Expected bonds broken after relax: {len(broken_expected)}\n\n")
            if broken:
                f.write("Broken bonds (i-j element_i-element_j d_before[A] -> d_after[A]):\n")
                for p in broken:
                    i, j = p
                    f.write(
                        f"  {i+1}-{j+1} {species[i]}-{species[j]} "
                        f"{before[p]:.4f} -> {after.get(p, float('nan')):.4f}\n"
                    )
            if formed:
                f.write("\nFormed bonds (i-j element_i-element_j d_before[A] -> d_after[A]):\n")
                for p in formed:
                    i, j = p
                    f.write(
                        f"  {i+1}-{j+1} {species[i]}-{species[j]} "
                        f"{before.get(p, float('nan')):.4f} -> {after[p]:.4f}\n"
                    )
            if broken_expected:
                f.write("\nExpected bonds broken after relax:\n")
                for b in broken_expected:
                    f.write(
                        f"  {b['atom_i']}-{b['atom_j']} "
                        f"{b['atom_i_element']}-{b['atom_j_element']} "
                        f"template={b['template_distance_A']:.4f} "
                        f"before={b['distance_before_A']:.4f} "
                        f"after={b['distance_after_A']:.4f} "
                        f"cutoff={b['break_cutoff_A']:.4f}\n"
                    )

    def run(self, fmax_target=0.01, max_steps=5000):
        t0 = time()
        s = self.structure.to_ase(resort=False)
        s.set_constraint(FixSymmetry(s))
        s.set_calculator(self.calculator)
    
        obj = UnitCellFilter(s) if self.opt_lat else s
        dyn = FIRE(obj, a=0.01, logfile=self.logfile)
        converged = dyn.run(fmax=fmax_target, steps=max_steps)
        self.nsteps = int(getattr(dyn, "nsteps", 0))
        self.reached_max_steps = (not bool(converged)) and (self.nsteps >= int(max_steps))

        update_structure_from_ase_atoms(
            self.structure,
            s,
            save_connectivity_failure_snapshot=self._save_connectivity_failure_snapshot,
            save_connectivity_report=self._save_connectivity_report,
        )
        self.cell = s.get_cell()
    
        s.set_calculator()
        s.set_constraint()
        self.cputime = time() - t0
        self.optimized = bool(converged)


if __name__ == "__main__":
    import os, warnings
    from pyxtal.db import database
    warnings.filterwarnings("ignore")

    work_dir = "tmp"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    data = [
    ("/Users/mmukta/Desktop/Nitro/Data_nitro__cifs/TICJUN.cif", "O=N(=O)C1=CN(N=C1N1C=NN=N1)C(N(=O)=O)(N(=O)=O)N(=O)=O")
    ]

    for d in data:
        cif, smiles = d
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    
        mol = Chem.AddHs(mol)
        smiles_H = Chem.MolToSmiles(mol)
    
        c = load_seed_with_hydrogen_retry(cif, smiles)
        
        # --- Build PyXtal structure --

        pmg0 = c.to_pymatgen()
        if c.has_special_site():
            c1 = c.to_subgroup(); print(c1)
            pmg = c1.to_pymatgen()
            if get_pmg_dist(pmg0, pmg) > 0.1:
                print("The reference structure is not a valid subgroup.")
                m = c1.mol_sites[0]
                m.rotate(ax_id=2, angle=180)
                pmg = c1.to_pymatgen()
                print("Distance after flip", get_pmg_dist(pmg0, pmg))
            c = c1
            pmg = c.to_pymatgen()
        else:
            pmg = pmg0
    calc = ASE_optimizer(c)
    print(calc.structure.lattice)
    #calc.run(steps=1500)
    calc.run(fmax_target=0.01)
    print(calc.structure.energy)
    print(calc.structure.lattice)
    calc.structure.to_file("TICJUN/mace.cif")
    from pymatgen.core import Structure
    # Load CIF
    structure = Structure.from_file("TICJUN/mace.cif")
    # Get density in g/cm³
    print("Density:", structure.density, "g/cm³")
    total_molecules = 0
    print("Molecular sites and multiplicities:")
    for i, site in enumerate(c.mol_sites):
        print(f"Site {i+1}: Wyckoff multiplicity = {site.wp.multiplicity}")
        total_molecules += site.wp.multiplicity

    print(f"\nTotal molecules per unit cell: {total_molecules}")
