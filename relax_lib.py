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
import argparse
import traceback
from pathlib import Path
from ase.data import atomic_numbers, covalent_radii
_cached_mace_mp = None
_cached_uma = None
MATCH_FAILURE_TEXT = "This molecule cannot be matched to the reference"


def _tail_text(path, max_lines=80):
    path = Path(path)
    if not path.exists():
        return f"{path}: not found"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"{path}: could not read ({type(exc).__name__}: {exc})"
    return "\n".join(lines[-int(max_lines):])


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


def _ase_forces(atoms):
    try:
        return np.asarray(atoms.get_forces()).tolist()
    except Exception:
        return None


def _force_max(forces):
    if forces is None:
        return None
    values = np.asarray(forces)
    if values.size == 0:
        return None
    return float(np.linalg.norm(values, axis=1).max())


def _force_rms(forces):
    if forces is None:
        return None
    values = np.asarray(forces)
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def _ase_stress_tensor(atoms):
    try:
        return np.asarray(atoms.get_stress(voigt=False)).tolist()
    except TypeError:
        try:
            voigt = np.asarray(atoms.get_stress())
            if voigt.shape == (6,):
                xx, yy, zz, yz, xz, xy = voigt.tolist()
                return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
            return voigt.tolist()
        except Exception:
            return None
    except Exception:
        return None


def _ase_energy(atoms):
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return None


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


def _smiles_from_pyxtal_tag(structure):
    tag = getattr(structure, "tag", None)
    if isinstance(tag, dict):
        for key in ("smiles", "SMILES", "CHIRAL SMILES", "chiral_smiles"):
            value = (tag.get(key) or "").strip()
            if value:
                return value
    return ""


def load_pyxtal_from_database(db_file, code, smiles=None):
    """Load one molecular pyxtal structure from a pyxtal.db database."""
    from pyxtal.db import database

    db_path = Path(db_file).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"pyxtal database not found: {db_path}")

    refcode = str(code).strip()
    if not refcode:
        raise ValueError("Database code/refcode is empty.")

    db = database(str(db_path))
    structure = db.get_pyxtal(refcode)
    if structure is None:
        raise KeyError(f"{refcode!r} was not found in {db_path}")

    smiles = (smiles or "").strip() or _smiles_from_pyxtal_tag(structure)
    return _ensure_structure_smiles(structure, smiles)


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


def load_relaxation_structure(cif=None, smiles=None, db_file=None, code=None):
    if db_file:
        if not code:
            raise ValueError("code/refcode is required when loading from a pyxtal database.")
        return load_pyxtal_from_database(db_file, code, smiles)

    if not cif:
        raise ValueError("Either cif or db_file must be provided.")
    if not (smiles or "").strip():
        raise ValueError("smiles is required when loading directly from a CIF.")
    return load_seed_with_hydrogen_retry(cif, smiles)


def prepare_structure_for_relaxation(structure):
    pmg0 = structure.to_pymatgen()
    if structure.has_special_site():
        subgroup = structure.to_subgroup()
        pmg = subgroup.to_pymatgen()
        if get_pmg_dist(pmg0, pmg) > 0.1:
            site = subgroup.mol_sites[0]
            site.rotate(ax_id=2, angle=180)
            pmg = subgroup.to_pymatgen()
            if get_pmg_dist(pmg0, pmg) > 0.1:
                raise RuntimeError("The reference structure is not a valid subgroup.")
        return subgroup
    return structure


def run_ff_then_ml_relax(
    structure,
    refcode,
    out_dir,
    ff_style="openff",
    ml_calculator="MACEOFF",
    ff_charge_method="am1bcc",
    opt_lat=True,
    ff_fmax=0.05,
    ml_fmax=0.05,
    ff_max_steps=5000,
    ml_max_steps=5000,
    ff_nproc=1,
    charmm_info=None,
):
    refcode = str(refcode).strip()
    ff_style = str(ff_style).strip().lower()
    ml_calculator = str(ml_calculator).strip().upper()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    structure = prepare_structure_for_relaxation(structure)

    ff_log = out_dir / f"{refcode}_fire_{ff_style}.log"
    ml_suffix = ml_calculator.lower()
    ml_log = out_dir / f"{refcode}_fire_{ff_style}_{ml_suffix}.log"
    ff_cif = out_dir / f"{refcode}_prerelaxed_{ff_style}.cif"
    ml_cif = out_dir / f"{refcode}_relaxed_{ff_style}_{ml_suffix}.cif"

    ff_opt = FF_optimizer(
        structure,
        ff_style=ff_style,
        chargemethod=ff_charge_method,
        opt_lat=False,
        logfile=str(ff_log),
        workdir=str(out_dir / f"ff_work_{ff_style}"),
        nproc=int(ff_nproc),
        charmm_info=charmm_info,
    )
    ff_opt.run(fmax_target=float(ff_fmax), max_steps=int(ff_max_steps))
    structure = ff_opt.structure
    structure.to_file(str(ff_cif))

    ml_opt = ASE_optimizer(
        structure,
        calculator=ml_calculator,
        opt_lat=True,
        logfile=str(ml_log),
    )
    ml_opt.run(fmax_target=float(ml_fmax), max_steps=int(ml_max_steps))
    structure.to_file(str(ml_cif))

    return {
        "refcode": refcode,
        "ff_style": ff_style,
        "ml_calculator": ml_calculator,
        "status": "OK" if bool(ml_opt.optimized) else "TIMEOUT",
        "energy": float(structure.energy),
        "ff_cif": str(ff_cif),
        "out_cif": str(ml_cif),
        "ff_log": str(ff_log),
        "ml_log": str(ml_log),
        "ff_converged": bool(ff_opt.optimized),
        "ff_steps": int(ff_opt.nsteps),
        "ff_seconds": float(ff_opt.cputime),
        "ml_converged": bool(ml_opt.optimized),
        "ml_steps": int(ml_opt.nsteps),
        "ml_seconds": float(ml_opt.cputime),
        "initial_energy": ml_opt.initial_energy,
        "final_energy": ml_opt.final_energy,
        "initial_fmax": ml_opt.initial_fmax,
        "final_fmax": ml_opt.final_fmax,
        "initial_force_rms": ml_opt.initial_force_rms,
        "final_force_rms": ml_opt.final_force_rms,
        "initial_stress": ml_opt.initial_stress,
        "final_stress": ml_opt.final_stress,
    }


def run_ml_relax(
    structure,
    refcode,
    out_dir,
    ml_calculator="MACE",
    opt_lat=True,
    ml_fmax=0.05,
    ml_max_steps=5000,
):
    refcode = str(refcode).strip()
    ml_calculator = str(ml_calculator).strip().upper()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    structure = prepare_structure_for_relaxation(structure)

    ml_suffix = ml_calculator.lower()
    ml_log = out_dir / f"{refcode}_fire_{ml_suffix}.log"
    ml_cif = out_dir / f"{refcode}_relaxed_{ml_suffix}.cif"

    ml_opt = ASE_optimizer(
        structure,
        calculator=ml_calculator,
        opt_lat=opt_lat,
        logfile=str(ml_log),
    )
    ml_opt.run(fmax_target=float(ml_fmax), max_steps=int(ml_max_steps))
    structure.to_file(str(ml_cif))

    return {
        "refcode": refcode,
        "ff_style": "none",
        "ml_calculator": ml_calculator,
        "status": "OK" if bool(ml_opt.optimized) else "TIMEOUT",
        "energy": float(structure.energy),
        "ff_cif": "",
        "out_cif": str(ml_cif),
        "ff_log": "",
        "ml_log": str(ml_log),
        "ff_converged": False,
        "ff_steps": 0,
        "ff_seconds": 0.0,
        "ml_converged": bool(ml_opt.optimized),
        "ml_steps": int(ml_opt.nsteps),
        "ml_seconds": float(ml_opt.cputime),
        "initial_energy": ml_opt.initial_energy,
        "final_energy": ml_opt.final_energy,
        "initial_fmax": ml_opt.initial_fmax,
        "final_fmax": ml_opt.final_fmax,
        "initial_force_rms": ml_opt.initial_force_rms,
        "final_force_rms": ml_opt.final_force_rms,
        "initial_stress": ml_opt.initial_stress,
        "final_stress": ml_opt.final_stress,
    }


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
    Perform a CHARMM molecular-crystal pre-relaxation and keep the result as a
    pyxtal structure for the downstream MLP optimizer.

    Args:
        struc: pyxtal object
        ff_style (str): 'gaff' or 'openff'
        chargemethod (str): charge assignment method understood by pyocse
        opt_lat (bool): kept for API compatibility; ignored because CHARMM pre-relaxes atomic positions only
        logfile (str): FIRE log file
        workdir (str | None): scratch folder for CHARMM inputs
        nproc (int): kept for compatibility with older callers
    """

    def __init__(
        self,
        struc,
        ff_style="openff",
        chargemethod="am1bcc",
        opt_lat=False,
        logfile="TICJUN/ff_TICJUN.log",
        workdir=None,
        nproc=1,
        charmm_info=None,
    ):
        self.structure = struc
        self.ff_style = str(ff_style).strip().lower()
        self.chargemethod = str(chargemethod).strip()
        self.opt_lat = False
        self.logfile = logfile
        self.workdir = workdir
        self.nproc = max(1, int(nproc))
        self.charmm_info = charmm_info
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

    def run(self, fmax_target=0.05, max_steps=5000):
        t0 = time()
        workdir = self._resolve_workdir()
        for i, mol in enumerate(self.structure.molecules):
            smi = getattr(mol, "smile", None)
            if smi in (None, ""):
                raise ValueError(
                    f"Missing SMILES on structure.molecules[{i}] before FF setup; "
                    "ensure seed-loading fallback populates molecule.smile"
                )

        from pyxtal.interface.charmm import CHARMM

        prefix = "charmm"
        if self.charmm_info is not None:
            try:
                with open(workdir / f"{prefix}.prm", "w") as f:
                    f.write(self.charmm_info["prm"])
                with open(workdir / f"{prefix}.rtf", "w") as f:
                    f.write(self.charmm_info["rtf"])
                atom_info = self.charmm_info["atom_info"]
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load cached CHARMM force-field info "
                    f"for ff_style={self.ff_style} in {workdir}\n"
                    f"exception: {type(exc).__name__}: {exc!r}\n"
                    f"{traceback.format_exc()}"
                ) from exc
        else:
            try:
                ase_with_ff = self.structure.get_forcefield(
                    ff_style=self.ff_style,
                    code="charmm",
                    chargemethod=self.chargemethod,
                )
                with open(workdir / f"{prefix}.prm", "w") as f:
                    ase_with_ff.write_prm(f)
                with open(workdir / f"{prefix}.rtf", "w") as f:
                    ase_with_ff.write_rtf(f)
                atom_info = ase_with_ff.get_atom_info()
            except Exception as exc:
                smiles = [
                    getattr(mol, "smile", "")
                    for mol in getattr(self.structure, "molecules", [])
                ]
                raise RuntimeError(
                    "Failed to generate CHARMM force-field files "
                    f"for ff_style={self.ff_style}, chargemethod={self.chargemethod}, "
                    f"workdir={workdir}\n"
                    f"molecule_smiles={smiles}\n"
                    f"exception: {type(exc).__name__}: {exc!r}\n"
                    f"{traceback.format_exc()}"
                ) from exc
        steps = [int(max_steps)]
        try:
            calc = CHARMM(
                self.structure,
                prefix=prefix,
                atom_info=atom_info,
                folder=str(workdir),
                steps=steps,
                exe=os.environ.get("CHARMM_EXE", "charmm"),
            )
            calc.run(clean=False)
        except Exception as exc:
            raise RuntimeError(
                "CHARMM FF pre-relaxation crashed "
                f"for ff_style={self.ff_style}, workdir={workdir}\n"
                f"exception: {type(exc).__name__}: {exc!r}\n"
                f"{traceback.format_exc()}"
            ) from exc
        if calc.error:
            output = workdir / "_charmm.log"
            dump = workdir / "_result.pdb"
            inp = workdir / "_charmm.in"
            reason = (
                "CHARMM FF pre-relaxation failed after CHARMM execution "
                f"for ff_style={self.ff_style}\n"
                "PyXtal/CHARMM set calc.error=True while reading the CHARMM result. "
                "This usually means the CHARMM-relaxed coordinates could not be "
                "updated back onto the molecular pyxtal object, for example due to "
                "connectivity, lattice, or Wyckoff-site consistency checks.\n"
                f"workdir={workdir}\n"
                f"input={inp} exists={inp.exists()}\n"
                f"output={output} exists={output.exists()}\n"
                f"result_pdb={dump} exists={dump.exists()}\n"
                f"charmm_optimized={getattr(calc, 'optimized', None)}\n"
                f"structure_energy={getattr(calc.structure, 'energy', None)}\n"
                f"last_charmm_log_lines:\n{_tail_text(output)}"
            )
            raise RuntimeError(reason)

        self.structure = calc.structure
        self.cell = getattr(calc, "cell", None)
        self.nsteps = int(getattr(self.structure, "iter", 0) or sum(steps))
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
        self.initial_stress = None
        self.final_stress = None
        self.initial_fmax = None
        self.final_fmax = None
        self.initial_force_rms = None
        self.final_force_rms = None
        self.initial_energy = None
        self.final_energy = None
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
        self.initial_energy = _ase_energy(s)
        initial_forces = _ase_forces(s)
        self.initial_fmax = _force_max(initial_forces)
        self.initial_force_rms = _force_rms(initial_forces)
        self.initial_stress = _ase_stress_tensor(s)
    
        obj = UnitCellFilter(s) if self.opt_lat else s
        dyn = FIRE(obj, a=0.01, logfile=self.logfile)
        converged = dyn.run(fmax=fmax_target, steps=max_steps)
        self.nsteps = int(getattr(dyn, "nsteps", 0))
        self.reached_max_steps = (not bool(converged)) and (self.nsteps >= int(max_steps))
        self.final_energy = _ase_energy(s)
        final_forces = _ase_forces(s)
        self.final_fmax = _force_max(final_forces)
        self.final_force_rms = _force_rms(final_forces)
        self.final_stress = _ase_stress_tensor(s)
        self.forces = final_forces
        self.stress = self.final_stress

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


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Relax one structure with a CHARMM position-only pre-relaxation followed by an ML calculator."
    )
    parser.add_argument("code", nargs="?", help="Refcode/code when using --db-file.")
    parser.add_argument("--db-file", default="", help="pyxtal database file, for example HEM.db.")
    parser.add_argument("--cif", default="", help="Input CIF. Used when --db-file is not set.")
    parser.add_argument("--smiles", default="", help="SMILES for CIF loading, or DB molecule override.")
    parser.add_argument("--out-dir", default="single_charmm_maceoff", help="Output directory.")
    parser.add_argument("--ff-style", default="openff", choices=["gaff", "openff"], help="FF pre-relaxation style.")
    parser.add_argument("--ff-charge-method", default="am1bcc", help="Charge method for FF setup.")
    parser.add_argument(
        "--ml-calculator",
        default="MACEOFF",
        choices=["MACE", "MACEOFF", "UMA", "ANI"],
        help="ML calculator used after FF pre-relaxation.",
    )
    parser.add_argument("--ff-fmax", type=float, default=0.05, help="FF force convergence target.")
    parser.add_argument("--ml-fmax", type=float, default=0.05, help="ML force convergence target.")
    parser.add_argument("--ff-max-steps", type=int, default=5000, help="Maximum CHARMM position-only relaxation steps.")
    parser.add_argument("--ml-max-steps", type=int, default=5000, help="Maximum ML relaxation steps.")
    parser.add_argument("--ff-nproc", type=int, default=1, help="Compatibility option for older FF workflows.")
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    args = _parse_cli_args()
    code = (args.code or "").strip()
    if args.db_file and not code:
        raise ValueError("Provide a code/refcode when using --db-file.")

    structure = load_relaxation_structure(
        cif=args.cif or None,
        smiles=args.smiles,
        db_file=args.db_file or None,
        code=code or None,
    )
    refcode = code or Path(args.cif).stem

    result = run_ff_then_ml_relax(
        structure,
        refcode=refcode,
        out_dir=args.out_dir,
        ff_style=args.ff_style,
        ml_calculator=args.ml_calculator,
        ff_charge_method=args.ff_charge_method,
        opt_lat=True,
        ff_fmax=args.ff_fmax,
        ml_fmax=args.ml_fmax,
        ff_max_steps=args.ff_max_steps,
        ml_max_steps=args.ml_max_steps,
        ff_nproc=args.ff_nproc,
    )

    summary_json = Path(args.out_dir) / f"{refcode}_{args.ff_style}_{args.ml_calculator.lower()}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)

    print(f"refcode={result['refcode']}")
    print(f"pipeline={result['ff_style']}+{result['ml_calculator']}")
    print(f"status={result['status']}")
    print(f"energy={result['energy']}")
    print(f"ff_cif={result['ff_cif']}")
    print(f"out_cif={result['out_cif']}")
    print(f"summary_json={summary_json}")
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
