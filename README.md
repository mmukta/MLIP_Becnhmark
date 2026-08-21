# MLIP_Becnhmark

Small benchmark workspace for relaxing molecular crystals with `MACE`, `MACEOFF`, and `UMA`.

See also: [TOC.pdf](TOC.pdf)

## Environment

Use the shared conda environment:

```bash
conda activate htocsp
```

The scripts expect the environment to provide at least:

- `torch`
- `rdkit`
- `pyxtal`
- `ase`
- `numpy`
- `mace`
- `fairchem`, only for `UMA`
- `charmm`, only when `--ff-style openff` or `--ff-style gaff` is used

For `UMA`, set `UMA_CKPT_PATH` if the checkpoint is not in one of the default locations checked by `relax_lib.py`.

## Calculator Modes

By default, `run_individual.py` runs the selected ML calculator only. It does not run CHARMM or write `charmm.rtf` / `charmm.prm` unless an FF pre-relaxation is explicitly requested.

Calculator-only modes:

```bash
python run_individual.py PUKPOB --calculator MACE --db-file HEM.db --ml-fmax 0.04
python run_individual.py PUKPOB --calculator MACEOFF --db-file HEM.db --ml-fmax 0.04
python run_individual.py PUKPOB --calculator UMA --db-file HEM.db --ml-fmax 0.04
```

Add a CHARMM position-only pre-relaxation by choosing `--ff-style`:

```bash
python run_individual.py PUKPOB --calculator MACE --ff-style openff --db-file HEM.db --ml-fmax 0.04
python run_individual.py PUKPOB --calculator MACE --ff-style gaff --db-file HEM.db --ml-fmax 0.04
```

Supported `--ff-style` values:

- `none`: ML calculator only; this is the default.
- `openff`: CHARMM/OpenFF pre-relaxation, then ML relaxation.
- `gaff`: CHARMM/GAFF pre-relaxation, then ML relaxation.
- `auto`: `MACE`, `MACEOFF`, and `UMA` use `none`; `GAFF_MACE`, `GAFF_MACEOFF`, and `GAFF_UMA` use `gaff`.

The legacy calculator names `GAFF_MACE`, `GAFF_MACEOFF`, and `GAFF_UMA` still mean GAFF pre-relaxation followed by the corresponding ML calculator.

## Inputs And Outputs

Input structures normally live in:

```text
FAIL/<REFCODE>/<REFCODE>.cif
```

When `--db-file HEM.db` is used, the structure and SMILES are loaded from the pyxtal database instead. If no SMILES is found in the database, the script falls back to `entire_data.csv`.

Outputs go to `FAIL/<REFCODE>/` by default. To write outputs somewhere else, set `OUTPUT_DIR`:

```bash
OUTPUT_DIR=output_mace python run_individual.py PUKPOB --calculator MACE --db-file HEM.db --ml-fmax 0.04
```

With that command, outputs are written under:

```text
output_mace/PUKPOB/
output_mace/single_runs/
```

Typical calculator-only output files:

```text
<OUTPUT_DIR>/<REFCODE>/<REFCODE>_fire_mace.log
<OUTPUT_DIR>/<REFCODE>/<REFCODE>_relaxed_mace.cif
<OUTPUT_DIR>/single_runs/<REFCODE>_mace_<TIMESTAMP>.json
```

The JSON summary includes ML observables before and after relaxation:

- `initial_energy` and `final_energy`
- `initial_fmax` and `final_fmax`
- `initial_force_rms` and `final_force_rms`
- `initial_stress` and `final_stress`

For FF+ML runs, these are measured for the ML relaxation stage, after any FF pre-relaxation has completed.

Typical FF+ML output files:

```text
<OUTPUT_DIR>/<REFCODE>/<REFCODE>_fire_openff.log
<OUTPUT_DIR>/<REFCODE>/<REFCODE>_prerelaxed_openff.cif
<OUTPUT_DIR>/<REFCODE>/<REFCODE>_fire_openff_mace.log
<OUTPUT_DIR>/<REFCODE>/<REFCODE>_relaxed_openff_mace.cif
<OUTPUT_DIR>/<REFCODE>/ff_work_openff/charmm.rtf
<OUTPUT_DIR>/<REFCODE>/ff_work_openff/charmm.prm
```

## Resume Modes

Use `--mode` to choose the starting structure when not loading directly from the DB:

- `original`: use `FAIL/<REFCODE>/<REFCODE>.cif`.
- `resume`: require an existing timeout or failure snapshot for the selected calculator.
- `auto`: prefer a timeout snapshot, then a failure snapshot, then the original CIF.

Examples:

```bash
python run_individual.py FANTIX --calculator MACEOFF --mode original
python run_individual.py FANTIX --calculator MACEOFF --mode resume
python run_individual.py FANTIX --calculator MACEOFF --mode auto
```

## Useful Options

```bash
python run_individual.py FANTIX \
  --calculator MACEOFF \
  --mode auto \
  --ff-style none \
  --timeout-sec 14400 \
  --max-relax-steps 5000 \
  --ml-fmax 0.04 \
  --log-tag test
```

Important environment variables:

- `OUTPUT_DIR`: base directory for output CIFs, logs, and default `single_runs` summaries.
- `CALCULATOR`: default calculator if `--calculator` is omitted.
- `FF_STYLE`: default FF style if `--ff-style` is omitted.
- `ML_FMAX`: default ML force target if `--ml-fmax` is omitted.
- `DB_FILE`: default pyxtal database path.
- `CSV_FILE`: fallback CSV for SMILES.

## Local Batch Runs

Use `run_individual_local.sh` to run many refcodes from a methods CSV. The CSV defaults to `t.csv` and should contain `refcode` and `methods` columns.

```bash
METHODS_CSV=methods.csv OUTPUT_DIR=output_mace ML_FMAX=0.04 bash run_individual_local.sh MACE 6
```

Choose the optional FF pre-relaxation:

```bash
METHODS_CSV=methods.csv OUTPUT_DIR=output_mace FF_STYLE=none bash run_individual_local.sh MACE 6
METHODS_CSV=methods.csv OUTPUT_DIR=output_openff_mace FF_STYLE=openff bash run_individual_local.sh MACE 6
METHODS_CSV=methods.csv OUTPUT_DIR=output_gaff_mace FF_STYLE=gaff bash run_individual_local.sh MACE 6
```

The second argument is the number of parallel local jobs.

## Slurm

Use `run_individual.sbatch` for one Slurm target at a time:

```bash
REFCODE=PUKPOB CALCULATOR=MACE FF_STYLE=none sbatch run_individual.sbatch
REFCODE=PUKPOB CALCULATOR=MACE FF_STYLE=openff sbatch run_individual.sbatch
REFCODE=PUKPOB CALCULATOR=MACE FF_STYLE=gaff sbatch run_individual.sbatch
```

The Slurm wrapper uses `DB_FILE=$BASE_DIR/HEM.db` by default and passes it to `run_individual.py` as `--db-file`, so original structures are loaded from `HEM.db`. `CSV_FILE` is still passed as a fallback source for SMILES.

Useful Slurm environment variables:

- `REFCODE`: required target refcode.
- `CALCULATOR`: `MACE`, `MACEOFF`, `UMA`, `GAFF_MACE`, `GAFF_MACEOFF`, or `GAFF_UMA`.
- `FF_STYLE`: `none`, `openff`, `gaff`, or `auto`.
- `DB_FILE`: pyxtal database path; defaults to `$BASE_DIR/HEM.db`.
- `MODE`: `auto`, `original`, or `resume`.
- `TIMEOUT_SEC`: wall-clock timeout passed to `run_individual.py`.
- `MAX_RELAX_STEPS`: maximum ML relaxation steps.
- `LOG_TAG`: optional suffix for JSON summary filenames.

## Adding A New System

Create a folder and add the original CIF:

```bash
mkdir -p FAIL/NEWREF
cp NEWREF.cif FAIL/NEWREF/NEWREF.cif
```

Make sure `NEWREF` exists in `HEM.db` or in `entire_data.csv`, then run:

```bash
python run_individual.py NEWREF --calculator MACEOFF --mode original
```

## Notes

- Direct `run_individual.py` runs do not use CHARMM unless `--ff-style openff` or `--ff-style gaff` is passed.
- `OUTPUT_DIR` controls where new result files are written, but original CIF lookup still uses `FAIL/<REFCODE>/<REFCODE>.cif`.
- `resume` mode looks for calculator-specific timeout or failure snapshots in the output directory.
- `auto` mode is usually the safest choice for retrying failed systems.
