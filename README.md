# MLIP_Becnhmark

This folder is a small benchmark workspace for testing `MACE` and `MACEOFF` relaxations on selected systems.

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

## Folder Layout

Each system should live inside:

```text
FAIL/<REFCODE>/
```

Typical contents:

```text
FAIL/FANTIX/FANTIX.cif
FAIL/FANTIX/FANTIX_relaxed_maceoff_fail.cif
FAIL/FANTIX/fire_maceoff.log
```

Files used by the workflow:

- Original experimental CIF:
  `FAIL/<REFCODE>/<REFCODE>.cif`
- Resume snapshot for `MACE`:
  `FAIL/<REFCODE>/<REFCODE>_relaxed_mace_timeout.cif`
  or
  `FAIL/<REFCODE>/<REFCODE>_relaxed_mace_fail.cif`
- Resume snapshot for `MACEOFF`:
  `FAIL/<REFCODE>/<REFCODE>_relaxed_maceoff_timeout.cif`
  or
  `FAIL/<REFCODE>/<REFCODE>_relaxed_maceoff_fail.cif`

The refcode must also exist in [entire_data.csv] so the script can recover the SMILES.

## Simple Local Test

Run from this folder:

```bash
cd MLIP_Becnhmark
conda activate htocsp
```

Run one system from the original CIF:

```bash
python run_individual.py FANTIX --calculator MACEOFF --mode original
```

Resume one system from saved snapshots:

```bash
python run_individual.py FANTIX --calculator MACEOFF --mode resume
```

Auto mode prefers timeout snapshot, then fail snapshot, then original CIF:

```bash
python run_individual.py FANTIX --calculator MACEOFF --mode auto
```

Run with `MACE` instead:

```bash
python run_individual.py FANTIX --calculator MACE --mode auto
```

Useful optional arguments:

```bash
python run_individual.py FANTIX \
  --calculator MACEOFF \
  --mode auto \
  --timeout-sec 14400 \
  --max-relax-steps 5000 \
  --log-tag test
```

Outputs are written into:

- `FAIL/<REFCODE>/` for CIF/log outputs
- `single_runs/` for JSON and JSONL summaries

## Submit Multiple Systems By Name

Use the helper script to submit one Slurm job per system:

```bash
cd MLIP_Becnhmark
conda activate htocsp
```

Submit specific systems:

```bash
bash submit_fail_targets.sh FANTIX ACAMIA XUBZEZ
```

Submit all systems currently present inside `FAIL/`:

```bash
bash submit_fail_targets.sh
```

Choose calculator:

```bash
CALCULATOR=MACE bash submit_fail_targets.sh FANTIX
CALCULATOR=MACEOFF bash submit_fail_targets.sh FANTIX
```

Choose start mode:

```bash
MODE=original bash submit_fail_targets.sh FANTIX
MODE=resume bash submit_fail_targets.sh FANTIX
MODE=auto bash submit_fail_targets.sh FANTIX
```

Override resources if needed:

```bash
CPUS_PER_TASK=16 MEMORY=128G WALLTIME=36:00:00 \
CALCULATOR=MACEOFF MODE=auto \
bash submit_fail_targets.sh FANTIX ACAMIA
```

Override the base directory if running from somewhere else:

```bash
BASE_DIR=/scratch/mmukta/MLIP_Becnhmark \
bash /scratch/mmukta/MLIP_Becnhmark/submit_fail_targets.sh FANTIX
```

## Slurm Files

- [run_individual.py]
  runs one target locally
- [run_individual.sbatch]
  Slurm wrapper for one target
- [submit_fail_targets.sh]
  submits multiple targets by system name

## Adding A New System

1. Create a folder:

```bash
mkdir -p FAIL/NEWREF
```

2. Put the original CIF here:

```bash
FAIL/NEWREF/NEWREF.cif
```

3. Make sure `NEWREF` exists in [entire_data.csv].

4. Run locally or submit:

```bash
python run_individual.py NEWREF --calculator MACEOFF --mode original
```

or

```bash
bash submit_fail_targets.sh NEWREF
```

## Notes

- `resume` mode errors if no calculator-specific snapshot exists.
- `auto` mode is usually the safest choice for retrying failed systems.
- The scripts do not depend on marker files to choose the input CIF for `run_individual.py`.
