#!/usr/bin/env python3
"""Draw energy-parity and delta-energy panels for MLIP/GAFF+MLIP pairs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sqlite3
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rmsd-pdf")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


METHODS = ["mace", "maceoff", "uma", "gaff_mace", "gaff_maceoff", "gaff_uma"]
PAIR_PANELS = [
    ("MACE vs GAFF+MACE", ["mace", "gaff_mace"]),
    ("MACEOFF vs GAFF+MACEOFF", ["maceoff", "gaff_maceoff"]),
    ("UMA vs GAFF+UMA", ["uma", "gaff_uma"]),
]
COLORS = {
    "mace": "darkgreen",
    "maceoff": "blue",
    "uma": "brown",
    "gaff_mace": "limegreen",
    "gaff_maceoff": "cyan",
    "gaff_uma": "lightcoral",
}
DELTA_Y_LIMITS = {
    "mace": 200.0,
    "uma": 200.0,
}
EV_TO_KJ_PER_MOL = 96.4853321233


class TwoDecimalScalarFormatter(ScalarFormatter):
    """Scientific-notation formatter with two decimal places."""

    def __init__(self, *args, hide_first: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hide_first = hide_first

    def _set_format(self) -> None:
        self.format = "%1.2f"

    def __call__(self, x, pos=None) -> str:
        if self.hide_first and pos == 0:
            return ""
        return super().__call__(x, pos)


def display_method(method: str) -> str:
    return {
        "mace": "MACE",
        "maceoff": "MACEOFF",
        "uma": "UMA",
        "gaff_mace": "GAFF+MACE",
        "gaff_maceoff": "GAFF+MACEOFF",
        "gaff_uma": "GAFF+UMA",
    }.get(method, method)


def energy_symbol(method: str) -> str:
    return rf"E_{{\mathrm{{{display_method(method).replace('+', '{+}')}}}}}"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot energy parity and delta-energy panels."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "common_fmax_energy_panels.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--energy-db",
        type=Path,
        default=here / "HEM_relaxed.db",
        help="ASE SQLite database supplying paired energies for the energy plots.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=here / "energy_stress_rmsd.csv",
        help=(
            "Common-threshold paired-energy CSV. When present, status=ok rows "
            "from this file replace the database final energies."
        ),
    )
    parser.add_argument(
        "--energy-delta-threshold",
        type=float,
        default=1.0,
        help="Similarity threshold marked in the energy-difference panels (kJ/mol).",
    )
    parser.add_argument(
        "--energy-delta-ymax",
        type=float,
        default=0.0,
        help="Optional symmetric y-axis limit for delta-energy scatter panels.",
    )
    return parser.parse_args()


def read_database_energies(path: Path) -> dict[str, dict[str, float]]:
    """Create kJ/mol-per-molecule plot energies without changing the ASE database."""
    if not path.is_file():
        raise SystemExit(f"Missing energy database: {path}")

    energies = {method: {} for method in METHODS}
    placeholders = ",".join("?" for _ in METHODS)
    reference_query = """
        SELECT
            json_extract(key_value_pairs, '$.refcode') AS refcode,
            natoms,
            json_extract(key_value_pairs, '$.Z') AS molecule_count,
            json_extract(key_value_pairs, '$.mol_formula') AS formula
        FROM systems
        WHERE json_extract(key_value_pairs, '$.method') = 'original'
    """
    energy_query = f"""
        SELECT
            json_extract(key_value_pairs, '$.refcode') AS refcode,
            json_extract(key_value_pairs, '$.method') AS method,
            energy,
            natoms,
            id
        FROM systems
        WHERE energy IS NOT NULL
          AND json_extract(key_value_pairs, '$.method') IN ({placeholders})
    """
    with sqlite3.connect(path) as connection:
        formula_sizes = {}
        for refcode, natoms, molecule_count, formula in connection.execute(reference_query):
            if not refcode or not natoms or not molecule_count or not formula:
                continue
            elements = re.findall(r"([A-Z][a-z]?)(\d*)", str(formula))
            total_atoms = sum(int(count or 1) for _, count in elements)
            hydrogen_atoms = sum(int(count or 1) for element, count in elements if element == "H")
            heavy_atoms = total_atoms - hydrogen_atoms
            expected_total = float(natoms) / float(molecule_count)
            if total_atoms > 0 and math.isclose(
                total_atoms, expected_total, rel_tol=0.0, abs_tol=1e-6
            ):
                formula_sizes[str(refcode)] = (total_atoms, heavy_atoms)

        hydrogen_counts = {
            int(row_id): int(count)
            for row_id, count in connection.execute("SELECT id, n FROM species WHERE Z = 1")
        }
        skipped = 0
        for refcode, method, energy, natoms, row_id in connection.execute(
            energy_query, METHODS
        ):
            refcode = str(refcode) if refcode else ""
            sizes = formula_sizes.get(refcode)
            if method not in energies or energy is None or not natoms or not sizes:
                skipped += 1
                continue
            total_atoms, heavy_atoms = sizes
            hydrogen_count = hydrogen_counts.get(int(row_id), 0)
            atoms_per_molecule = total_atoms if hydrogen_count else heavy_atoms
            if atoms_per_molecule <= 0:
                skipped += 1
                continue
            candidate = float(natoms) / atoms_per_molecule
            molecule_count = round(candidate)
            if molecule_count <= 0 or not math.isclose(
                candidate, molecule_count, rel_tol=0.0, abs_tol=1e-6
            ):
                skipped += 1
                continue
            value = (float(energy) / molecule_count) * EV_TO_KJ_PER_MOL
            if math.isfinite(value):
                energies[method][refcode] = value
            else:
                skipped += 1
    if skipped:
        print(f"Skipped {skipped:,} database energies without a valid molecule count")
    return energies


def read_common_fmax_energies(path: Path) -> dict[str, dict[str, float]]:
    """Read comparable kJ/mol energies extracted at a shared FIRE threshold."""
    energies = {method: {} for method in METHODS}
    required = {
        "refcode",
        "mlp_method",
        "gaff_method",
        "status",
        "mlp_energy_kJ_mol",
        "gaff_energy_kJ_mol",
    }
    accepted = 0
    excluded = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Missing columns in {path}: {', '.join(sorted(missing))}"
            )
        for row in reader:
            if str(row.get("status") or "").strip().lower() != "ok":
                excluded += 1
                continue
            refcode = str(row.get("refcode") or "").strip()
            mlp_method = str(row.get("mlp_method") or "").strip()
            gaff_method = str(row.get("gaff_method") or "").strip()
            if (
                not refcode
                or mlp_method not in energies
                or gaff_method not in energies
            ):
                excluded += 1
                continue
            try:
                mlp_energy = float(row["mlp_energy_kJ_mol"])
                gaff_energy = float(row["gaff_energy_kJ_mol"])
            except (TypeError, ValueError):
                excluded += 1
                continue
            if not (math.isfinite(mlp_energy) and math.isfinite(gaff_energy)):
                excluded += 1
                continue
            energies[mlp_method][refcode] = mlp_energy
            energies[gaff_method][refcode] = gaff_energy
            accepted += 1
    print(
        f"Loaded {accepted:,} common-threshold energy pairs from {path}; "
        f"ignored {excluded:,} excluded/invalid rows"
    )
    return energies


def paired_energies(
    energies: dict[str, dict[str, float]], mlp_method: str, gaff_method: str
) -> list[tuple[str, float, float]]:
    """Return paired (refcode, E_MLP, E_GAFF+MLP), sorted by refcode."""
    mlp_energies = energies.get(mlp_method, {})
    gaff_energies = energies.get(gaff_method, {})
    common_refcodes = sorted(mlp_energies.keys() & gaff_energies.keys())
    return [
        (refcode, mlp_energies[refcode], gaff_energies[refcode])
        for refcode in common_refcodes
    ]


def draw_energy_scatter(
    ax,
    energies: dict[str, dict[str, float]],
    mlp_method: str,
    gaff_method: str,
    axis_limits: tuple[float, float] | None = None,
) -> None:
    paired = paired_energies(energies, mlp_method, gaff_method)
    if not paired:
        ax.text(0.5, 0.5, "No paired energies", ha="center", va="center")
        ax.set_axis_off()
        return

    mlp_energies = np.asarray([mlp_energy for _, mlp_energy, _ in paired])
    gaff_energies = np.asarray([gaff_energy for _, _, gaff_energy in paired])
    finite = np.isfinite(mlp_energies) & np.isfinite(gaff_energies)
    mlp_energies = mlp_energies[finite]
    gaff_energies = gaff_energies[finite]
    if mlp_energies.size == 0:
        ax.text(0.5, 0.5, "No finite paired energies", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(
        mlp_energies,
        gaff_energies,
        color=COLORS[mlp_method],
        s=14.0,
        alpha=0.75,
        linewidths=0,
    )

    if axis_limits is None:
        shared_min = min(float(mlp_energies.min()), float(gaff_energies.min()))
        shared_max = max(float(mlp_energies.max()), float(gaff_energies.max()))
        padding = 0.04 * (shared_max - shared_min) if shared_max > shared_min else 1.0
        axis_min = shared_min - padding
        axis_max = shared_max + padding
    else:
        axis_min, axis_max = axis_limits
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    # The x and y axes use the same numerical range, while the physical panel
    # aspect is controlled in main() so both plot rows have identical sizes.
    ax.set_aspect("auto")
    shared_ticks = np.linspace(axis_min, axis_max, 5)
    ax.set_xticks(shared_ticks)
    ax.set_yticks(shared_ticks)
    ax.set_xlabel(f"{display_method(mlp_method)}", fontsize=18)
    ax.set_ylabel(f"{display_method(gaff_method)}", fontsize=18)
    # The first x and y ticks represent the same shared axis minimum. Label it
    # only on the y-axis to avoid showing the value twice at the lower-left.
    x_formatter = TwoDecimalScalarFormatter(useMathText=True, hide_first=True)
    y_formatter = TwoDecimalScalarFormatter(useMathText=True)
    for formatter in (x_formatter, y_formatter):
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.xaxis.get_offset_text().set_fontsize(16.0)
    ax.yaxis.get_offset_text().set_fontsize(16.0)
    ax.tick_params(labelsize=16)
    ax.grid(color="#E6E6E6", linewidth=0.7)
    ax.set_facecolor("white")


def draw_delta_energy_rank_scatter(
    ax,
    energies: dict[str, dict[str, float]],
    mlp_method: str,
    gaff_method: str,
    args: argparse.Namespace,
) -> None:
    paired = paired_energies(energies, mlp_method, gaff_method)
    rows = [
        (refcode, gaff_energy - mlp_energy)
        for refcode, mlp_energy, gaff_energy in paired
        if math.isfinite(mlp_energy) and math.isfinite(gaff_energy)
    ]
    rows.sort(key=lambda item: item[1])
    if not rows:
        ax.text(0.5, 0.5, "No finite ΔE values", ha="center", va="center")
        ax.set_axis_off()
        return

    refcodes = [refcode for refcode, _delta in rows]
    delta = np.asarray([value for _refcode, value in rows], dtype=float)
    x = np.arange(1, delta.size + 1)
    threshold = abs(float(args.energy_delta_threshold))
    color = COLORS.get(mlp_method, "tab:blue")

    ax.scatter(x, delta, s=5.0, alpha=0.55, color=color, linewidths=0)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.axhspan(-threshold, threshold, color="#ECEFF1", zorder=0)
    ax.axhline(threshold, color="black", linestyle=":", linewidth=0.8)
    ax.axhline(-threshold, color="black", linestyle=":", linewidth=0.8)

    ax.scatter([1, delta.size], [delta[0], delta[-1]], s=15, color="red", zorder=3)
    ax.annotate(
        f"{refcodes[0]}\n{delta[0]:+.1f}",
        xy=(1, delta[0]),
        xytext=(11, 5),
        textcoords="offset points",
        va="bottom",
        fontsize=12.0,
        color="red",
    )
    ax.annotate(
        f"{refcodes[-1]}\n{delta[-1]:+.1f}",
        xy=(delta.size, delta[-1]),
        xytext=(-11, -7),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=12.0,
        color="red",
    )

    if args.energy_delta_ymax and args.energy_delta_ymax > 0:
        y_limit = float(args.energy_delta_ymax)
    else:
        y_limit = DELTA_Y_LIMITS.get(
            mlp_method, max(1.0, math.ceil(float(np.max(np.abs(delta))) * 1.03))
        )
    # Small vertical headroom keeps endpoint labels away from the frame.
    ax.set_ylim(-1.08 * y_limit, 1.08 * y_limit)

    ax.set_xlabel("Structure Count", fontsize=18)
    ax.set_ylabel(r"$\Delta E$ (kJ/mol)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14.0)
    ax.grid(color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    args = parse_args()
    if args.csv.is_file():
        energies = read_common_fmax_energies(args.csv)
    else:
        print(
            f"CSV not found: {args.csv};"
            "falling back to database final energies"
        )
        energies = read_database_energies(args.energy_db)

    # A wider canvas gives the ranked-energy panels enough room for outlier
    # labels and keeps all six plotting boxes the same size.
    # Use one energy range for all parity panels so their positions and slopes
    # can be compared directly.
    all_paired_values: list[float] = []
    for _title, methods in PAIR_PANELS:
        for _refcode, mlp_energy, gaff_energy in paired_energies(
            energies, methods[0], methods[1]
        ):
            if math.isfinite(mlp_energy) and math.isfinite(gaff_energy):
                all_paired_values.extend((mlp_energy, gaff_energy))
    if all_paired_values:
        global_min = min(all_paired_values)
        global_max = max(all_paired_values)
        global_padding = (
            0.04 * (global_max - global_min) if global_max > global_min else 1.0
        )
        shared_energy_limits = (
            global_min - global_padding,
            global_max + global_padding,
        )
    else:
        shared_energy_limits = None

    fig = plt.figure(figsize=(18.5, 9.8), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
    )
    scatter_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    delta_axes = [fig.add_subplot(grid[1, column]) for column in range(3)]
    fig.set_constrained_layout_pads(h_pad=0.10, hspace=0.12)

    for column, (title, methods) in enumerate(PAIR_PANELS):
        scatter_ax = scatter_axes[column]
        delta_ax = delta_axes[column]
        draw_energy_scatter(
            scatter_ax,
            energies,
            methods[0],
            methods[1],
            axis_limits=shared_energy_limits,
        )
        scatter_ax.set_box_aspect(0.72)
        scatter_ax.set_title(
            f"({chr(ord('a') + column)}) Energy Distribution(kJ/mol)",
            fontsize=18,
        )
        draw_delta_energy_rank_scatter(delta_ax, energies, methods[0], methods[1], args)
        # Keep the lower panels broad and leave clearance around the two
        # annotated endpoint structures.
        delta_ax.set_box_aspect(0.72)
        delta_ax.set_xlim(-350, 14750)
        delta_ax.set_xticks([4000, 8000, 12000])
        if column in (0, 2):
            delta_ax.set_ylim(-225, 225)
            delta_ax.set_yticks([-200, -100, 0, 100, 200])
        delta_ax.set_title(
            (
                rf"({chr(ord('d') + column)}) "
                rf"$\Delta E = {energy_symbol(methods[1])} - "
                rf"{energy_symbol(methods[0])}$"
            ),
            fontsize=18,
        )

    for _, methods in PAIR_PANELS:
        count = len(paired_energies(energies, methods[0], methods[1]))
        print(f"Paired energies for {display_method(methods[0])}/{display_method(methods[1])}: {count:,}")

    fig.patch.set_facecolor("white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
