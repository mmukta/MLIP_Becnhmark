#!/usr/bin/env python3
"""Plot paired relaxed-structure RMSD and residual-stress distributions."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stress-rmsd")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


METHODS = ("mace", "gaff_mace", "maceoff", "gaff_maceoff", "uma", "gaff_uma")
PAIR_PANELS = (
    ("MACE vs GAFF+MACE", ("mace", "gaff_mace")),
    ("MACEOFF vs GAFF+MACEOFF", ("maceoff", "gaff_maceoff")),
    ("UMA vs GAFF+UMA", ("uma", "gaff_uma")),
)
DISPLAY = {
    "mace": "MACE",
    "gaff_mace": "GAFF+MACE",
    "maceoff": "MACEOFF",
    "gaff_maceoff": "GAFF+MACEOFF",
    "uma": "UMA",
    "gaff_uma": "GAFF+UMA",
}
# Keep the exact method colors used by plot_rmsd_pdf.py. Each method retains
# the same color in its RMSD and residual-stress panels.
COLORS = {
    "mace": "darkgreen",
    "gaff_mace": "limegreen",
    "maceoff": "blue",
    "gaff_maceoff": "cyan",
    "uma": "brown",
    "gaff_uma": "lightcoral",
}


def zero_as_integer(value: float, _position: int) -> str:
    """Show the origin as 0 while leaving other tick labels compact."""
    return "0" if math.isclose(value, 0.0, abs_tol=1e-12) else f"{value:g}"


def stress_tick_label(value: float, _position: int) -> str:
    """Hide the x-axis origin label; the y-axis already displays zero."""
    return "" if math.isclose(value, 0.0, abs_tol=1e-12) else f"{value:.2f}"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=here / "energy_stress_rmsd.csv",
        help="RMSD CSV containing method, rmsd, status, and max_abs_stress_GPa.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "relaxed_rmsd_stress_panels.png",
    )
    parser.add_argument("--rmsd-xmax", type=float, default=0.20)
    parser.add_argument("--rmsd-bins", type=int, default=80)
    parser.add_argument(
        "--hist-style",
        choices=("filled", "step"),
        default="filled",
        help="Style of the RMSD histograms.",
    )
    parser.add_argument("--stress-xmax", type=float, default=0.20)
    parser.add_argument("--stress-bins", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.62)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def finite_float(value: str | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    long_required = {"method", "rmsd", "status", "max_abs_stress_GPa"}
    if long_required.issubset(fieldnames):
        return rows

    paired_required = {
        "mlp_method",
        "gaff_method",
        "mlp_rmsd",
        "gaff_rmsd",
        "mlp_max_abs_stress_GPa",
        "gaff_max_abs_stress_GPa",
        "mlp_status",
        "gaff_status",
    }
    if paired_required.issubset(fieldnames):
        expanded = []
        for row in rows:
            for prefix in ("mlp", "gaff"):
                expanded.append(
                    {
                        "refcode": row.get("refcode", ""),
                        "method": row.get(f"{prefix}_method", ""),
                        "rmsd": row.get(f"{prefix}_rmsd", ""),
                        "max_abs_stress_GPa": row.get(
                            f"{prefix}_max_abs_stress_GPa", ""
                        ),
                        "status": row.get(f"{prefix}_status", ""),
                    }
                )
        return expanded

    expected = sorted(long_required | paired_required)
    raise SystemExit(
        f"Unsupported columns in {path}. Expected the long format or paired "
        f"format containing: {', '.join(expected)}"
    )


def values_by_method(
    rows: list[dict[str, str]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    rmsd = {method: [] for method in METHODS}
    stress = {method: [] for method in METHODS}
    for row in rows:
        method = row.get("method", "")
        if method not in rmsd:
            continue
        rmsd_value = finite_float(row.get("rmsd"))
        stress_value = finite_float(row.get("max_abs_stress_GPa"))
        if rmsd_value is not None and rmsd_value >= 0.0:
            rmsd[method].append(rmsd_value)
        if stress_value is not None and stress_value >= 0.0:
            stress[method].append(stress_value)
    return rmsd, stress


def draw_rmsd_histogram(
    ax,
    methods: tuple[str, str],
    values: dict[str, list[float]],
    args: argparse.Namespace,
) -> None:
    edges = np.linspace(0.0, args.rmsd_xmax, args.rmsd_bins + 1)
    for method in methods:
        visible = [value for value in values[method] if value <= args.rmsd_xmax]
        if not visible:
            continue
        label = DISPLAY[method]
        if args.hist_style == "filled":
            ax.hist(
                visible,
                bins=edges,
                color=COLORS[method],
                alpha=args.alpha,
                label=label,
            )
        else:
            ax.hist(
                visible,
                bins=edges,
                histtype="step",
                color=COLORS[method],
                linewidth=2.2,
                label=label,
            )

    ax.set_xlim(0.0, args.rmsd_xmax)
    ax.set_xlabel("RMSD (Å)", fontsize=18)
    ax.set_ylabel("Frequency", fontsize=18)
    ax.legend(frameon=True, fontsize=14)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def draw_stress_distribution(
    ax,
    methods: tuple[str, str],
    values: dict[str, list[float]],
    args: argparse.Namespace,
) -> None:
    edges = np.linspace(0.0, args.stress_xmax, args.stress_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    for method in methods:
        data = np.asarray(values[method], dtype=float)
        if data.size == 0:
            continue
        visible = data[(data >= 0.0) & (data <= args.stress_xmax)]
        counts, _ = np.histogram(visible, bins=edges)
        # Normalize by the full method count. Values beyond stress-xmax remain
        # represented as missing area rather than renormalizing the visible tail.
        density = counts / (data.size * widths)
        ax.fill_between(
            centers,
            0.0,
            density,
            step="mid",
            color=COLORS[method],
            alpha=0.20,
            linewidth=0,
        )
        ax.step(
            centers,
            density,
            where="mid",
            color=COLORS[method],
            linewidth=2.0,
            label=f"{DISPLAY[method]}",
        )

    ax.set_xlim(0.0, args.stress_xmax)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Residual Stress (GPa)", fontsize=18)
    ax.set_ylabel("Probability Density", fontsize=18)
    ax.legend(frameon=True, fontsize=14)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.8)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv)
    rmsd, stress = values_by_method(rows)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.0, 9.0),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(hspace=0.12)

    for column, (title, methods) in enumerate(PAIR_PANELS):
        draw_rmsd_histogram(axes[0, column], methods, rmsd, args)
        draw_stress_distribution(axes[1, column], methods, stress, args)
        axes[0, column].yaxis.set_major_locator(MultipleLocator(1000))
        axes[0, column].xaxis.set_major_formatter(FuncFormatter(stress_tick_label))
        axes[1, column].yaxis.set_major_locator(MultipleLocator(20))
        axes[1, column].xaxis.set_major_locator(MultipleLocator(0.05))
        axes[1, column].xaxis.set_major_formatter(FuncFormatter(stress_tick_label))
        #axes[0, column].set_title(f"({chr(ord('a') + column)}) {title}: RMSD Distribution", fontsize=12)
        axes[0, column].set_title(f"({chr(ord('a') + column)}) RMSD Distribution", fontsize=16)
        axes[1, column].set_title(
            f"({chr(ord('d') + column)}) Residual Stress Distribution",
            fontsize=16,
        )

    # Avoid repeating y-axis titles while retaining numeric tick labels.
    for row in range(2):
        for column in (1, 2):
            axes[row, column].set_ylabel("")
            axes[row, column].tick_params(axis="y", labelleft=True)

    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.tick_params(axis="both", labelsize=16)
        ax.yaxis.set_major_formatter(FuncFormatter(zero_as_integer))
    fig.patch.set_facecolor("white")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    fig.savefig(f"{args.output}.pdf", dpi=args.dpi, facecolor="white")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
