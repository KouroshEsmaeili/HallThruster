#!/usr/bin/env python3
"""Generate figures from committed M3 summary metrics in VALIDATION.md.

These are NOT regenerated raw-simulation plots. The raw Julia and Python M3
outputs are intentionally unavailable; every value below is copied from the
authoritative committed validation report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent

SCALAR_RELATIVE_ERRORS = {
    "Thrust": 4.702240e-7,
    "Discharge current": 6.948353e-7,
    "Ion current": 6.573864e-7,
    "Max electron temp.": 1.161662e-6,
    "Max electric field": 7.200894e-7,
    "Anode efficiency": 1.686929e-7,
}

PROFILE_RELATIVE_L2 = {
    "Electric field": 5.138842e-6,
    "Ion flux": 1.602786e-6,
    "Ion velocity": 6.659293e-7,
    "Electron temp.": 2.304299e-6,
    "Electron density": 1.092194e-6,
    "Ion density": 1.090252e-6,
    "Magnetic field": 1.418270e-16,
    "Neutral density": 4.775431e-8,
    "Electric potential": 5.186442e-7,
}

TRAJECTORY_TIMES_S = [
    0.0,
    1.001001e-6,
    1.001001e-5,
    1.001001e-4,
    5.005005e-4,
    1.0e-3,
]
TRAJECTORY_RELATIVE_L2 = [
    1.274726e-16,
    4.554096e-14,
    2.283763e-14,
    1.495378e-13,
    3.105864e-12,
    4.411136e-4,
]


def save_bar_plot(
    values: dict[str, float], title: str, ylabel: str, output: Path
) -> None:
    """Save one log-scale bar chart using Matplotlib defaults."""
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(values.keys(), values.values())
    axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=35)
    axis.grid(axis="y")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def generate_plots() -> None:
    """Generate the three committed summary figures beside this script."""
    save_bar_plot(
        SCALAR_RELATIVE_ERRORS,
        "Historical SPT-100 Julia/Python scalar differences",
        "Julia/Python relative difference",
        OUTPUT_DIR / "scalar_relative_errors.png",
    )
    save_bar_plot(
        PROFILE_RELATIVE_L2,
        "Historical SPT-100 time-averaged profile differences",
        "Relative-L2 error",
        OUTPUT_DIR / "profile_relative_l2.png",
    )

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(TRAJECTORY_TIMES_S, TRAJECTORY_RELATIVE_L2, marker="o")
    axis.set_yscale("log")
    axis.set_xlabel("Physical simulation time (s)")
    axis.set_ylabel("Total trajectory relative-L2")
    axis.set_title("Historical SPT-100 trajectory difference checkpoints")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "trajectory_relative_l2.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    generate_plots()
