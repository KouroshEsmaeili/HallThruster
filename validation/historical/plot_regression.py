#!/usr/bin/env python3
"""Generate ignored scientific comparison plots from historical exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REFERENCE_LABEL = "Julia 1.10.11 / HallThruster.jl@014a12f"
PYTHON_LABEL = "Python translation"


def plot_profile(output, z_julia, z_python, julia, python, ylabel, title):
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(z_julia, julia, label=REFERENCE_LABEL)
    axis.plot(z_python, python, "--", label=PYTHON_LABEL)
    axis.set_xlabel("Axial position (m)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("julia", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    julia = json.loads(args.julia.read_text())["run"]
    python = json.loads(args.python.read_text())["run"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zj = np.asarray(julia["grid"]["z"])
    zp = np.asarray(python["grid"]["z"])
    jp = julia["averaged_profiles"]
    pp = python["averaged_profiles"]

    scalar_profiles = (
        ("nn", "Neutral density (m$^{-3}$)", "Time-averaged neutral density", "neutral_density.png"),
        ("ne", "Electron density (m$^{-3}$)", "Time-averaged electron density", "electron_density.png"),
        ("Tev", "Electron temperature (eV)", "Time-averaged electron temperature", "electron_temperature.png"),
        ("potential", "Electric potential (V)", "Time-averaged electric potential", "electric_potential.png"),
        ("electric_field", "Electric field (V/m)", "Time-averaged electric field", "electric_field.png"),
        ("magnetic_field", "Magnetic field (T)", "SPT-100 magnetic field", "magnetic_field.png"),
    )
    for field, ylabel, title, filename in scalar_profiles:
        plot_profile(args.output_dir / filename, zj, zp, jp[field], pp[field], ylabel, title)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for charge, (julia_charge, python_charge) in enumerate(zip(jp["ni"], pp["ni"]), start=1):
        line = axis.plot(zj, julia_charge, label=f"Julia Xe{charge}+")[0]
        axis.plot(zp, python_charge, "--", color=line.get_color(), label=f"Python Xe{charge}+")
    axis.set_xlabel("Axial position (m)")
    axis.set_ylabel("Ion density (m$^{-3}$)")
    axis.set_title("Time-averaged ion densities — HallThruster.jl@014a12f")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "ion_density.png", dpi=160)
    plt.close(fig)

    for field, ylabel, title, filename in (
        ("discharge_current_A", "Discharge current (A)", "Discharge current history", "discharge_current.png"),
        ("dt", "Adaptive timestep (s)", "Saved adaptive timestep history", "adaptive_dt.png"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(julia["histories"]["time"], julia["histories"][field], label=REFERENCE_LABEL)
        axis.plot(python["histories"]["time"], python["histories"][field], "--", label=PYTHON_LABEL)
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{title} — HallThruster.jl@014a12f")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
