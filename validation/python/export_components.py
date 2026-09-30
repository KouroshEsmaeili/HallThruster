#!/usr/bin/env python3
"""Export deterministic Python validation cases as JSON.

The matching Julia exporter lives in ``validation/julia/export_components.jl``.
Both scripts intentionally use the public/direct APIs instead of JSON input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from hallthruster import Config, EvenGrid, SPT_100, SimParams, UnevenGrid, Xenon, generate_grid
from hallthruster.collisions.elastic import ElasticCollision
from hallthruster.collisions.excitation import ExcitationReaction
from hallthruster.collisions.ionization import IonizationReaction
from hallthruster.collisions.reactions import load_rate_coeffs, rate_coeff
from hallthruster.numerics.finite_differences import (
    backward_diff_coeffs,
    backward_difference,
    central_diff_coeffs,
    central_difference,
    downwind_diff_coeffs,
    forward_diff_coeffs,
    forward_difference,
    second_deriv_central_diff,
    second_deriv_coeffs,
    upwind_diff_coeffs,
)
from hallthruster.numerics.flux_functions import HLLE, flux, global_lax_friedrichs, rusanov
from hallthruster.numerics.limiters import slope_limiters
from hallthruster.physics.fluid import EulerEquations
from hallthruster.physics.physicalconstants import NA, R0, e, kB, me
from hallthruster.simulation.allocation import allocate_arrays_from_grid
from hallthruster.simulation.configuration import configure_fluids, configure_index, params_from_config
from hallthruster.simulation.initialization import DefaultInitialization, initialize
from hallthruster.simulation.simulation import run_simulation, setup_simulation
from hallthruster.thruster.geometry import channel_perimeter, channel_width
from hallthruster.utilities.integration import cumtrapz
from hallthruster.utilities.interpolation import LinearInterpolation
from hallthruster.utilities.linearalgebra import Tridiagonal, tridiagonal_solve


REFERENCE_COMMIT = "014a12fb193af6927cb10f77da5e7baf215b5bc0"


def json_value(value):
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def config_for_validation(**overrides):
    values = {
        "thruster": SPT_100,
        "domain": (0.0, 0.08),
        "discharge_voltage": 300.0,
        "anode_mass_flow_rate": 5e-6,
        "neutral_temperature_K": 500.0,
    }
    values.update(overrides)
    return Config(**values)


def grid_data(spec):
    grid = generate_grid(spec, SPT_100.geometry, (0.0, 0.08))
    return {
        "num_cells": grid.num_cells,
        "edges": grid.edges,
        "cell_centers": grid.cell_centers,
        "dz_edge": grid.dz_edge,
        "dz_cell": grid.dz_cell,
    }


def reaction_data():
    energies = [0.0, 1.0, 5.5, 10.0, 20.0, 50.0, 100.0, 255.0]
    output = {}
    cases = [
        ("elastic", None, ElasticCollision),
        ("excitation", None, ExcitationReaction),
        ("ionization", Xenon(1), IonizationReaction),
    ]
    for reaction_type, product, reaction_class in cases:
        threshold, coefficients = load_rate_coeffs(Xenon(0), product, reaction_type)
        if reaction_type == "elastic":
            reaction = reaction_class(Xenon(0), coefficients)
        elif reaction_type == "excitation":
            reaction = reaction_class(threshold, Xenon(0), coefficients)
        else:
            reaction = reaction_class(threshold, Xenon(0), product, coefficients)
        output[reaction_type] = {
            "threshold": threshold,
            "energies": energies,
            "coefficients": [rate_coeff(reaction, energy) for energy in energies],
        }
    return output


def config_data():
    config = config_for_validation(ncharge=3)
    return {
        "thruster": config.thruster.name,
        "propellant": repr(config.propellant),
        "ncharge": config.ncharge,
        "domain": config.domain,
        "discharge_voltage": config.discharge_voltage,
        "anode_mass_flow_rate": config.anode_mass_flow_rate,
        "cathode_coupling_voltage": config.cathode_coupling_voltage,
        "anode_Tev": config.anode_Tev,
        "cathode_Tev": config.cathode_Tev,
        "neutral_velocity": config.neutral_velocity,
        "neutral_temperature_K": config.neutral_temperature_K,
        "ion_temperature_K": config.ion_temperature_K,
        "background_pressure_Torr": config.background_pressure_Torr,
        "background_temperature_K": config.background_temperature_K,
        "transition_length": config.transition_length,
        "solve_plume": config.solve_plume,
        "ion_wall_losses": config.ion_wall_losses,
        "electron_ion_collisions": config.electron_ion_collisions,
        "anom_model": type(config.anom_model).__name__,
        "wall_loss_model": type(config.wall_loss_model).__name__,
        "conductivity_model": type(config.conductivity_model).__name__,
        "ionization_model": config.ionization_model,
        "excitation_model": config.excitation_model,
        "electron_neutral_model": config.electron_neutral_model,
        "source_neutrals_zero": config.source_neutrals(None, None, 0),
        "source_energy_zero": config.source_energy(None, 0),
        "source_ion_continuity_zero": [source(None, None, 0) for source in config.source_ion_continuity],
        "source_ion_momentum_zero": [source(None, None, 0) for source in config.source_ion_momentum],
    }


def index_data():
    output = {}
    for ncharge in (1, 2, 3):
        fluids, fluid_ranges, species, species_ranges, is_velocity = configure_fluids(
            config_for_validation(ncharge=ncharge)
        )
        index = configure_index(fluids, fluid_ranges)
        output[str(ncharge)] = {
            "species": [repr(item) for item in species],
            "fluid_ranges_one_based_inclusive": [
                [item.start, item.stop - 1] for item in fluid_ranges
            ],
            "species_ranges_one_based_inclusive": {
                key: [value.start, value.stop - 1] for key, value in species_ranges.items()
            },
            "neutral": index["ρn"].start,
            "ion_density": index["ρi"],
            "ion_momentum": index["ρiui"],
            "velocity_mask": is_velocity,
        }
    return output


def initialization_data():
    config = config_for_validation(
        discharge_voltage=500.0,
        anode_mass_flow_rate=3e-6,
        ncharge=3,
        cathode_Tev=5.0,
        anode_Tev=3.0,
        neutral_velocity=300.0,
        neutral_temperature_K=100.0,
        ion_temperature_K=300.0,
        initial_condition=DefaultInitialization(max_electron_temperature=10.0),
    )
    grid = generate_grid(EvenGrid(4), config.thruster.geometry, config.domain)
    fluids, fluid_ranges, _, _, _ = configure_fluids(config)
    index = configure_index(fluids, fluid_ranges)
    state, cache = allocate_arrays_from_grid(grid, config)
    params = {
        **params_from_config(config),
        "grid": grid,
        "index": index,
        "cache": cache,
        "min_Te": min(config.anode_Tev, config.cathode_Tev),
    }
    initialize(state, params, config)
    return {
        "cell_centers": grid.cell_centers,
        "state": state,
        "electron_density": cache["ne"],
        "electron_energy_density": cache["nϵ"],
        "electron_temperature_ev": cache["Tev"],
    }


def setup_data():
    config = config_for_validation()
    sim = SimParams(
        grid=EvenGrid(20), dt=1e-8, duration=1e-7, num_save=3,
        adaptive=False, verbose=False, print_errors=True,
    )
    state, params = setup_simulation(config, sim)
    cache = params["cache"]
    return {
        "state": state,
        "cell_centers": params["grid"].cell_centers,
        "edges": params["grid"].edges,
        "magnetic_field": cache["B"],
        "neutral_density": cache["nn"],
        "ion_density": cache["ni"],
        "ion_flux": cache["niui"],
        "electron_density": cache["ne"],
        "electron_temperature_ev": cache["Tev"],
        "electron_energy_density": cache["nϵ"],
        "potential": cache["ϕ"],
        "potential_gradient": cache["∇ϕ"],
        "channel_area": cache["channel_area"],
        "inner_radius": cache["inner_radius"],
        "outer_radius": cache["outer_radius"],
    }


def short_simulation_data():
    config = config_for_validation()
    sim = SimParams(
        grid=EvenGrid(20), dt=1e-8, duration=1e-7, num_save=3,
        adaptive=False, verbose=False, print_errors=True,
    )
    solution = run_simulation(config, sim)
    selected = ["nn", "ni", "niui", "ui", "ne", "Tev", "∇ϕ", "Id", "νiz", "νc"]
    return {
        "retcode": solution.retcode,
        "times": solution.t,
        "frames": [
            {field: frame[field] for field in selected}
            for frame in solution.frames
        ],
    }


def collect():
    geometry = SPT_100.geometry
    magnetic = LinearInterpolation(SPT_100.magnetic_field.z, SPT_100.magnetic_field.B)
    fd_points = (0.0, 0.5, 2.0)
    fd_values = tuple(point**2 for point in fd_points)
    matrix = Tridiagonal([1.0, -1.0, 2.0], [4.0, 5.0, 6.0, 7.0], [2.0, 3.0, -2.0])
    expected_solution = np.array([1.0, -2.0, 3.0, 0.5])
    rhs = matrix.matvec(expected_solution)
    ratios = [-2.0, 0.0, 0.25, 1.0, 10.0, math.inf, math.nan]
    fluid = EulerEquations(Xenon(1))
    left = (1.0, 300.0, Xenon.cv * 300.0 + 0.5 * 300.0**2)
    right = (0.5, 50.0, 0.5 * (Xenon.cv * 600.0 + 0.5 * 100.0**2))

    return json_value({
        "meta": {"language": "python", "julia_reference_commit": REFERENCE_COMMIT},
        "physical_constants": {"e": e, "me": me, "kB": kB, "NA": NA, "R0": R0},
        "gas_species": {
            "name": repr(Xenon), "M": Xenon.M, "m": Xenon.m, "R": Xenon.R,
            "cp": Xenon.cp, "cv": Xenon.cv, "gamma": Xenon.gamma,
            "species": [repr(Xenon(charge)) for charge in (0, 1, 2, 3)],
        },
        "geometry": {
            "channel_length": geometry.channel_length,
            "inner_radius": geometry.inner_radius,
            "outer_radius": geometry.outer_radius,
            "channel_area": geometry.channel_area,
            "channel_width": channel_width(geometry.outer_radius, geometry.inner_radius),
            "channel_perimeter": channel_perimeter(geometry.outer_radius, geometry.inner_radius),
        },
        "even_grid": {str(count): grid_data(EvenGrid(count)) for count in (4, 10, 20)},
        "uneven_grid": {str(count): grid_data(UnevenGrid(count)) for count in (4, 10, 20)},
        "interpolation": {
            "x": [-1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 9.0],
            "y": [LinearInterpolation([0.0, 2.0, 5.0], [0.0, 4.0, 10.0])(x)
                  for x in (-1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 9.0)],
        },
        "finite_differences": {
            "forward_coefficients": forward_diff_coeffs(*fd_points),
            "central_coefficients": central_diff_coeffs(*fd_points),
            "backward_coefficients": backward_diff_coeffs(*fd_points),
            "second_coefficients": second_deriv_coeffs(*fd_points),
            "upwind_coefficients": upwind_diff_coeffs(*fd_points),
            "downwind_coefficients": downwind_diff_coeffs(*fd_points),
            "forward_quadratic": forward_difference(*fd_values, *fd_points),
            "central_quadratic": central_difference(*fd_values, *fd_points),
            "backward_quadratic": backward_difference(*fd_values, *fd_points),
            "second_quadratic": second_deriv_central_diff(*fd_values, *fd_points),
        },
        "integration": cumtrapz([0.0, 0.5, 2.0, 3.0], [1.0, 2.0, -1.0, 4.0], 3.0),
        "linear_algebra": {"rhs": rhs, "solution": tridiagonal_solve(matrix, rhs)},
        "limiters": {
            name: {"ratios": ratios, "values": [limiter(ratio) for ratio in ratios]}
            for name, limiter in slope_limiters.items()
        },
        "flux_functions": {
            "left_state": left,
            "right_state": right,
            "physical_left": flux(left, fluid),
            "physical_right": flux(right, fluid),
            "rusanov": rusanov(left, right, fluid),
            "global_lax_friedrichs": global_lax_friedrichs(
                left, right, fluid, max_wave_speed=500.0
            ),
            "hlle": HLLE(left, right, fluid),
        },
        "reaction_tables": reaction_data(),
        "magnetic_field": {
            "positions": [0.0, 0.0125, 0.025, 0.05, 0.08, 0.1],
            "values": [magnetic(z) for z in (0.0, 0.0125, 0.025, 0.05, 0.08, 0.1)],
        },
        "config_defaults": config_data(),
        "fluid_index_mapping": index_data(),
        "initialization": initialization_data(),
        "setup": setup_data(),
        "short_simulation": short_simulation_data(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="JSON output path")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
