#!/usr/bin/env python3
"""Run and export the historical SPT-100 regression through the direct Python API."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import pickle
import platform
import time
import traceback
from pathlib import Path

import numpy as np

from hallthruster import Config, EvenGrid, Geometry1D, SimParams, Xenon
from hallthruster.collisions.anomalous import (
    GaussianBohm,
    LogisticPressureShift,
    num_anom_variables,
)
from hallthruster.numerics.flux_functions import rusanov
from hallthruster.numerics.limiters import van_leer
from hallthruster.simulation.postprocess import (
    anode_eff_all,
    current_eff_all,
    discharge_current,
    discharge_current_all,
    divergence_eff_all,
    ion_current,
    ion_current_all,
    mass_eff_all,
    thrust_all,
    time_average_from_frame,
    voltage_eff_all,
)
from hallthruster.simulation.simulation import setup_simulation
from hallthruster.simulation.solution import (
    Solution,
    _saved_fields_matrix,
    _saved_fields_vector,
    saved_fields,
)
from hallthruster.simulation.update_heavy_species import (
    integrate_heavy_species_stage,
    update_heavy_species,
)
from hallthruster.thruster.magnetic_field import MagneticField
from hallthruster.thruster.thruster import Thruster
from hallthruster.walls.materials import BNSiO2
from hallthruster.walls.wall_sheath import WallSheath


REFERENCE_COMMIT = "014a12fb193af6927cb10f77da5e7baf215b5bc0"
CHECKPOINT_TARGETS = (0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3)
CHECKPOINT_FORMAT_VERSION = 1


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def build_case(baseline_file: Path, duration=None, num_save=None, num_cells=None):
    baseline = json.loads(baseline_file.read_text())
    data = baseline["config"]
    thruster_data = data["thruster"]
    geometry_data = thruster_data["geometry"]
    geometry = Geometry1D(
        channel_length=geometry_data["channel_length"],
        inner_radius=geometry_data["inner_radius"],
        outer_radius=geometry_data["outer_radius"],
    )
    thruster = Thruster(
        name=thruster_data["name"],
        geometry=geometry,
        magnetic_field=MagneticField(file=thruster_data["magnetic_field"]["file"]),
        shielded=False,
    )
    anom_data = data["anom_model"]
    gaussian_data = anom_data["model"]
    gaussian = GaussianBohm(
        gaussian_data["hall_min"],
        gaussian_data["hall_max"],
        gaussian_data["center"],
        gaussian_data["width"],
    )
    anomalous = LogisticPressureShift(
        gaussian,
        z0=anom_data["z0"],
        dz=anom_data["dz"],
        pstar=anom_data["pstar"],
        alpha=anom_data["alpha"],
    )
    wall_data = data["wall_loss_model"]
    wall = WallSheath(BNSiO2, wall_data["loss_scale"])

    config = Config(
        thruster=thruster,
        domain=tuple(data["domain"]),
        discharge_voltage=data["discharge_voltage"],
        anode_mass_flow_rate=data["anode_mass_flow_rate"],
        ncharge=data["ncharge"],
        cathode_coupling_voltage=data["cathode_coupling_voltage"],
        anode_Tev=data["anode_Tev"],
        cathode_Tev=data["cathode_Tev"],
        anom_model=anomalous,
        wall_loss_model=wall,
        electron_ion_collisions=data["electron_ion_collisions"],
        neutral_velocity=data["neutral_velocity"],
        neutral_temperature_K=data["neutral_temperature_K"],
        ion_temperature_K=data["ion_temperature_K"],
        ion_wall_losses=data["ion_wall_losses"],
        background_pressure_Torr=data["background_pressure_Torr"],
        background_temperature_K=data["background_temperature_K"],
        neutral_ingestion_multiplier=data["neutral_ingestion_multiplier"],
        solve_plume=data["solve_plume"],
        apply_thrust_divergence_correction=data["apply_thrust_divergence_correction"],
        transition_length=data["transition_length"],
    )
    sim_data = baseline["simulation"]
    sim = SimParams(
        grid=EvenGrid(num_cells or sim_data["grid"]["num_cells"]),
        dt=sim_data["dt"],
        duration=duration if duration is not None else sim_data["duration"],
        num_save=num_save if num_save is not None else sim_data["num_save"],
        adaptive=sim_data["adaptive"],
    )
    return baseline, config, sim


def normalized_config(config, sim):
    geometry = config.thruster.geometry
    anom = config.anom_model
    gaussian = anom.model
    wall = config.wall_loss_model
    scheme = config.scheme
    return {
        "ncharge": config.ncharge,
        "propellant": repr(config.propellant),
        "thruster": {
            "name": config.thruster.name,
            "shielded": config.thruster.shielded,
            "geometry": {
                "channel_length": geometry.channel_length,
                "inner_radius": geometry.inner_radius,
                "outer_radius": geometry.outer_radius,
                "channel_area": geometry.channel_area,
            },
            "magnetic_field_file": config.thruster.magnetic_field.file,
        },
        "domain": config.domain,
        "anode_mass_flow_rate": config.anode_mass_flow_rate,
        "discharge_voltage": config.discharge_voltage,
        "cathode_coupling_voltage": config.cathode_coupling_voltage,
        "neutral_velocity": config.neutral_velocity,
        "neutral_temperature_K": config.neutral_temperature_K,
        "ion_temperature_K": config.ion_temperature_K,
        "anode_Tev": config.anode_Tev,
        "cathode_Tev": config.cathode_Tev,
        "background_pressure_Torr": config.background_pressure_Torr,
        "background_temperature_K": config.background_temperature_K,
        "transition_length": config.transition_length,
        "neutral_ingestion_multiplier": config.neutral_ingestion_multiplier,
        "solve_plume": config.solve_plume,
        "apply_thrust_divergence_correction": config.apply_thrust_divergence_correction,
        "ion_wall_losses": config.ion_wall_losses,
        "electron_ion_collisions": config.electron_ion_collisions,
        "electron_plume_loss_scale": config.electron_plume_loss_scale,
        "magnetic_field_scale": config.magnetic_field_scale,
        "anom_smoothing_iters": config.anom_smoothing_iters,
        "anom_model": {
            "type": "LogisticPressureShift",
            "z0": anom.z0,
            "dz": anom.dz,
            "pstar": anom.pstar,
            "alpha": anom.alpha,
            "model": {
                "type": "GaussianBohm",
                "hall_min": gaussian.hall_min,
                "hall_max": gaussian.hall_max,
                "center": gaussian.center,
                "width": gaussian.width,
            },
        },
        "wall_loss_model": {
            "type": "WallSheath",
            "material": wall.material.name,
            "loss_scale": wall.loss_scale,
        },
        "scheme": {
            "flux_function": "rusanov" if scheme.flux_function is rusanov else repr(scheme.flux_function),
            "limiter": "van_leer" if scheme.limiter is van_leer else repr(scheme.limiter),
            "reconstruct": scheme.reconstruct,
        },
        "ionization_model": str(config.ionization_model),
        "excitation_model": str(config.excitation_model),
        "electron_neutral_model": str(config.electron_neutral_model),
        "simulation": {
            "grid_type": "EvenGrid",
            "num_cells": sim.grid.num_cells,
            "dt": sim.dt,
            "adaptive": sim.adaptive,
            "CFL": sim.CFL,
            "min_dt": sim.min_dt,
            "max_dt": sim.max_dt,
            "max_small_steps": sim.max_small_steps,
            "duration": sim.duration,
            "num_save": sim.num_save,
        },
    }


def stored_baseline():
    return {
        "thrust_mN": 87.304,
        "discharge_current_A": 4.614,
        "ion_current_A": 3.922,
        "max_electron_temperature_eV": 24.832,
        "max_electric_field_V_per_m": 6.665e4,
        "max_neutral_density_per_m3": 2.088e19,
        "max_ion_density_per_m3": 9.651e17,
        "mass_efficiency": 0.954,
        "current_efficiency": 0.873,
        "divergence_efficiency": 0.949,
        "voltage_efficiency": 0.661,
        "anode_efficiency": 0.5656,
    }


def initialize_resumable_state(config, sim, include_dirs):
    """Set up the normal solver state plus explicit loop-local restart data."""
    U, params = setup_simulation(config, sim, include_dirs=include_dirs)
    params["iteration"][0] = 1
    saveat = np.linspace(0.0, sim.duration, num=sim.num_save)
    first_frame = {
        field: params["cache"][field]
        for field in saved_fields()
        if field in params["cache"]
    }
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "configuration": normalized_config(config, sim),
        "U": U,
        "params": params,
        "t": 0.0,
        "saveat": saveat,
        "frames": [copy.deepcopy(first_frame) for _ in saveat],
        "save_ind": 1,
        "small_step_count": 0,
        "uniform_steps": False,
        "retcode": "success",
        "error": "",
        "runtime_seconds": 0.0,
    }


def write_checkpoint(path, state):
    """Atomically write generated state without risking the prior checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def read_checkpoint(path):
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {state.get('format_version')!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}."
        )
    return state


def advance_resumable_state(
    state,
    config,
    *,
    checkpoint_path=None,
    checkpoint_steps=5000,
    progress_seconds=30.0,
    max_accepted_steps=None,
):
    """Continue the translated solver loop from an explicit validation checkpoint.

    The operations and their order mirror ``simulation.solution.solve``. The
    extra work only serializes state after complete accepted steps and reports
    progress; it does not alter any numerical input or update.
    """
    U = state["U"]
    params = state["params"]
    iteration = params["iteration"]
    cache = params["cache"]
    sim = params["simulation"]
    t = state["t"]
    saveat = state["saveat"]
    frames = state["frames"]
    save_ind = state["save_ind"]
    small_step_count = state["small_step_count"]
    uniform_steps = state["uniform_steps"]
    retcode = state["retcode"]
    errstring = state["error"]
    num_save = len(saveat)
    sources = {
        "source_neutrals": config.source_neutrals,
        "source_ion_continuity": config.source_ion_continuity,
        "source_ion_momentum": config.source_ion_momentum,
    }
    scheme = config.scheme
    update_module = importlib.import_module("hallthruster.simulation.update_electrons")
    plume_module = importlib.import_module("hallthruster.simulation.plume")
    invocation_start = time.perf_counter()
    last_progress = invocation_start
    prior_runtime = state["runtime_seconds"]

    def sync_state():
        state.update({
            "t": t,
            "save_ind": save_ind,
            "small_step_count": small_step_count,
            "uniform_steps": uniform_steps,
            "retcode": retcode,
            "error": errstring,
            "runtime_seconds": prior_runtime + time.perf_counter() - invocation_start,
        })

    try:
        while t < saveat[-1]:
            if sim.adaptive:
                if uniform_steps:
                    params["dt"][0] = sim.dt
                    small_step_count -= 1
                else:
                    params["dt"][0] = max(
                        sim.min_dt, min(cache["dt"][0], sim.max_dt)
                    )

            t += params["dt"][0]

            if params["dt"][0] == sim.min_dt:
                small_step_count += 1
            elif not uniform_steps:
                small_step_count = 0

            if small_step_count >= sim.max_small_steps:
                uniform_steps = True
            elif small_step_count == 0:
                uniform_steps = False

            integrate_heavy_species_stage(
                U, params, scheme, sources, params["dt"][0]
            )
            update_heavy_species(U, params)
            if not np.all(np.isfinite(U)):
                if sim.print_errors:
                    print(
                        f"Warning: NaN or Inf detected in heavy species solver at time {t}"
                    )
                retcode = "failure"
                break

            update_module.update_electrons(params, config, t)
            if config.solve_plume:
                plume_module.update_plume_geometry(params)

            iteration[0] += 1
            if iteration[0] % 100 == 0:
                time.sleep(0)

            if save_ind < num_save and t > saveat[save_ind]:
                for field in _saved_fields_vector():
                    if field == "anom_variables":
                        for index in range(num_anom_variables(config.anom_model)):
                            frames[save_ind][field][index] = cache[field][index]
                    else:
                        frames[save_ind][field] = np.copy(cache[field])
                for field in _saved_fields_matrix():
                    frames[save_ind][field] = np.copy(cache[field])
                save_ind += 1

            accepted_steps = iteration[0] - 1
            now = time.perf_counter()
            if accepted_steps % 1000 == 0 and now - last_progress >= progress_seconds:
                print(
                    f"progress time={t:.9e}/{saveat[-1]:.9e} "
                    f"steps={accepted_steps} dt={params['dt'][0]:.6e} "
                    f"saved_frames={save_ind} "
                    f"elapsed_seconds={prior_runtime + now - invocation_start:.1f}",
                    flush=True,
                )
                last_progress = now

            should_checkpoint = (
                checkpoint_path is not None
                and checkpoint_steps > 0
                and accepted_steps % checkpoint_steps == 0
            )
            should_pause = (
                max_accepted_steps is not None
                and accepted_steps >= max_accepted_steps
            )
            if should_checkpoint or should_pause:
                sync_state()
                if checkpoint_path is not None:
                    write_checkpoint(checkpoint_path, state)
            if should_pause:
                return None

    except Exception:
        errstring = traceback.format_exc()
        retcode = "error"
        if sim.print_errors:
            print("Warning: Error detected in solution:")
            print(errstring)

    sync_state()
    if checkpoint_path is not None:
        write_checkpoint(checkpoint_path, state)
    return Solution(
        saveat[: min(save_ind, num_save)],
        frames[: min(save_ind, num_save)],
        params,
        config,
        retcode,
        errstring,
    )


def frame_data(frame):
    return {
        "nn": frame["nn"],
        "ni": frame["ni"],
        "niui": frame["niui"],
        "ui": frame["ui"],
        "ne": frame["ne"],
        "Tev": frame["Tev"],
        "potential": frame["ϕ"],
        "electric_field": -frame["∇ϕ"],
        "nu_ionization": frame["νiz"],
        "nu_collisions": frame["νc"],
    }


def scalar_metrics(sol):
    nsave = len(sol.frames)
    avg_start_one_based = nsave // 3
    avg_start_zero_based = avg_start_one_based - 1
    n_avg = nsave - avg_start_one_based
    tail = slice(avg_start_zero_based, nsave)
    thrust_values = np.asarray(thrust_all(sol)) * 1000.0
    discharge_values = np.asarray(discharge_current_all(sol))
    ion_values = np.asarray(ion_current_all(sol))
    efficiency_values = {
        "mass_efficiency": np.asarray(mass_eff_all(sol)),
        "current_efficiency": np.asarray(current_eff_all(sol)),
        "divergence_efficiency": np.asarray(divergence_eff_all(sol)),
        "voltage_efficiency": np.asarray(voltage_eff_all(sol)),
        "anode_efficiency": np.asarray(anode_eff_all(sol)),
    }
    averaged = time_average_from_frame(sol, avg_start_zero_based).frames[0]

    def summary(values, selected=None):
        selected_values = values if selected is None else values[selected]
        return {
            "value": float(np.mean(selected_values)),
            "standard_error": float(np.std(selected_values, ddof=1) / math.sqrt(n_avg)),
        }

    output = {
        "avg_start_frame_one_based": avg_start_one_based,
        "averaged_frame_count": nsave - avg_start_zero_based,
        "historical_standard_error_denominator_count": n_avg,
        "thrust_mN": summary(thrust_values, tail),
        "discharge_current_A": summary(discharge_values, tail),
        "ion_current_A": summary(ion_values, tail),
        "max_electron_temperature_eV": {"value": float(np.max(averaged["Tev"]))},
        "max_electric_field_V_per_m": {"value": float(np.max(-averaged["∇ϕ"]))},
        "max_neutral_density_per_m3": {"value": float(np.max(averaged["nn"]))},
        "max_ion_density_per_m3": {"value": float(np.max(averaged["ni"]))},
    }
    output.update({name: summary(values) for name, values in efficiency_values.items()})
    return output


def collect_run(sol, runtime_seconds):
    nsave = len(sol.frames)
    avg_start_zero_based = nsave // 3 - 1
    averaged = time_average_from_frame(sol, avg_start_zero_based).frames[0]
    checkpoints = []
    times = np.asarray(sol.t)
    for target in CHECKPOINT_TARGETS:
        index = int(np.argmin(np.abs(times - target)))
        checkpoints.append({
            "target_time": target,
            "frame_index_zero_based": index,
            "time": sol.t[index],
            "fields": frame_data(sol.frames[index]),
            "discharge_current_A": discharge_current(sol, index),
            "ion_current_A": ion_current(sol, index),
        })
    saved_dt = [float(frame["dt"][0]) for frame in sol.frames]
    return {
        "status": {
            "retcode": sol.retcode,
            "error": sol.error,
            "reported_final_time": sol.t[-1],
            "saved_frames": nsave,
            "accepted_steps": sol.params["iteration"][0] - 1,
            "runtime_seconds": runtime_seconds,
            "warnings": [],
        },
        "adaptive": {
            "initial_requested_dt": sol.params["simulation"].dt,
            "initial_internal_dt": 100 * np.finfo(float).eps,
            "minimum_saved_dt": min(saved_dt),
            "maximum_saved_dt": max(saved_dt),
            "saved_dt": saved_dt,
            "accepted_steps": sol.params["iteration"][0] - 1,
            "save_times": sol.t,
        },
        "grid": {
            "z": sol.params["grid"].cell_centers,
            "magnetic_field": sol.params["cache"]["B"],
        },
        "metrics": scalar_metrics(sol),
        "histories": {
            "time": sol.t,
            "dt": saved_dt,
            "discharge_current_A": discharge_current_all(sol),
            "ion_current_A": ion_current_all(sol),
            "thrust_mN": np.asarray(thrust_all(sol)) * 1000.0,
        },
        "checkpoints": checkpoints,
        "averaged_profiles": {
            "nn": averaged["nn"],
            "ni": averaged["ni"],
            "niui": averaged["niui"],
            "ui": averaged["ui"],
            "ne": averaged["ne"],
            "Tev": averaged["Tev"],
            "potential": averaged["ϕ"],
            "electric_field": -averaged["∇ϕ"],
            "magnetic_field": sol.params["cache"]["B"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("test/regression/baseline.json"))
    parser.add_argument("--duration", type=float)
    parser.add_argument("--num-save", type=int)
    parser.add_argument("--num-cells", type=int)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-steps", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-accepted-steps", type=int)
    args = parser.parse_args()

    _, config, sim = build_case(
        args.baseline.resolve(),
        duration=args.duration,
        num_save=args.num_save,
        num_cells=args.num_cells,
    )
    if args.resume:
        if args.checkpoint is None:
            parser.error("--resume requires --checkpoint")
        state = read_checkpoint(args.checkpoint)
        requested_configuration = normalized_config(config, sim)
        if state["configuration"] != requested_configuration:
            raise ValueError(
                "Checkpoint configuration does not match the requested regression case."
            )
    else:
        state = initialize_resumable_state(
            config, sim, [str(args.baseline.resolve().parent)]
        )
    solution = advance_resumable_state(
        state,
        config,
        checkpoint_path=args.checkpoint,
        checkpoint_steps=args.checkpoint_steps,
        progress_seconds=args.progress_seconds,
        max_accepted_steps=args.max_accepted_steps,
    )
    if solution is None:
        print(
            f"checkpointed steps={state['params']['iteration'][0] - 1} "
            f"time={state['t']:.9e} path={args.checkpoint}",
            flush=True,
        )
        return
    runtime_seconds = state["runtime_seconds"]
    run = collect_run(solution, runtime_seconds)
    payload = {
        "metadata": {
            "language": "python",
            "reference_commit": REFERENCE_COMMIT,
            "python_version": platform.python_version(),
            "baseline_file": str(args.baseline.resolve()),
        },
        "configuration": normalized_config(config, sim),
        "stored_baseline": stored_baseline(),
        "run": run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n")
    status = run["status"]
    print(
        f"retcode={status['retcode']} final_time={status['reported_final_time']} "
        f"steps={status['accepted_steps']} frames={status['saved_frames']} "
        f"runtime_seconds={status['runtime_seconds']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
