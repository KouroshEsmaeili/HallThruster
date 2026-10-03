#!/usr/bin/env python3
"""Compare full historical SPT-100 exports from Julia and Python."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from validation.compare_outputs import collect_pairs, metrics  # noqa: E402


SCALAR_ORDER = (
    "thrust_mN",
    "discharge_current_A",
    "ion_current_A",
    "max_electron_temperature_eV",
    "max_electric_field_V_per_m",
    "max_neutral_density_per_m3",
    "max_ion_density_per_m3",
    "mass_efficiency",
    "current_efficiency",
    "divergence_efficiency",
    "voltage_efficiency",
    "anode_efficiency",
)


def array_metrics(reference, candidate, relative_floor_fraction=1e-12):
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch {ref.shape} != {cand.shape}")
    error = np.abs(cand - ref)
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    floor = max(1e-30, relative_floor_fraction * scale)
    mask = np.abs(ref) > floor
    max_relative = float(np.max(error[mask] / np.abs(ref[mask]))) if np.any(mask) else 0.0
    denominator = float(np.linalg.norm(ref.ravel()))
    relative_l2 = (
        float(np.linalg.norm((cand - ref).ravel()) / denominator)
        if denominator
        else float(np.linalg.norm(cand.ravel()))
    )
    return {
        "max_abs": float(np.max(error)) if error.size else 0.0,
        "max_rel": max_relative,
        "relative_l2": relative_l2,
        "relative_floor": floor,
    }


def scalar_value(payload, name):
    return float(payload["run"]["metrics"][name]["value"])


def historical_pass(name, target, result, metric_payload):
    if name in {"thrust_mN", "discharge_current_A", "ion_current_A"}:
        tolerance = float(metric_payload[name]["standard_error"])
        return abs(result - target) <= tolerance, f"atol={tolerance:.6e}"
    return math.isclose(result, target, rel_tol=1e-2, abs_tol=0.0), "rtol=1e-2"


def compare_configurations(julia, python):
    pairs, errors = [], []
    collect_pairs(julia["configuration"], python["configuration"], "configuration", pairs, errors)
    tolerance_failures = [
        path for path, left, right in pairs
        if abs(right - left) > 1e-30 + 1e-12 * abs(left)
    ]
    return metrics(pairs), errors + [f"{path}: tolerance exceeded" for path in tolerance_failures]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("julia", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--translation-rtol", type=float, default=1e-8)
    parser.add_argument("--adaptive-l2-tolerance", type=float, default=1e-8)
    parser.add_argument("--profile-l2-tolerance", type=float, default=1e-8)
    parser.add_argument("--trajectory-l2-tolerance", type=float, default=1e-8)
    parser.add_argument("--skip-historical-baseline", action="store_true")
    args = parser.parse_args()

    julia = json.loads(args.julia.read_text())
    python = json.loads(args.python.read_text())
    failed = False
    summary = {
        "configuration": {}, "repeatability": {}, "scalars": {}, "adaptive": {},
        "profiles": {}, "trajectory": {},
        "tolerances": {
            "translation_rtol": args.translation_rtol,
            "adaptive_relative_l2": args.adaptive_l2_tolerance,
            "profile_relative_l2": args.profile_l2_tolerance,
            "trajectory_relative_l2": args.trajectory_l2_tolerance,
        },
    }

    repeat_runs = julia.get("repeatability", [])
    repeat_pairs = []
    repeat_step_match = True
    if len(repeat_runs) >= 2:
        first = repeat_runs[0]
        for repeated in repeat_runs[1:]:
            repeat_step_match &= first["adaptive"]["accepted_steps"] == repeated["adaptive"]["accepted_steps"]
            for name in SCALAR_ORDER:
                repeat_pairs.append((
                    f"repeatability.{name}",
                    float(first["metrics"][name]["value"]),
                    float(repeated["metrics"][name]["value"]),
                ))
    repeat_metrics = metrics(repeat_pairs)
    repeat_pass = len(repeat_runs) >= 2 and repeat_step_match and repeat_metrics[1] <= 1e-14
    failed |= not repeat_pass
    summary["repeatability"] = {
        "run_count": len(repeat_runs),
        "max_abs": repeat_metrics[0],
        "max_rel": repeat_metrics[1],
        "relative_l2": repeat_metrics[2],
        "accepted_steps_match": repeat_step_match,
        "status": "PASS" if repeat_pass else "FAIL",
    }
    print("Julia repeatability")
    print(f"  runs={len(repeat_runs)} max_abs={repeat_metrics[0]:.6e} "
          f"max_rel={repeat_metrics[1]:.6e} rel_l2={repeat_metrics[2]:.6e} "
          f"steps_match={repeat_step_match} status={'PASS' if repeat_pass else 'FAIL'}")

    config_metrics, config_errors = compare_configurations(julia, python)
    summary["configuration"] = {
        "max_abs": config_metrics[0], "max_rel": config_metrics[1],
        "relative_l2": config_metrics[2], "errors": config_errors,
    }
    failed |= bool(config_errors)
    print("\nconfiguration parity")
    print(f"  max_abs={config_metrics[0]:.6e} max_rel={config_metrics[1]:.6e} "
          f"rel_l2={config_metrics[2]:.6e} status={'FAIL' if config_errors else 'PASS'}")
    for error in config_errors:
        print(f"  {error}")

    print("\nhistorical scalar regression")
    print(
        f"{'metric':36} {'stored':>14} {'julia':>14} {'python':>14} "
        f"{'J/P rel':>12} {'hist':>6} {'parity':>7}"
    )
    for name in SCALAR_ORDER:
        stored = float(julia["stored_baseline"][name])
        julia_value = scalar_value(julia, name)
        python_value = scalar_value(python, name)
        relative = abs(python_value - julia_value) / abs(julia_value) if julia_value else abs(python_value)
        julia_baseline_pass, julia_tolerance = historical_pass(
            name, stored, julia_value, julia["run"]["metrics"]
        )
        python_baseline_pass, python_tolerance = historical_pass(
            name, stored, python_value, python["run"]["metrics"]
        )
        parity_pass = relative <= args.translation_rtol
        baseline_pass = (
            args.skip_historical_baseline or (julia_baseline_pass and python_baseline_pass)
        )
        status = "PASS" if baseline_pass and parity_pass else "FAIL"
        failed |= status == "FAIL"
        print(f"{name:36} {stored:14.6e} {julia_value:14.6e} {python_value:14.6e} "
              f"{relative:12.6e} {'PASS' if baseline_pass else 'FAIL':>6} "
              f"{'PASS' if parity_pass else 'FAIL':>7}")
        summary["scalars"][name] = {
            "stored": stored,
            "julia": julia_value,
            "python": python_value,
            "julia_python_relative_error": relative,
            "julia_historical_pass": julia_baseline_pass,
            "python_historical_pass": python_baseline_pass,
            "julia_historical_tolerance": julia_tolerance,
            "python_historical_tolerance": python_tolerance,
            "status": status,
        }

    julia_adaptive = julia["run"]["adaptive"]
    python_adaptive = python["run"]["adaptive"]
    dt_metrics = array_metrics(julia_adaptive["saved_dt"], python_adaptive["saved_dt"])
    time_metrics = array_metrics(julia_adaptive["save_times"], python_adaptive["save_times"])
    step_match = julia_adaptive["accepted_steps"] == python_adaptive["accepted_steps"]
    adaptive_pass = (
        dt_metrics["relative_l2"] <= args.adaptive_l2_tolerance
        and time_metrics["max_abs"] <= 1e-18
        and step_match
    )
    failed |= not adaptive_pass
    summary["adaptive"] = {
        "julia": {
            "initial_requested_dt": julia_adaptive["initial_requested_dt"],
            "minimum_saved_dt": julia_adaptive["minimum_saved_dt"],
            "maximum_saved_dt": julia_adaptive["maximum_saved_dt"],
            "accepted_steps": julia_adaptive["accepted_steps"],
        },
        "python": {
            "initial_requested_dt": python_adaptive["initial_requested_dt"],
            "minimum_saved_dt": python_adaptive["minimum_saved_dt"],
            "maximum_saved_dt": python_adaptive["maximum_saved_dt"],
            "accepted_steps": python_adaptive["accepted_steps"],
        },
        "saved_dt_metrics": dt_metrics,
        "save_time_metrics": time_metrics,
        "status": "PASS" if adaptive_pass else "FAIL",
    }
    print("\nadaptive timestep comparison")
    print(f"  Julia:  initial={julia_adaptive['initial_requested_dt']:.6e} "
          f"min(saved)={julia_adaptive['minimum_saved_dt']:.6e} "
          f"max(saved)={julia_adaptive['maximum_saved_dt']:.6e} "
          f"steps={julia_adaptive['accepted_steps']}")
    print(f"  Python: initial={python_adaptive['initial_requested_dt']:.6e} "
          f"min(saved)={python_adaptive['minimum_saved_dt']:.6e} "
          f"max(saved)={python_adaptive['maximum_saved_dt']:.6e} "
          f"steps={python_adaptive['accepted_steps']}")
    print(f"  dt rel_l2={dt_metrics['relative_l2']:.6e} "
          f"save_time max_abs={time_metrics['max_abs']:.6e} "
          f"status={'PASS' if adaptive_pass else 'FAIL'}")

    print("\ntime-averaged axial profiles")
    print(f"{'field':24} {'max_abs':>13} {'max_rel*':>13} {'rel_l2':>13} {'rel_floor':>13} status")
    for field, julia_profile in julia["run"]["averaged_profiles"].items():
        result = array_metrics(julia_profile, python["run"]["averaged_profiles"][field])
        status = "PASS" if result["relative_l2"] <= args.profile_l2_tolerance else "FAIL"
        failed |= status == "FAIL"
        summary["profiles"][field] = {**result, "status": status}
        print(f"{field:24} {result['max_abs']:13.6e} {result['max_rel']:13.6e} "
              f"{result['relative_l2']:13.6e} {result['relative_floor']:13.6e} {status}")

    julia_checkpoints = julia["run"]["checkpoints"]
    python_checkpoints = python["run"]["checkpoints"]
    print("\ncheckpoint trajectory")
    print(f"{'time':>13} {'max_abs':>13} {'max_rel*':>13} {'rel_l2':>13} status")
    for julia_checkpoint, python_checkpoint in zip(julia_checkpoints, python_checkpoints):
        pairs, errors = [], []
        collect_pairs(
            julia_checkpoint["fields"], python_checkpoint["fields"],
            f"checkpoint[{julia_checkpoint['time']}]", pairs, errors,
        )
        reference_values = [item[1] for item in pairs]
        candidate_values = [item[2] for item in pairs]
        overall = array_metrics(reference_values, candidate_values)
        field_results = {
            field: array_metrics(
                julia_checkpoint["fields"][field],
                python_checkpoint["fields"][field],
            )
            for field in julia_checkpoint["fields"]
        }
        time_equal = math.isclose(
            julia_checkpoint["time"], python_checkpoint["time"], rel_tol=0.0, abs_tol=1e-18
        )
        status = (
            "PASS" if not errors and time_equal
            and all(item["relative_l2"] <= args.trajectory_l2_tolerance for item in field_results.values())
            else "FAIL"
        )
        failed |= status == "FAIL"
        key = f"{julia_checkpoint['time']:.17g}"
        summary["trajectory"][key] = {
            **overall,
            "fields": field_results,
            "status": status,
            "errors": errors,
        }
        print(f"{julia_checkpoint['time']:13.6e} {overall['max_abs']:13.6e} "
              f"{overall['max_rel']:13.6e} {overall['relative_l2']:13.6e} {status}")

    julia_status = julia["run"]["status"]
    python_status = python["run"]["status"]
    status_pass = (
        julia_status["retcode"] == "success"
        and python_status["retcode"] == "success"
        and math.isclose(
            julia_status["reported_final_time"], python_status["reported_final_time"],
            rel_tol=0.0, abs_tol=1e-18,
        )
        and julia_status["saved_frames"] == python_status["saved_frames"]
    )
    failed |= not status_pass
    summary["status"] = {
        "julia": julia_status,
        "python": python_status,
        "status": "PASS" if status_pass else "FAIL",
    }
    print("\nrun status")
    print(f"  Julia:  retcode={julia_status['retcode']} final={julia_status['reported_final_time']} "
          f"frames={julia_status['saved_frames']} runtime={julia_status['runtime_seconds']:.6f}s")
    print(f"  Python: retcode={python_status['retcode']} final={python_status['reported_final_time']} "
          f"frames={python_status['saved_frames']} runtime={python_status['runtime_seconds']:.6f}s")
    print(f"  status={'PASS' if status_pass else 'FAIL'}")

    summary["overall_status"] = "FAIL" if failed else "PASS"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
