#!/usr/bin/env python3
"""Compare JSON files emitted by the Julia and Python validation exporters."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


NONFINITE = {"NaN", "Inf", "-Inf"}


def collect_pairs(reference, candidate, path, pairs, errors):
    if isinstance(reference, dict) and isinstance(candidate, dict):
        reference_keys = set(reference) - {"meta"}
        candidate_keys = set(candidate) - {"meta"}
        if reference_keys != candidate_keys:
            errors.append(
                f"{path}: key mismatch reference-only={sorted(reference_keys - candidate_keys)} "
                f"candidate-only={sorted(candidate_keys - reference_keys)}"
            )
        for key in sorted(reference_keys & candidate_keys):
            collect_pairs(reference[key], candidate[key], f"{path}.{key}", pairs, errors)
        return

    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            errors.append(f"{path}: length mismatch {len(reference)} != {len(candidate)}")
            return
        for index, (left, right) in enumerate(zip(reference, candidate)):
            collect_pairs(left, right, f"{path}[{index}]", pairs, errors)
        return

    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        pairs.append((path, float(reference), float(candidate)))
        return

    if reference in NONFINITE or candidate in NONFINITE:
        if reference != candidate:
            errors.append(f"{path}: non-finite mismatch {reference!r} != {candidate!r}")
        return

    if reference != candidate:
        errors.append(f"{path}: value mismatch {reference!r} != {candidate!r}")


def metrics(pairs):
    if not pairs:
        return 0.0, 0.0, 0.0
    reference = np.array([item[1] for item in pairs], dtype=float)
    candidate = np.array([item[2] for item in pairs], dtype=float)
    error = np.abs(candidate - reference)
    nonzero = np.abs(reference) > np.finfo(float).tiny
    relative = np.zeros_like(error)
    relative[nonzero] = error[nonzero] / np.abs(reference[nonzero])
    denominator = np.linalg.norm(reference)
    relative_l2 = np.linalg.norm(candidate - reference) / denominator if denominator else np.linalg.norm(candidate)
    return float(np.max(error)), float(np.max(relative)), float(relative_l2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="Julia JSON output")
    parser.add_argument("candidate", type=Path, help="Python JSON output")
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-30)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    components = sorted((set(reference) & set(candidate)) - {"meta"})
    failed = False
    print(f"{'component':28} {'max_abs':>13} {'max_rel':>13} {'rel_l2':>13} status")
    for component in components:
        pairs = []
        errors = []
        collect_pairs(reference[component], candidate[component], component, pairs, errors)
        max_abs, max_rel, relative_l2 = metrics(pairs)
        tolerance_failures = [
            path for path, left, right in pairs
            if abs(right - left) > args.atol + args.rtol * abs(left)
        ]
        status = "PASS" if not errors and not tolerance_failures else "FAIL"
        failed |= status == "FAIL"
        print(f"{component:28} {max_abs:13.6e} {max_rel:13.6e} {relative_l2:13.6e} {status}")
        for error in errors[:10]:
            print(f"  {error}")
        for path in tolerance_failures[:10]:
            print(f"  {path}: tolerance exceeded")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
