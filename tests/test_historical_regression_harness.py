import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hallthruster import run_simulation


EXPORTER = Path(__file__).parents[1] / "validation/historical/python/export_regression.py"
SPEC = importlib.util.spec_from_file_location("historical_regression_exporter", EXPORTER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_historical_baseline_direct_api_configuration_matches_source_file():
    baseline = Path(__file__).parents[1] / "test/regression/baseline.json"

    _, config, sim = MODULE.build_case(baseline)
    normalized = MODULE.normalized_config(config, sim)

    assert normalized["ncharge"] == 3
    assert normalized["domain"] == pytest.approx((0.0, 0.08))
    assert normalized["thruster"]["geometry"]["inner_radius"] == 0.035
    assert normalized["thruster"]["geometry"]["outer_radius"] == 0.05
    assert normalized["thruster"]["geometry"]["channel_length"] == 0.025
    assert normalized["anom_model"]["type"] == "LogisticPressureShift"
    assert normalized["anom_model"]["model"]["type"] == "GaussianBohm"
    assert normalized["wall_loss_model"]["loss_scale"] == 0.86904
    assert normalized["neutral_ingestion_multiplier"] == 6.23341
    assert normalized["solve_plume"] is True
    assert normalized["ion_wall_losses"] is True
    assert normalized["electron_ion_collisions"] is True
    assert normalized["simulation"] == {
        "grid_type": "EvenGrid",
        "num_cells": 200,
        "dt": 5e-9,
        "adaptive": True,
        "CFL": 0.799,
        "min_dt": 1e-10,
        "max_dt": 1e-7,
        "max_small_steps": 100,
        "duration": 1e-3,
        "num_save": 1000,
    }


def test_validation_checkpoint_resume_is_bitwise_identical_to_direct_solver(tmp_path):
    baseline = Path(__file__).parents[1] / "test/regression/baseline.json"
    _, direct_config, direct_sim = MODULE.build_case(
        baseline, duration=1e-7, num_save=3, num_cells=20
    )
    direct = run_simulation(
        direct_config, direct_sim, include_dirs=[str(baseline.parent)]
    )

    _, resumed_config, resumed_sim = MODULE.build_case(
        baseline, duration=1e-7, num_save=3, num_cells=20
    )
    checkpoint = tmp_path / "historical.pkl"
    state = MODULE.initialize_resumable_state(
        resumed_config, resumed_sim, [str(baseline.parent)]
    )
    paused = MODULE.advance_resumable_state(
        state,
        resumed_config,
        checkpoint_path=checkpoint,
        checkpoint_steps=1,
        progress_seconds=float("inf"),
        max_accepted_steps=2,
    )
    assert paused is None

    loaded = MODULE.read_checkpoint(checkpoint)
    resumed = MODULE.advance_resumable_state(
        loaded,
        resumed_config,
        checkpoint_path=checkpoint,
        checkpoint_steps=1,
        progress_seconds=float("inf"),
    )

    assert resumed.retcode == direct.retcode == "success"
    assert resumed.t == direct.t
    assert resumed.params["iteration"] == direct.params["iteration"]
    for direct_frame, resumed_frame in zip(direct.frames, resumed.frames):
        assert direct_frame.keys() == resumed_frame.keys()
        for field in direct_frame:
            if field == "anom_variables":
                for direct_value, resumed_value in zip(
                    direct_frame[field], resumed_frame[field]
                ):
                    np.testing.assert_array_equal(resumed_value, direct_value)
            else:
                np.testing.assert_array_equal(resumed_frame[field], direct_frame[field])
