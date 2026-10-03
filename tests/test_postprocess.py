from types import SimpleNamespace

import numpy as np
import pytest

from hallthruster import Xenon
from hallthruster.physics.physicalconstants import e
from hallthruster.simulation.postprocess import (
    anode_eff,
    current_eff,
    discharge_current,
    divergence_eff,
    frame_dict,
    ion_current,
    mass_eff,
    thrust,
    thrust_all,
    time_average_from_frame,
    voltage_eff,
)
from hallthruster.simulation.solution import Solution


def synthetic_solution():
    config = SimpleNamespace(
        ncharge=2,
        propellant=Xenon,
        discharge_voltage=300.0,
        anode_mass_flow_rate=5e-6,
        apply_thrust_divergence_correction=False,
    )
    frame = {
        "channel_area": np.array([2.0, 3.0]),
        "ni": np.array([[2.0, 4.0], [1.0, 2.0]]),
        "ui": np.array([[2.0, 2.0], [3.0, 3.0]]),
        "niui": np.array([[4.0, 8.0], [3.0, 6.0]]),
        "Id": np.array([10.0]),
        "tanδ": np.array([0.0, 0.5]),
        "nn": np.array([7.0, 8.0]),
        "ne": np.array([4.0, 6.0]),
        "ue": np.array([1.0, 2.0]),
        "ϕ": np.array([300.0, 0.0]),
        "∇ϕ": np.array([-1.0, -2.0]),
        "Tev": np.array([3.0, 4.0]),
        "pe": np.array([12.0, 24.0]),
        "∇pe": np.array([1.0, 2.0]),
        "νen": np.array([1.0, 1.0]),
        "νei": np.array([2.0, 2.0]),
        "νan": np.array([3.0, 3.0]),
        "νc": np.array([4.0, 4.0]),
        "μ": np.array([5.0, 5.0]),
    }
    second = {key: np.copy(value) for key, value in frame.items()}
    second["Id"][0] = 20.0
    params = {
        "mi": Xenon.m,
        "grid": SimpleNamespace(cell_centers=np.array([0.0, 1.0])),
        "cache": {"B": np.array([0.01, 0.02])},
    }
    return Solution([0.0, 1.0], [frame, second], params, config, "success", "")


def test_historical_postprocessing_formulas_use_python_frame_indices():
    sol = synthetic_solution()
    expected_thrust = 68.0 * Xenon.m
    expected_ion_current = 60.0 * e

    assert thrust(sol, 0) == pytest.approx(expected_thrust)
    assert discharge_current(sol, 0) == 10.0
    assert discharge_current(sol, 1) == 20.0
    assert ion_current(sol, 0) == pytest.approx(expected_ion_current)
    assert mass_eff(sol, 0) == pytest.approx(42.0 * Xenon.m / 5e-6)
    assert voltage_eff(sol, 0) == pytest.approx(0.5 * Xenon.m * 2.0**2 / e / 300.0)
    assert divergence_eff(sol, 0) == pytest.approx(0.8)
    assert current_eff(sol, 0) == pytest.approx(expected_ion_current / 10.0)
    assert anode_eff(sol, 0) == pytest.approx(
        0.5 * expected_thrust**2 / 10.0 / 300.0 / 5e-6
    )


def test_frame_dict_uses_solution_config_and_grid_objects():
    output = frame_dict(synthetic_solution(), 1)

    assert output["t"] == 1.0
    assert output["discharge_current"] == 20.0
    np.testing.assert_array_equal(output["z"], [0.0, 1.0])
    assert len(output["ni"]) == 2


def test_postprocessing_all_frames_and_zero_based_time_average():
    sol = synthetic_solution()

    assert thrust_all(sol) == pytest.approx([68.0 * Xenon.m, 68.0 * Xenon.m])
    averaged = time_average_from_frame(sol, 0)
    assert averaged.t == [1.0]
    assert averaged.frames[0]["Id"][0] == 15.0
