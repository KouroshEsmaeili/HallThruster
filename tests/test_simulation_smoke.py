import numpy as np

from hallthruster import Config, EvenGrid, SPT_100, SimParams, run_simulation


def test_tiny_direct_simulation_reaches_requested_time_with_finite_state():
    config = Config(
        thruster=SPT_100,
        domain=(0.0, 0.08),
        discharge_voltage=300.0,
        anode_mass_flow_rate=5e-6,
        neutral_temperature_K=500.0,
    )
    sim = SimParams(
        grid=EvenGrid(20),
        dt=1e-8,
        duration=1e-7,
        num_save=3,
        adaptive=False,
        verbose=False,
        print_errors=False,
    )

    solution = run_simulation(config, sim)

    assert solution.retcode == "success", solution.error
    assert solution.t[-1] == sim.duration
    assert len(solution.frames) == sim.num_save
    for frame in solution.frames:
        for value in frame.values():
            if isinstance(value, np.ndarray):
                assert np.all(np.isfinite(value))


def test_historical_keyword_runner_delegates_to_simparams_api():
    config = Config(
        thruster=SPT_100,
        domain=(0.0, 0.08),
        discharge_voltage=300.0,
        anode_mass_flow_rate=5e-6,
        neutral_temperature_K=500.0,
    )

    solution = run_simulation(
        config,
        ncells=10,
        dt=1e-8,
        duration=2e-8,
        nsave=2,
        adaptive=False,
        verbose=False,
        print_errors=False,
    )

    assert solution.retcode == "success", solution.error
    compatibility_sim = solution.params["simulation"]
    assert compatibility_sim.grid.num_cells == 10
    assert compatibility_sim.dt == 1e-8
    assert compatibility_sim.duration == 2e-8
    assert compatibility_sim.num_save == 2
    assert compatibility_sim.adaptive is False
