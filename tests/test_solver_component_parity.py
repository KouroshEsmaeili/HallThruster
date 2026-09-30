"""Focused source-parity checks for pre-timestep and single-step helpers."""

import numpy as np

from hallthruster import Config, EvenGrid, SPT_100, SimParams
from hallthruster.simulation.simulation import setup_simulation
from hallthruster.simulation.update_heavy_species import integrate_heavy_species
from hallthruster.utilities.interpolation import LinearInterpolation


def test_config_overload_for_heavy_species_integration_uses_config_scheme():
    config = Config(
        thruster=SPT_100,
        domain=(0.0, 0.08),
        discharge_voltage=300.0,
        anode_mass_flow_rate=5e-6,
        neutral_temperature_K=500.0,
    )
    sim = SimParams(
        grid=EvenGrid(4), dt=1e-8, duration=1e-8, num_save=2,
        adaptive=False, verbose=False, print_errors=False,
    )
    state, params = setup_simulation(config, sim)
    before = state.copy()

    integrate_heavy_species(state, params, config, dt=0.0)

    np.testing.assert_allclose(state, before, rtol=0.0, atol=0.0)


def test_setup_arrays_obey_historical_physical_index_relationships():
    config = Config(
        thruster=SPT_100,
        domain=(0.0, 0.08),
        discharge_voltage=300.0,
        anode_mass_flow_rate=5e-6,
        neutral_temperature_K=500.0,
        ncharge=3,
    )
    sim = SimParams(
        grid=EvenGrid(10), dt=1e-8, duration=1e-7, num_save=3,
        adaptive=False, verbose=False, print_errors=False,
    )
    state, params = setup_simulation(config, sim)
    cache = params["cache"]
    index = params["index"]
    inverse_mass = 1.0 / config.propellant.m

    assert state.shape == (7, 12)
    assert np.all(np.isfinite(state))
    np.testing.assert_allclose(cache["nn"], state[index["ρn"].start - 1] * inverse_mass)
    for charge in range(1, 4):
        density = state[index["ρi"][charge] - 1] * inverse_mass
        flux = state[index["ρiui"][charge] - 1] * inverse_mass
        np.testing.assert_allclose(cache["ni"][charge - 1], density)
        np.testing.assert_allclose(cache["niui"][charge - 1], flux)
        np.testing.assert_allclose(cache["ui"][charge - 1], flux / density)
    np.testing.assert_allclose(
        cache["ne"], sum(charge * cache["ni"][charge - 1] for charge in range(1, 4))
    )
    np.testing.assert_allclose(cache["nϵ"], 1.5 * cache["ne"] * cache["Tev"])

    field = LinearInterpolation(SPT_100.magnetic_field.z, SPT_100.magnetic_field.B)
    np.testing.assert_allclose(cache["B"], [field(z) for z in params["grid"].cell_centers])
    np.testing.assert_allclose(cache["channel_area"], SPT_100.geometry.channel_area)
    np.testing.assert_allclose(cache["inner_radius"], SPT_100.geometry.inner_radius)
    np.testing.assert_allclose(cache["outer_radius"], SPT_100.geometry.outer_radius)
