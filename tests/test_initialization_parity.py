"""Source-derived initialization parity checks for Julia commit 014a12f."""

import numpy as np

from hallthruster import Config, EvenGrid, SPT_100, Xenon, generate_grid
from hallthruster.simulation.allocation import allocate_arrays_from_grid
from hallthruster.simulation.configuration import configure_fluids, configure_index, params_from_config
from hallthruster.simulation.initialization import DefaultInitialization, initialize


def test_default_initialization_matches_historical_reference_arrays():
    config = Config(
        thruster=SPT_100,
        domain=(0.0, 0.08),
        discharge_voltage=500.0,
        anode_mass_flow_rate=3e-6,
        ncharge=3,
        propellant=Xenon,
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

    expected_state = np.array(
        [
            [2.7977045974887016e-06, 2.775096499378319e-06, 2.7977045975939686e-08,
             2.7977045975932698e-08, 2.7977045975932698e-08, 2.7977045975932698e-08],
            [4.801453835489269e-08, 1.5724744083658822e-07, 3.541724065485228e-08,
             3.3775072752608056e-08, 3.3775072535741284e-08, 3.3775072535741284e-08],
            [-7.129240196091699e-05, 2.4611951142047194e-04, 6.532374048736055e-04,
             7.400098185653353e-04, 8.570704558478136e-04, 9.156007768648216e-04],
            [1.2003634588723173e-08, 3.9311860209147056e-08, 8.85431016371307e-09,
             8.443768188152014e-09, 8.443768133935321e-09, 8.443768133935321e-09],
            [-2.5205670436820762e-05, 8.701638775386776e-05, 2.309542993554143e-04,
             2.616329804260876e-04, 3.030201656423172e-04, 3.2371375909039313e-04],
            [5.334948706099188e-09, 1.7471937870732025e-08, 3.935248961650253e-09,
             3.752785861400895e-09, 3.752785837304587e-09, 3.752785837304587e-09],
            [-1.372022915443681e-05, 4.736572205714286e-05, 1.257155971828362e-04,
             1.4241495598388745e-04, 1.6494328613273678e-04, 1.7620745166437837e-04],
        ]
    )
    expected_ne = np.array(
        [4.037602148750516e17, 1.3223132550281649e18, 2.978279743395136e17,
         2.8401878054553683e17, 2.8401877872187722e17, 2.8401877872187722e17]
    )
    expected_energy_density = np.array(
        [1.8174441606123973e18, 6.990039963145497e18, 3.857051388512641e18,
         1.8109877583494702e18, 2.0236337983940214e18, 2.1301408404140792e18]
    )
    expected_tev = np.array(
        [3.0008638686286067, 3.524147265692909, 8.633734282497219,
         4.250863868628607, 4.750000000001517, 5.0]
    )

    np.testing.assert_allclose(state, expected_state, rtol=2e-15, atol=1e-30)
    np.testing.assert_allclose(cache["ne"], expected_ne, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(cache["nϵ"], expected_energy_density, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(cache["Tev"], expected_tev, rtol=2e-15, atol=1e-15)
