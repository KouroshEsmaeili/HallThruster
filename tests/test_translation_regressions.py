from types import SimpleNamespace

import numpy as np

from hallthruster import REACTION_FOLDER, Xenon
from hallthruster.collisions.reactions import load_rate_coeffs
from hallthruster.simulation.sourceterms import (
    _apply_reactions,
    apply_user_ion_source_terms,
)


def test_user_source_terms_are_called_per_interior_cell():
    state = np.zeros((3, 5))
    derivative = np.zeros_like(state)
    params = {
        "grid": SimpleNamespace(cell_centers=np.arange(5.0)),
        "index": {"ρn": range(1, 2), "ρi": {1: 2}, "ρiui": {1: 3}},
        "ncharge": 1,
    }

    apply_user_ion_source_terms(
        derivative,
        state,
        params,
        lambda U, p, i: i,
        [lambda U, p, i: 2 * i],
        [lambda U, p, i: 3 * i],
    )

    np.testing.assert_allclose(derivative[:, 1:4], [[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    np.testing.assert_allclose(derivative[:, [0, 4]], 0.0)


def test_neutral_ionization_uses_neutral_velocity_for_product_momentum():
    state = np.zeros((3, 3))
    state[0, :] = 2.0
    state[1, :] = 1.0
    derivative = np.zeros_like(state)
    cache = {
        "inelastic_losses": np.zeros(3),
        "νiz": np.zeros(3),
        "ϵ": np.zeros(3),
        "ne": np.zeros(3),
        "K": np.zeros(3),
        "cell_cache_1": np.zeros(3),
        "nϵ": np.ones(3),
        "dt_iz": np.zeros(1),
    }
    index = {"ρn": range(1, 2), "ρi": {1: 2}, "ρiui": {1: 3}}
    reaction = SimpleNamespace(rate_coeffs=[1.0, 1.0, 1.0], energy=1.0)

    _apply_reactions(
        derivative,
        state,
        cache,
        index,
        ncharge=1,
        mi=1.0,
        landmark=False,
        un=3.0,
        rxns=[(reaction, 1, 2)],
    )

    assert derivative[0, 1] == -2.0
    assert derivative[1, 1] == 2.0
    assert derivative[2, 1] == 6.0


def test_elastic_lookup_without_reaction_energy_header_loads():
    energy, coefficients = load_rate_coeffs(
        Xenon(0), None, "elastic", REACTION_FOLDER
    )

    assert energy == 0.0
    assert len(coefficients) == 256
    assert coefficients[0] == 0.0
