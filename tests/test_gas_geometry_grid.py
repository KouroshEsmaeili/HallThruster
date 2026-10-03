import math

import numpy as np
import pytest

from hallthruster import EvenGrid, Geometry1D, Species, Xenon, generate_grid
from hallthruster.physics.gas import Gas
from hallthruster.physics.physicalconstants import NA, R0
from hallthruster.thruster.geometry import channel_perimeter, channel_width


def test_gas_and_species_match_historical_julia_behavior():
    fake = Gas("Fake", "Fa", gamma=5 / 3, M=5.0)

    assert repr(Xenon) == "Xenon"
    assert repr(Species(Xenon, 0)) == "Xe"
    assert repr(Species(Xenon, 1)) == "Xe+"
    assert repr(Species(Xenon, 3)) == "Xe3+"
    assert fake.m == 5.0 / NA
    assert fake.R == R0 / 5.0
    assert fake.cp == fake.gamma / (fake.gamma - 1) * fake.R
    assert fake.cv == fake.cp - fake.R
    assert fake(2) == Species(fake, 2)


def test_geometry_formulas():
    geometry = Geometry1D(channel_length=0.025, inner_radius=0.03, outer_radius=0.05)

    assert geometry.channel_area == math.pi * (0.05**2 - 0.03**2)
    assert channel_perimeter(geometry.outer_radius, geometry.inner_radius) == 2 * math.pi * 0.08
    assert channel_width(geometry.outer_radius, geometry.inner_radius) == pytest.approx(0.02)


def test_even_grid_includes_historical_ghost_cell_centers():
    geometry = Geometry1D(channel_length=0.025, inner_radius=0.03, outer_radius=0.05)
    grid = generate_grid(EvenGrid(4), geometry, (0.0, 0.08))

    assert grid.num_cells == 6
    np.testing.assert_allclose(grid.edges, [0.0, 0.02, 0.04, 0.06, 0.08])
    np.testing.assert_allclose(grid.cell_centers, [0.0, 0.01, 0.03, 0.05, 0.07, 0.08])
    np.testing.assert_allclose(grid.dz_edge, [0.01, 0.02, 0.02, 0.02, 0.01])
    np.testing.assert_allclose(grid.dz_cell, [0.02] * 6)
