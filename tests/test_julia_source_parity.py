"""Component checks derived from HallThruster.jl commit 014a12f.

These are source-level parity tests because Julia is not required by the normal
Python suite.  The optional executable harness under ``validation/`` regenerates
the same cases with Julia when a Julia 1.10 runtime is available.
"""

import math

import numpy as np
import pytest

from hallthruster import Config, EvenGrid, Geometry1D, SPT_100, UnevenGrid, Xenon, generate_grid
from hallthruster.collisions.elastic import ElasticCollision
from hallthruster.collisions.anomalous import TwoZoneBohm
from hallthruster.collisions.excitation import ExcitationReaction, ovs_rate_coeff_ex
from hallthruster.collisions.ionization import IonizationReaction
from hallthruster.collisions.reactions import load_rate_coeffs, rate_coeff
from hallthruster.numerics.finite_differences import (
    backward_diff_coeffs,
    backward_difference,
    central_diff_coeffs,
    central_difference,
    downwind_diff_coeffs,
    forward_diff_coeffs,
    forward_difference,
    second_deriv_central_diff,
    second_deriv_coeffs,
    upwind_diff_coeffs,
)
from hallthruster.numerics.edge_fluxes import reconstruct
from hallthruster.numerics.flux_functions import HLLE, flux, global_lax_friedrichs, rusanov
from hallthruster.numerics.limiters import no_limiter, slope_limiters
from hallthruster.physics.fluid import EulerEquations, IsothermalEuler
from hallthruster.physics.physicalconstants import NA, R0, e, kB, me
from hallthruster.physics.thermal_conductivity import Mitchner
from hallthruster.simulation.configuration import configure_fluids, configure_index
from hallthruster.thruster.geometry import channel_perimeter, channel_width
from hallthruster.utilities.integration import cumtrapz
from hallthruster.utilities.interpolation import LinearInterpolation
from hallthruster.utilities.linearalgebra import Tridiagonal, tridiagonal_solve
from hallthruster.walls.materials import BNSiO2
from hallthruster.walls.wall_sheath import WallSheath


REFERENCE_COMMIT = "014a12fb193af6927cb10f77da5e7baf215b5bc0"


def make_config(**overrides):
    values = {
        "thruster": SPT_100,
        "domain": (0.0, 0.08),
        "discharge_voltage": 300.0,
        "anode_mass_flow_rate": 5e-6,
    }
    values.update(overrides)
    return Config(**values)


def test_physical_constants_and_xenon_match_reference_literals_and_formulas():
    assert e == 1.602176634e-19
    assert me == 9.10938356e-31
    assert kB == 1.380649e-23
    assert NA == 6.02214076e26
    assert R0 == pytest.approx(8314.46261815324, rel=1e-15)

    assert Xenon.M == 131.293
    assert Xenon.m == pytest.approx(2.1801715574645584e-25, rel=1e-15)
    assert Xenon.R == pytest.approx(63.32753930638526, rel=1e-15)
    assert Xenon.cp == pytest.approx(158.31884826596314, rel=1e-15)
    assert Xenon.cv == pytest.approx(94.99130895957788, rel=1e-15)
    assert Xenon.gamma == 5 / 3
    assert [repr(Xenon(z)) for z in (0, 1, 2, 3)] == ["Xe", "Xe+", "Xe2+", "Xe3+"]


def test_spt100_geometry_and_magnetic_field_source_values():
    geometry = SPT_100.geometry
    assert geometry.channel_length == 0.025
    assert geometry.inner_radius == 0.0345
    assert geometry.outer_radius == 0.05
    assert geometry.channel_area == pytest.approx(0.004114700978039233, rel=1e-15)
    assert channel_width(geometry.outer_radius, geometry.inner_radius) == pytest.approx(0.0155)
    assert channel_perimeter(geometry.outer_radius, geometry.inner_radius) == pytest.approx(
        0.5309291584566751, rel=1e-15
    )

    field = LinearInterpolation(SPT_100.magnetic_field.z, SPT_100.magnetic_field.B)
    expected = {
        0.0: 0.0011336081206959115,
        0.0125: 0.007864905175897288,
        0.025: 0.014998492898220965,
        0.05: 0.005717885924781589,
        0.08: 0.00014083593108615071,
        0.1: 2.5478501484211467e-06,
    }
    for z, value in expected.items():
        assert field(z) == pytest.approx(value, rel=2e-15, abs=1e-18)


@pytest.mark.parametrize("num_cells", [4, 10, 20])
def test_even_grid_reference_formulas(num_cells):
    grid = generate_grid(EvenGrid(num_cells), SPT_100.geometry, (0.0, 0.08))
    expected_edges = np.linspace(0.0, 0.08, num_cells + 1)
    expected_centers = np.concatenate(
        ([expected_edges[0]], 0.5 * (expected_edges[:-1] + expected_edges[1:]), [expected_edges[-1]])
    )
    expected_dz_cell = np.concatenate(
        ([np.diff(expected_edges)[0]], np.diff(expected_edges), [np.diff(expected_edges)[-1]])
    )

    assert grid.num_cells == num_cells + 2
    np.testing.assert_allclose(grid.edges, expected_edges, rtol=0.0, atol=1e-17)
    np.testing.assert_allclose(grid.cell_centers, expected_centers, rtol=0.0, atol=1e-17)
    np.testing.assert_allclose(grid.dz_edge, np.diff(expected_centers), rtol=0.0, atol=1e-17)
    np.testing.assert_allclose(grid.dz_cell, expected_dz_cell, rtol=0.0, atol=1e-17)


def test_uneven_grid_matches_historical_inverse_cdf_algorithm():
    grid = generate_grid(UnevenGrid(10), SPT_100.geometry, (0.0, 0.08))
    expected_edges = [
        0.0,
        0.006231258508079,
        0.012462517016158,
        0.018693775524236997,
        0.024925034032316,
        0.031156292540395,
        0.037495287704939,
        0.045055055521590295,
        0.0552520246017432,
        0.06753975250047804,
        0.08,
    ]
    np.testing.assert_allclose(grid.edges, expected_edges, rtol=2e-15, atol=1e-17)
    assert grid.num_cells == 12
    assert grid.dz_cell[-1] == pytest.approx(2 * grid.dz_cell[0], rel=2e-4)


def test_interpolation_clamps_and_handles_nonuniform_locations():
    interpolation = LinearInterpolation([0.0, 2.0, 5.0], [0.0, 4.0, 10.0])
    xs = [-1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 9.0]
    assert [interpolation(x) for x in xs] == [0.0, 0.0, 2.0, 4.0, 7.0, 10.0, 10.0]


def test_finite_difference_coefficients_and_helpers_on_uneven_points():
    points = (0.0, 0.5, 2.0)
    expected = {
        forward_diff_coeffs: (-2.5, 8 / 3, -1 / 6),
        central_diff_coeffs: (-1.5, 4 / 3, 1 / 6),
        backward_diff_coeffs: (1.5, -8 / 3, 7 / 6),
        second_deriv_coeffs: (2.0, -8 / 3, 2 / 3),
        upwind_diff_coeffs: (-2.0, 2.0, 0.0),
        downwind_diff_coeffs: (0.0, -2 / 3, 2 / 3),
    }
    for function, coefficients in expected.items():
        np.testing.assert_allclose(function(*points), coefficients, rtol=1e-15, atol=0.0)

    values = tuple(x * x for x in points)
    assert forward_difference(*values, *points) == pytest.approx(0.0, abs=1e-15)
    assert central_difference(*values, *points) == pytest.approx(1.0, rel=1e-15)
    assert backward_difference(*values, *points) == pytest.approx(4.0, rel=1e-15)
    assert second_deriv_central_diff(*values, *points) == pytest.approx(2.0, rel=1e-15)


def test_cumulative_integration_endpoint_treatment():
    result = cumtrapz([0.0, 0.5, 2.0, 3.0], [1.0, 2.0, -1.0, 4.0], y0=3.0)
    np.testing.assert_allclose(result, [3.0, 3.75, 4.5, 6.0], rtol=0.0, atol=1e-15)


def test_nonmutating_tridiagonal_solve_matches_reference_and_preserves_inputs():
    matrix = Tridiagonal([1.0, -1.0, 2.0], [4.0, 5.0, 6.0, 7.0], [2.0, 3.0, -2.0])
    expected = np.array([1.0, -2.0, 3.0, 0.5])
    rhs = matrix.matvec(expected)
    diagonals_before = (matrix._dl.copy(), matrix._d.copy(), matrix._du.copy())
    rhs_before = rhs.copy()

    np.testing.assert_allclose(tridiagonal_solve(matrix, rhs), expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(rhs, rhs_before)
    for actual, before in zip((matrix._dl, matrix._d, matrix._du), diagonals_before):
        np.testing.assert_array_equal(actual, before)


def test_slope_limiters_cover_invalid_and_representative_ratios():
    ratios = [-2.0, 0.0, 0.25, 1.0, 10.0, math.inf, math.nan]
    expected = {
        "piecewise_constant": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "no_limiter": [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        "van_leer": [0.0, 0.0, 0.64, 1.0, 40 / 121, 0.0, 0.0],
        "van_albada": [0.0, 0.0, 8 / 17, 1.0, 20 / 101, 0.0, 0.0],
        "minmod": [0.0, 0.0, 0.4, 1.0, 2 / 11, 0.0, 0.0],
        "koren": [0.0, 0.0, 0.8, 1.0, 4 / 11, 0.0, 0.0],
    }
    for name, limiter in slope_limiters.items():
        np.testing.assert_allclose([limiter(r) for r in ratios], expected[name], rtol=1e-15, atol=0.0)


def test_reconstruction_with_flat_upwind_slope_matches_julia_nan_limiter_path():
    assert reconstruct(1.0, 1.0, 2.0, no_limiter) == (1.0, 1.0)


def test_flux_functions_match_historical_reference_case():
    fluid = EulerEquations(Xenon(1))
    left = (1.0, 300.0, Xenon.cv * 300.0 + 0.5 * 300.0**2)
    right = (0.5, 50.0, 0.5 * (Xenon.cv * 600.0 + 0.5 * 100.0**2))

    np.testing.assert_allclose(
        flux(left, fluid), [300.0, 108998.26179191557, 27748696.34393668], rtol=2e-15
    )
    np.testing.assert_allclose(
        rusanov(left, right, fluid),
        [294.48579102729923, 126241.1573055652, 26530423.13327822],
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        global_lax_friedrichs(left, right, fluid, max_wave_speed=500.0),
        [300.0, 128998.26179191557, 26999130.895957787],
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        HLLE(left, right, fluid),
        [297.34359165652796, 117304.8333663405, 27161806.884824194],
        rtol=2e-15,
    )


def test_zero_density_flux_uses_julia_ieee_float_semantics():
    result = flux((0.0, 0.0), IsothermalEuler(Xenon(1), 1000.0))
    assert result[0] == 0.0
    assert math.isnan(result[1])


@pytest.mark.parametrize(
    ("reaction_type", "product", "reaction_class", "expected_threshold", "expected"),
    [
        (
            "elastic",
            None,
            ElasticCollision,
            0.0,
            [0.0, 5.130755817456869e-14, 4.013652725669257e-13, 5.751634667357914e-13,
             6.760494967682132e-13, 6.866432405805591e-13, 7.188048405344664e-13,
             7.652612330904358e-13],
        ),
        (
            "excitation",
            None,
            ExcitationReaction,
            8.32,
            [2.909965013767145e-20, 2.909965013767145e-20, 5.538741602616681e-15,
             2.3582514835893516e-14, 5.578121624167219e-14, 8.128975696965179e-14,
             8.144339610474207e-14, 7.702891199553481e-14],
        ),
        (
            "ionization",
            Xenon(1),
            IonizationReaction,
            12.1298437,
            [0.0, 3.6020165314348953e-16, 3.3936755254699027e-15,
             1.77123567888999e-14, 6.500964285714286e-14, 1.6658392857142858e-13,
             2.450357142857143e-13, 3.23348e-13],
        ),
    ],
)
def test_reaction_tables_match_source_derived_reference_values(
    reaction_type, product, reaction_class, expected_threshold, expected
):
    threshold, coefficients = load_rate_coeffs(Xenon(0), product, reaction_type)
    if reaction_type == "elastic":
        reaction = reaction_class(Xenon(0), coefficients)
    elif reaction_type == "excitation":
        reaction = reaction_class(threshold, Xenon(0), coefficients)
    else:
        reaction = reaction_class(threshold, Xenon(0), product, coefficients)

    assert threshold == pytest.approx(expected_threshold, rel=0.0, abs=1e-15)
    energies = [0.0, 1.0, 5.5, 10.0, 20.0, 50.0, 100.0, 255.0]
    np.testing.assert_allclose(
        [rate_coeff(reaction, energy) for energy in energies], expected, rtol=2e-15, atol=1e-30
    )


def test_ovs_excitation_zero_energy_matches_julia_float_behavior():
    assert ovs_rate_coeff_ex(0.0) == 0.0


def test_config_defaults_match_historical_spt100_semantics():
    config = make_config(ncharge=3)

    assert config.thruster is SPT_100
    assert config.propellant is Xenon
    assert config.ncharge == 3
    assert config.domain == (0.0, 0.08)
    assert config.discharge_voltage == 300.0
    assert config.anode_mass_flow_rate == 5e-6
    assert config.cathode_coupling_voltage == 0.0
    assert config.anode_Tev == config.cathode_Tev == 2.0
    assert config.neutral_velocity == 150.0
    assert config.neutral_temperature_K == 500.0
    assert config.ion_temperature_K == 1000.0
    assert config.background_pressure_Torr == 0.0
    assert config.background_temperature_K == 100.0
    assert config.transition_length == pytest.approx(0.0025)
    assert config.solve_plume is False
    assert config.ion_wall_losses is False
    assert config.electron_ion_collisions is True
    assert config.magnetic_field_scale == 1.0
    assert isinstance(config.anom_model, TwoZoneBohm)
    assert (config.anom_model.c1, config.anom_model.c2) == (1 / 160, 1 / 16)
    assert isinstance(config.wall_loss_model, WallSheath)
    assert config.wall_loss_model.material is BNSiO2
    assert config.wall_loss_model.loss_scale == 1.0
    assert isinstance(config.conductivity_model, Mitchner)
    assert (config.ionization_model, config.excitation_model, config.electron_neutral_model) == (
        "Lookup", "Lookup", "Lookup"
    )
    assert config.source_neutrals(None, None, 0) == 0.0
    assert config.source_energy(None, 0) == 0.0
    assert [source(None, None, 0) for source in config.source_ion_continuity] == [0.0] * 3
    assert [source(None, None, 0) for source in config.source_ion_momentum] == [0.0] * 3


@pytest.mark.parametrize("ncharge", [1, 2, 3])
def test_fluid_and_index_mapping_refers_to_same_physical_variables(ncharge):
    fluids, fluid_ranges, species, species_ranges, velocity_indices = configure_fluids(
        make_config(ncharge=ncharge)
    )
    index = configure_index(fluids, fluid_ranges)

    assert [fluid.species.Z for fluid in fluids] == list(range(ncharge + 1))
    assert [repr(item) for item in species] == ["Xe"] + [
        "Xe+" if charge == 1 else f"Xe{charge}+" for charge in range(1, ncharge + 1)
    ]
    assert [(item.start, item.stop) for item in fluid_ranges] == [(1, 2)] + [
        (2 * charge, 2 * charge + 2) for charge in range(1, ncharge + 1)
    ]
    assert (index["ρn"].start, index["ρn"].stop) == (1, 2)
    assert index["ρi"] == {charge: 2 * charge for charge in range(1, ncharge + 1)}
    assert index["ρiui"] == {charge: 2 * charge + 1 for charge in range(1, ncharge + 1)}
    assert velocity_indices == [position % 2 == 1 and position >= 3 for position in range(1, 2 * ncharge + 2)]
    for fluid, fluid_range in zip(fluids, fluid_ranges):
        assert species_ranges[str(fluid.species)] == fluid_range
