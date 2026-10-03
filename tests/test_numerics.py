import numpy as np

from hallthruster.numerics.finite_differences import (
    central_difference,
    second_deriv_central_diff,
)
from hallthruster.utilities.integration import cumtrapz
from hallthruster.utilities.interpolation import LinearInterpolation
from hallthruster.utilities.linearalgebra import Tridiagonal


def test_linear_interpolation_clamps_and_interpolates():
    interpolation = LinearInterpolation([1.0, 2.0], [2.0, 4.0])

    assert interpolation(0.0) == 2.0
    assert interpolation(1.5) == 3.0
    assert interpolation(3.0) == 4.0


def test_cumulative_trapezoid_rule():
    result = cumtrapz([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result, [0.0, 0.5, 2.0])


def test_finite_differences_on_quadratic():
    assert central_difference(0.0, 1.0, 4.0, 0.0, 1.0, 2.0) == 2.0
    assert second_deriv_central_diff(0.0, 1.0, 4.0, 0.0, 1.0, 2.0) == 2.0


def test_tridiagonal_solver():
    matrix = Tridiagonal([1.0, 1.0], [4.0, 4.0, 4.0], [1.0, 1.0])
    expected = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(matrix.solve(matrix.matvec(expected)), expected)
