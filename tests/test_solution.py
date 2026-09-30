from types import SimpleNamespace

import numpy as np

from hallthruster.simulation.solution import Solution


def test_solution_reads_dictionary_frames_consistently():
    frame = {
        "Tev": np.array([2.0, 3.0]),
        "∇ϕ": np.array([-4.0, -5.0]),
        "ni": np.array([[1.0, 2.0]]),
    }
    solution = Solution(
        t=[0.0],
        frames=[frame],
        params={"cache": {"B": np.array([0.1, 0.2])}, "grid": SimpleNamespace(cell_centers=[0.0, 1.0])},
        config=SimpleNamespace(ncharge=1),
        retcode="success",
        error="",
    )

    np.testing.assert_allclose(solution["Tev"][0], [2.0, 3.0])
    np.testing.assert_allclose(solution["E"][0], [4.0, 5.0])
    np.testing.assert_allclose(solution["ni", 1][0], [1.0, 2.0])
    np.testing.assert_allclose(solution["B"], [0.1, 0.2])
    assert solution[0].t == [0.0]
