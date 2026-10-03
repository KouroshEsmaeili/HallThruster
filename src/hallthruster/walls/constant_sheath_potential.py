import math
import numpy as np
from dataclasses import dataclass
from ..utilities.smoothing import linear_transition  # Adjust import path as needed.


class WallLossModel:
    pass


@dataclass
class ConstantSheathPotential(WallLossModel):
    sheath_potential: float = 20.0
    inner_loss_coeff: float = None
    outer_loss_coeff: float = None


def freq_electron_wall(model: ConstantSheathPotential, _params, _i) -> float:
    return 1e7


def wall_power_loss(Q: np.ndarray, model: ConstantSheathPotential, params: dict) -> None:
    cache = params["cache"]
    grid = params["grid"]
    transition_length = params["transition_length"]
    thruster = params["thruster"]
    L_ch = thruster.geometry.channel_length

    epsilon = cache["ϵ"]
    # Assume linear_transition is defined or imported.

    # Loop over interior cells: in Python, indices 1 to len(cell_centers)-2.
    for i in range(1, len(grid.cell_centers) - 1):
        alpha = linear_transition(grid.cell_centers[i], L_ch, transition_length,
                                  model.inner_loss_coeff, model.outer_loss_coeff)
        Q[i] = 1e7 * alpha * epsilon[i] * math.exp(-model.sheath_potential / epsilon[i])
    return


def wall_loss_scale(model: ConstantSheathPotential) -> float:
    return 1.0
