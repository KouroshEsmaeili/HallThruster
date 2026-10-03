import math
import numpy as np
from dataclasses import dataclass
from .materials import SEE_yield
from ..physics.physicalconstants import e, me

def edge_to_center_density_ratio() -> float:
    return 0.86 / math.sqrt(3)


def wall_electron_temperature(params: dict, transition_length: float, i: int) -> float:
    from ..utilities.smoothing import linear_transition

    cache = params["cache"]
    grid = params["grid"]
    thruster = params["thruster"]
    shielded = thruster.shielded
    # In Julia, Tev[i] with i starting at 1; here we assume i is already 0-indexed.
    Tev_i = cache["Tev"][i]
    # For the channel, use the first element of Tev (index 0) if shielded is True; else use Tev_i.
    Tev_channel = cache["Tev"][0] if shielded else Tev_i
    Tev_plume = Tev_i
    L_ch = thruster.geometry.channel_length
    # Apply linear transition function.
    Tev_new = linear_transition(grid.cell_centers[i], L_ch, transition_length, Tev_channel, Tev_plume)
    return Tev_new


def sheath_potential(Tev: float, gamma: float, mi: float) -> float:
    # Compute sqrt(mi / (2 * π * me))
    factor = math.sqrt(mi / (2 * math.pi * me))
    return Tev * math.log((1 - gamma) * factor)

from .wall_loss_base import WallLossModel

@dataclass
class WallSheath(WallLossModel):
    material: any
    loss_scale: float = 1.0

    @classmethod
    def create(cls, material, alpha: float = 0.15):
        return cls(material=material, loss_scale=alpha)


def freq_electron_wall(model: WallSheath, params: dict, i: int) -> float:
    cache = params["cache"]
    ncharge = params["ncharge"]
    thruster = params["thruster"]
    mi = params["mi"]
    transition_length = params["transition_length"]
    geometry = thruster.geometry
    Δr = geometry.outer_radius - geometry.inner_radius
    Tev = wall_electron_temperature(params, transition_length, i)
    gamma = SEE_yield(model.material, Tev, params["γ_SEE_max"])
    # Store computed SEE coefficient in cache["γ_SEE"]
    cache["γ_SEE"][i] = gamma
    h = edge_to_center_density_ratio()
    j_iw = 0.0
    # Loop over ion charge states (1-indexed)
    for Z in range(1, ncharge + 1):
        # In cache["ni"], assume 0-indexed rows: use index Z-1.
        niw = h * cache["ni"][Z - 1, i]
        j_iw += Z * model.loss_scale * niw * math.sqrt(Z * e * Tev / mi)
    # Compute electron wall collision frequency.
    # Assume cache["ne"] is a 1D array.
    nuew = j_iw / (Δr * (1 - gamma)) / cache["ne"][i]
    return nuew


def wall_power_loss(Q: np.ndarray, model: WallSheath, params: dict) -> None:
    cache = params["cache"]
    grid = params["grid"]
    thruster = params["thruster"]
    mi = params["mi"]
    transition_length = params["transition_length"]
    plume_loss_scale = params["plume_loss_scale"]
    L_ch = thruster.geometry.channel_length

    # Loop over interior cells: Python indices 1 to len(cell_centers)-2.
    for i in range(1, len(grid.cell_centers) - 1):
        Tev = wall_electron_temperature(params, transition_length, i)
        gamma = cache["γ_SEE"][i]
        ϕ_s = sheath_potential(Tev, gamma, mi)
        from ..utilities.smoothing import linear_transition
        nuew = cache["radial_loss_frequency"][i] * linear_transition(
            grid.cell_centers[i], L_ch, transition_length, 1.0, plume_loss_scale
        )
        Q[i] = nuew * (2 * Tev + (1 - gamma) * ϕ_s)
    return


def wall_loss_scale(model: WallSheath) -> float:
    return model.loss_scale
