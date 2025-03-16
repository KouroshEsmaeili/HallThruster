from .no_wall_losses import NoWallLosses
from .constant_sheath_potential import ConstantSheathPotential
from ..physics.physicalconstants import e
from .wall_loss_base import WallLossModel


def freq_electron_wall(model: WallLossModel, params, config, i) -> float:

    raise NotImplementedError(
        f"freq_electron_wall not implemented for wall loss model of type {type(model).__name__}. "
        "See documentation for WallLossModel for a list of required methods."
    )

def wall_power_loss(Q, model: WallLossModel, params, config) -> None:
    raise NotImplementedError(
        f"wall_power_loss not implemented for wall loss model of type {type(model).__name__}. "
        "See documentation for WallLossModel for a list of required methods."
    )

def wall_electron_current(model: WallLossModel, params, i) -> float:
    grid = params["grid"]
    cache = params["cache"]
    thruster = params["thruster"]
    ne = cache["ne"]
    nu_ew_momentum = cache["νew_momentum"]
    A_ch = thruster.geometry.channel_area
    # Assume grid has attribute dz_cell (a list or NumPy array).
    V_cell = A_ch * grid.dz_cell[i]
    return e * nu_ew_momentum[i] * V_cell * ne[i]

def wall_ion_current(model: WallLossModel, params, i, Z: int) -> float:
    cache = params["cache"]
    ne_arr = cache["ne"]
    ni = cache["ni"]
    Iew = wall_electron_current(model, params, i)
    # Adjust for 1-indexing: in Python, array indices are 0-indexed.
    return Z * ni[Z - 1, i] / ne_arr[i] * Iew * (1 - cache["γ_SEE"][i])

# --- Serialization/Model Registry Functions ---
def wall_loss_models():
    import wall_sheath

    return {
        "NoWallLosses": NoWallLosses,
        "ConstantSheathPotential": ConstantSheathPotential,
        "WallSheath": wall_sheath.WallSheath,
    }

