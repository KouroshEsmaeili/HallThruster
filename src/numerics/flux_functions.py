import numpy as np
from ..physics.thermodynamics import velocity, sound_speed, pressure


class FluxFunction:

    def __init__(self, flux_callable):
        self.flux = flux_callable

    def __call__(self, *args, **kwargs):

        return self.flux(*args, **kwargs)




def __rusanov(UL, UR, fluid, max_wave_speed=0.0, *args, **kwargs):
    dim = len(UL)
    # compute wave speeds
    uL = velocity(UL, fluid)
    uR = velocity(UR, fluid)
    aL = sound_speed(UL, fluid)
    aR = sound_speed(UR, fluid)

    sL_max = max(abs(uL - aL), abs(uL + aL), abs(uL))
    sR_max = max(abs(uR - aR), abs(uR + aR), abs(uR))
    smax = max(sL_max, sR_max)

    FL = flux(UL, fluid)
    FR = flux(UR, fluid)

    # print('[__rusanov] flux(UR, fluid)', flux(UR, fluid))
    # print('[__rusanov] flux(UL, fluid)', flux(UL, fluid))

    out = []
    for j in range(dim):
        val = 0.5 * ((FL[j] + FR[j]) - smax * (UR[j] - UL[j]))
        out.append(val)
    return tuple(out)


def __HLLE(UL, UR, fluid, *args, **kwargs):
    dim = len(UL)
    uL = velocity(UL, fluid)
    uR = velocity(UR, fluid)
    aL = sound_speed(UL, fluid)
    aR = sound_speed(UR, fluid)

    sL_min = min(0.0, uL - aL)
    sL_max = max(0.0, uL + aL)
    sR_min = min(0.0, uR - aR)
    sR_max = max(0.0, uR + aR)

    smin = min(sL_min, sR_min)
    smax = max(sL_max, sR_max)

    FL = flux(UL, fluid)
    FR = flux(UR, fluid)

    out = []
    denom = (smax - smin) if (smax - smin) != 0.0 else 1e-16  # avoid div by zero
    for j in range(dim):
        val = 0.5 * (FL[j] + FR[j]) \
              - 0.5 * ((smax + smin) / denom) * (FR[j] - FL[j]) \
              + (smax * smin / denom) * (UR[j] - UL[j])
        out.append(val)
    return tuple(out)


def __global_lax_friedrichs(UL, UR, fluid, max_wave_speed=0.0, *args, **kwargs):
    dim = len(UL)
    FL = flux(UL, fluid)
    FR = flux(UR, fluid)

    out = []
    for j in range(dim):
        val = 0.5 * (FL[j] + FR[j]) + 0.5 * max_wave_speed * (UL[j] - UR[j])
        out.append(val)
    return tuple(out)


rusanov = FluxFunction(__rusanov)
global_lax_friedrichs = FluxFunction(__global_lax_friedrichs)
HLLE = FluxFunction(__HLLE)


def flux(U, fluid):
    # print('[flux] U:', U)

    dim = len(U)
    if dim == 1:
        rho = U[0]

        if hasattr(fluid, "u"):
            # print('[flux] hasattr(fluid, "u")',hasattr(fluid, "u"))
            return (rho * fluid.u,)
        else:
            return (0.0,)
    elif dim == 2:
        rho, rho_u = U


        p = pressure(U, fluid)
        # print('[flux] rho:', rho)
        # print('[flux] rho_u:', rho_u)
        # print('[flux]  p:', p)


        momentum_term = rho_u * rho_u / rho if rho != 0 else float('inf')

        if not np.isfinite(momentum_term):
            print(f"Overflow check: rho={rho}, rho_u={rho_u}, => momentum_term={momentum_term}")

        return rho_u, (rho_u * rho_u / rho) + p
    elif dim == 3:
        # U = (ρ, ρu, ρE).
        rho, rho_u, rhoE = U
        u = rho_u / rho if rho != 0.0 else 0.0
        p = pressure(U, fluid)
        rhoH = rhoE + p
        return (rho_u, (rho_u * rho_u / rho) + p, rhoH * u)
    else:
        raise ValueError(f"flux not implemented for dim={dim}")


flux_functions = {
    "rusanov": rusanov,
    "global_lax_friedrichs": global_lax_friedrichs,
    "HLLE": HLLE
}
