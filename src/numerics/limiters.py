import math
from ..HallThruster import MIN_NUMBER_DENSITY

class SlopeLimiter:
    def __init__(self, limiter_fn):
        self.limiter = limiter_fn

    def __call__(self, r):
        if check_r(r):
            return self.limiter(r)
        else:
            return 0.0

def check_r(r):
    return math.isfinite(r) and r >= 0.0

def __piecewise_constant(_):
    return 0.0

def __no_limiter(_):
    return 1.0

def __van_leer(r):
    return (4.0*r)/((r + 1.0)**2)

def __van_albada(r):
    return 2.0*r/(r*r + 1.0)

def __minmod(r):
    return min(2.0/(1.0 + r), 2.0*r/(1.0 + r))

def __koren(r):
    val = max(0.0, min(2.0*r, min((1.0 + 2.0*r)/3.0, 2.0)))
    return val * 2.0/(r + 1.0)

piecewise_constant = SlopeLimiter(__piecewise_constant)
no_limiter = SlopeLimiter(__no_limiter)
van_leer = SlopeLimiter(__van_leer)
van_albada = SlopeLimiter(__van_albada)
minmod = SlopeLimiter(__minmod)
koren = SlopeLimiter(__koren)


def stage_limiter_U(U, params):
    grid = params["grid"]
    index = params["index"]
    min_Te = params["min_Te"]
    cache = params["cache"]
    mi = params["mi"]
    ncharge = params["ncharge"]

    z_cell = grid.cell_centers
    nϵ = cache["nϵ"]  # electron energy density array
    stage_limiter_U_extended(U, z_cell, nϵ, index, min_Te, ncharge, mi)

def stage_limiter_U_extended(U, z_cell, nϵ, index, min_Te, ncharge, mi):
    min_density = MIN_NUMBER_DENSITY * mi


    for i in range(len(z_cell)):
        # e.g. U[index["ρn"], i] = ...
        # U[index["ρn"], i] = max(U[index["ρn"], i], min_density)

        # Cyrus was here
        new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
        U[new_range, i] = max(U[new_range, i], min_density)



        for Z in range(1, ncharge+1):
            # e.g. index["ρi"][Z-1], index["ρiui"][Z-1]
            dens_idx = index["ρi"][Z]-1
            flux_idx = index["ρiui"][Z]-1

            density_floor = max(U[dens_idx, i], min_density)
            velocity = U[flux_idx, i] / U[dens_idx, i] if U[dens_idx, i] != 0 else 0.0

            U[dens_idx, i] = density_floor
            U[flux_idx, i] = density_floor*velocity

        nϵ[i] = max(nϵ[i], 1.5*MIN_NUMBER_DENSITY*min_Te)

slope_limiters = {
    "piecewise_constant": piecewise_constant,
    "no_limiter": no_limiter,
    "van_leer": van_leer,
    "van_albada": van_albada,
    "minmod": minmod,
    "koren": koren
}
