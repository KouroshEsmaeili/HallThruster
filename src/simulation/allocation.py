import numpy as np
from ..grid.gridspec import Grid1D
from ..collisions.anomalous import num_anom_variables
from ..utilities.linearalgebra import Tridiagonal


# counter = 0

def allocate_arrays_from_grid(grid:Grid1D, config):
    # global counter
    # counter += 1
    # print(f"allocate_arrays_from_grid c_{counter}")


    ncells = len(grid.cell_centers)
    nedges = len(grid.edges)
    ncharge = config.ncharge


    n_anom_vars = num_anom_variables(config.anom_model)
    return allocate_arrays(ncells, nedges, ncharge, n_anom_vars)


def allocate_arrays(ncells, nedges, ncharge, n_anom_vars):
    print('[allocate_arrays] allocate_arrays is started!!!')
    nvariables = 1 + 2 * ncharge


    U = np.zeros((nvariables, ncells), dtype=np.float64)


    tri_sub = np.ones(ncells - 1, dtype=np.float64)
    tri_main = np.ones(ncells, dtype=np.float64)
    tri_super = np.ones(ncells - 1, dtype=np.float64)

    cache = {
        # Caches for energy solve
        "Aϵ": Tridiagonal(tri_sub,tri_main,tri_super),

        "bϵ": np.zeros(ncells, dtype=np.float64),

        # Collision frequencies
        "νan": np.zeros(ncells, dtype=np.float64),
        "νc": np.zeros(ncells, dtype=np.float64),
        "νei": np.zeros(ncells, dtype=np.float64),
        "νen": np.zeros(ncells, dtype=np.float64),
        "radial_loss_frequency": np.zeros(ncells, dtype=np.float64),
        "νew_momentum": np.zeros(ncells, dtype=np.float64),
        "νiw": np.zeros(ncells, dtype=np.float64),
        "νe": np.zeros(ncells, dtype=np.float64),
        "νiz": np.zeros(ncells, dtype=np.float64),
        "νex": np.zeros(ncells, dtype=np.float64),

        # Magnetic field
        "B": np.zeros(ncells, dtype=np.float64),

        # Conductivity and mobility
        "κ": np.zeros(ncells, dtype=np.float64),
        "μ": np.zeros(ncells, dtype=np.float64),

        # Potential and electric field
        "ϕ": np.zeros(ncells, dtype=np.float64),
        "∇ϕ": np.zeros(ncells, dtype=np.float64),

        # Electron number density
        "ne": np.zeros(ncells, dtype=np.float64),

        # Electron energy density
        "nϵ": np.zeros(ncells, dtype=np.float64),

        # Electron temperature and energy [eV]
        "Tev": np.zeros(ncells, dtype=np.float64),
        "ϵ": np.zeros(ncells, dtype=np.float64),

        # Electron pressure and gradient
        "pe": np.zeros(ncells, dtype=np.float64),
        "∇pe": np.zeros(ncells, dtype=np.float64),

        # Electron axial velocity and kinetic energy
        "ue": np.zeros(ncells, dtype=np.float64),
        "K": np.zeros(ncells, dtype=np.float64),
        "λ_global": np.zeros(ncharge + 1, dtype=np.float64),

        # Electron source terms
        "ohmic_heating": np.zeros(ncells, dtype=np.float64),
        "wall_losses": np.zeros(ncells, dtype=np.float64),
        "inelastic_losses": np.zeros(ncells, dtype=np.float64),

        # Effective charge number
        "Z_eff": np.zeros(ncells, dtype=np.float64),

        # Ion density, velocity, flux
        "ni": np.zeros((ncharge, ncells), dtype=np.float64),
        "ui": np.zeros((ncharge, ncells), dtype=np.float64),
        "niui": np.zeros((ncharge, ncells), dtype=np.float64),

        # Ion current
        "ji": np.zeros(ncells, dtype=np.float64),

        # Neutral density
        "nn": np.zeros(ncells, dtype=np.float64),
        "γ_SEE": np.zeros(ncells, dtype=np.float64),  # e.g. c["γ_SEE"][i] ?

        "Id": [0.0],  # single-element list
        "Vs": [0.0],
        "anom_multiplier": [1.0],

        # Edge state caches
        "F": np.zeros((nvariables, nedges), dtype=np.float64),
        "UL": np.zeros((nvariables, nedges), dtype=np.float64),
        "UR": np.zeros((nvariables, nedges), dtype=np.float64),

        # timestepping caches
        "k": U.copy(),  # copy of U
        "u1": U.copy(),  # another copy

        # other caches
        "cell_cache_1": np.zeros(ncells, dtype=np.float64),

        # Plume divergence variables
        "channel_area": np.zeros(ncells, dtype=np.float64),
        "dA_dz": np.zeros(ncells, dtype=np.float64),
        "channel_height": np.zeros(ncells, dtype=np.float64),
        "inner_radius": np.zeros(ncells, dtype=np.float64),
        "outer_radius": np.zeros(ncells, dtype=np.float64),
        "tanδ": np.zeros(ncells, dtype=np.float64),

        # Anomalous transport variables
        "anom_variables": [np.zeros(ncells, dtype=np.float64) for _ in range(n_anom_vars)],

        # Timesteps
        "dt_iz": np.zeros(1, dtype=np.float64),
        "dt": np.zeros(1, dtype=np.float64),
        "dt_E": np.zeros(1, dtype=np.float64),
        "dt_u": np.zeros(nedges, dtype=np.float64),
    }

    return U, cache
