import math
import numpy as np
from ..numerics.edge_fluxes import compute_fluxes_overall
from ..simulation.sourceterms import apply_user_ion_source_terms, apply_reactions, apply_ion_acceleration, apply_ion_wall_losses
from ..simulation.boundaryconditions import left_boundary_state, right_boundary_state
from ..physics.physicalconstants import e

def update_convective_term(dU, U, F, grid, index, cache, ncharge):
    dA_dz = cache["dA_dz"]
    channel_area = cache["channel_area"]
    ncells = len(grid.cell_centers)

    # Loop over interior cells (Python indexing: i from 1 to ncells-2)
    for i in range(1, ncells - 1):
        left = i - 1
        right = i  # assuming flux differences between cell boundaries: adjust as needed
        dz = grid.dz_cell[i]
        dlnA_dz = dA_dz[i] / channel_area[i]

        # Neutral flux update:

        # dU[index["ρn"], i] = (F[index["ρn"], left] - F[index["ρn"], right]) / dz
        # Cyrus was here
        # dU[0, i] = (F[0, left] - F[0, right]) / dz

        new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
        dU[new_range, i] = (F[new_range, left] - F[new_range, right]) / dz


        # Ion updates:
        for Z in range(1, ncharge + 1):
            row_ni = index["ρi"][Z]-1      # assume index["ρi"] is a dict mapping Z to a row index
            row_niui = index["ρiui"][Z]-1    # similar for momentum
            rho_i = U[row_ni, i]
            rho_iui = U[row_niui, i]
            dU[row_ni, i] = (F[row_ni, left] - F[row_ni, right]) / dz - rho_iui * dlnA_dz
            dU[row_niui, i] = (F[row_niui, left] - F[row_niui, right]) / dz - (rho_iui**2 / rho_i) * dlnA_dz

def iterate_heavy_species(dU, U, params, scheme, sources, apply_boundary_conditions=False):
    # print("[iterate_heavy_species] U", U)

    index = params["index"]

    cache = params["cache"]
    grid = params["grid"]
    simulation = params["simulation"]
    ncharge = params["ncharge"]
    ion_wall_losses = params["ion_wall_losses"]

    # print('[iterate_heavy_species]params["cache"]', params["cache"])
    # print('[iterate_heavy_species]cache["F"]',cache["F"])
    # print('[iterate_heavy_species]cache["UL"]', cache["UL"])
    # print('[iterate_heavy_species]cache["UR"]', cache["UR"])

    # Unpack fields from cache:
    F = cache["F"]
    UL = cache["UL"]
    UR = cache["UR"]

    # Compute fluxes and update convective term:
    # (Assumes compute_fluxes and update_convective_term are implemented elsewhere)
    # print('[iterate_heavy_species]scheme',scheme)
    compute_fluxes_overall(F, UL, UR, U, params, scheme, apply_boundary_conditions=apply_boundary_conditions)
    update_convective_term(dU, U, F, grid, index, cache, ncharge)

    # Apply user-provided ion source terms:
    apply_user_ion_source_terms(dU, U, params,
                                sources["source_neutrals"],
                                sources["source_ion_continuity"],
                                sources["source_ion_momentum"])

    # Apply reaction terms:
    apply_reactions(dU, U, params)

    # Apply ion acceleration:
    apply_ion_acceleration(dU, U, params)

    # Optionally apply ion wall losses:
    if ion_wall_losses:
        apply_ion_wall_losses(dU, U, params)

    # Update the timestep from adaptive criteria:
    simulation_CFL = simulation.CFL
    ncells = len(grid.cell_centers)
    update_timestep(cache, dU, simulation_CFL, ncells)
    # No explicit return needed (updates in place)


def update_timestep(cache, dU, CFL, ncells):
    # Assume dt, dt_iz, dt_E are stored as one-element lists, and dt_u is a 1D NumPy array.
    dt_candidate = min(
        CFL * cache["dt_iz"][0],
        math.sqrt(CFL) * cache["dt_E"][0],
        CFL * np.min(cache["dt_u"][:ncells - 1])
    )
    cache["dt"][0] = dt_candidate

    # Set the first and last column of dU to zero:
    dU[:, 0] = 0.0
    dU[:, -1] = 0.0




def integrate_heavy_species(U, params, config, dt, apply_boundary_conditions=True):
    sources = {
        "source_neutrals": config.source_neutrals,
        "source_ion_continuity": config.source_ion_continuity,
        "source_ion_momentum": config.source_ion_momentum
    }
    # Assume that config["scheme"] holds a scheme object.
    integrate_heavy_species_stage(U, params, config["scheme"], sources, dt, apply_boundary_conditions)


def integrate_heavy_species_stage(U, params, scheme, sources, dt, apply_boundary_conditions=True):

    from ..numerics.limiters import stage_limiter_U

    cache = params["cache"]
    # Assume cache["k"] and cache["u1"] are NumPy arrays of same shape as U.
    k = cache["k"]
    u1 = cache["u1"]

    # First RK stage:
    # print('[integrate_heavy_species_stage]',scheme)
    iterate_heavy_species(k, U, params, scheme, sources, apply_boundary_conditions)
    # Update u1: elementwise u1 = U + dt * k.
    u1[:] = U + dt * k
    # Apply stage limiter (assumed implemented elsewhere)
    stage_limiter_U(u1, params)

    # Second RK stage:
    iterate_heavy_species(k, u1, params, scheme, sources, apply_boundary_conditions)
    # print('[integrate_heavy_species_stage] U',U)
    U[:] = (U + u1 + dt * k) / 2
    stage_limiter_U(U, params)


def update_heavy_species_calc(U, cache, index, z_cell, ncharge, mi, landmark):
    # print('[update_heavy_species_calc] mi:', mi)
    inv_m = 1.0 / mi
    # Update neutral number density:
    # Assuming index["ρn"] gives the row index for neutral density.
    # cache["nn"][:] = U[index["ρn"], :] * inv_m


    # Cyrus was here
    # cache["nn"][:] = U[0, :] * inv_m
    # print('[update_heavy_species_calc] new_range:',new_range)
    new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
    cache["nn"][:] = U[new_range, :] * inv_m

    ncells = len(z_cell)
    # print('[update_heavy_species_calc] ncells:',ncells)
    # print('[update_heavy_species_calc] cache["ni"]:',cache["ni"])

    for i in range(ncells):
        ne_val = 0.0
        Z_eff_val = 0.0
        ji_val = 0.0
        # print("[update_heavy_species_calc] U", U)

        for Z in range(1, ncharge + 1):
            row_ni = index["ρi"][Z]-1
            row_niui = index["ρiui"][Z]-1
            # print(row_niui)
            _ni = U[row_ni, i] * inv_m
            _niui = U[row_niui, i] * inv_m
            # print('[update_heavy_species_calc] _niui', _niui)

            cache["ni"][Z - 1, i] = _ni
            cache["niui"][Z - 1, i] = _niui
            cache["ui"][Z - 1, i] = _niui / _ni
            ne_val += Z * _ni
            Z_eff_val += _ni
            ji_val += Z * e * _niui
        cache["ne"][i] = ne_val
        # Effective ion charge state is density weighted average (avoid division by zero)
        cache["Z_eff"][i] = max(1.0, ne_val / Z_eff_val) if Z_eff_val != 0 else 1.0
        cache["ji"][i] = ji_val
        # print('[update_heavy_species_calc] i',i)
        # print('[update_heavy_species_calc] cache["ji"][i]',cache["ji"][i])

    # Update ϵ = nϵ/ne + landmark * K elementwise.
    # Assume cache["nϵ"] and cache["K"] and cache["ne"] are NumPy arrays.
    cache["ϵ"][:] = cache["nϵ"] / cache["ne"] + (landmark * cache["K"])


def update_heavy_species(U, params):

    index = params["index"]
    grid = params["grid"]
    cache = params["cache"]
    mi = params["mi"]
    ncharge = params["ncharge"]
    landmark = params["landmark"]

    # Apply fluid boundary conditions.
    # Assume left_boundary_state and right_boundary_state update U in place.
    left_boundary_state(U[:, 0], U, params)
    right_boundary_state(U[:, -1], U, params)

    update_heavy_species_calc(U, cache, index, grid.cell_centers, ncharge, mi, landmark)
    # print("[update_heavy_species] params_dict.cache['nn']",params['cache']['nn'])

