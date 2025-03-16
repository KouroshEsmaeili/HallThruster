import math
import numpy as np
from ..physics.physicalconstants import e
from ..walls.wall_sheath import wall_power_loss, edge_to_center_density_ratio
from ..collisions.reactions import rate_coeff
from ..utilities.smoothing import linear_transition


# counter_2 = 0
def reaction_rate(rate_coeff_value, ne_value, n_reactant):
    # global counter_2
    # counter_2 += 1
    # print('[reaction_rate] counter_2', counter_2)
    #
    # print('[reaction_rate] rate_coeff_value * ne_value * n_reactant', rate_coeff_value * ne_value * n_reactant)
    # print('[reaction_rate] rate_coeff_value', rate_coeff_value)
    # print('[reaction_rate] ne_value ', ne_value)
    # print('[reaction_rate] n_reactant', n_reactant)
    # if rate_coeff_value == 0.0:
    #     print('dummmy')
    #     return 0.0

    return rate_coeff_value * ne_value * n_reactant


# --- Apply Ionization Reactions ---
def apply_reactions(dU, U, params):
    index = params["index"]
    ionization_reactions = params["ionization_reactions"]
    ion_reactant_indices = params["ionization_reactant_indices"]
    ion_product_indices = params["ionization_product_indices"]
    cache = params["cache"]
    # print('[apply_reactions] cache', cache)
    mi = params["mi"]
    landmark = params["landmark"]
    ncharge = params["ncharge"]
    un = params["neutral_velocity"]

    # Create a zipped iterable over reactions and their indices.
    # print('[apply_reactions] ion_reactant_indices:', ion_reactant_indices)
    # print('[apply_reactions] ion_product_indices:', ion_product_indices)

    rxns = list(zip(ionization_reactions, ion_reactant_indices, ion_product_indices))
    _apply_reactions(dU, U, cache, index, ncharge, mi, landmark, un, rxns)
    return


def _apply_reactions(dU, U, cache, index, ncharge, mi, landmark, un, rxns):
    # print('[_apply_reactions] cache:', cache["ne"])
    inelastic_losses = cache["inelastic_losses"]
    νiz = cache["νiz"]
    ϵ = cache["ϵ"]
    ne = cache["ne"]
    K = cache["K"]
    ncells = len(ne)
    inv_m = 1.0 / mi

    # Set νiz and inelastic_losses to zero.
    νiz.fill(0.0)
    inelastic_losses.fill(0.0)

    # Recompute electron density from ion densities.
    for i in range(ncells):
        ne[i] = 0.0
        for Z in range(1, ncharge + 1):
            # Assume index["ρi"][Z] gives the row index for ion density of charge Z.
            ne[i] += Z * U[index["ρi"][Z] - 1, i]
        ne[i] *= inv_m

    # Compute inverse of ne and update ϵ = nϵ * inv(ne).
    inv_ne = cache["cell_cache_1"]  # We'll reuse this array to store inv(ne)
    # print('[_apply_reactions] cache["ne"]:',cache["ne"])
    np.copyto(inv_ne, 1.0 / (cache["ne"]))
    ϵ[:] = cache["nϵ"] * inv_ne
    if not landmark:
        ϵ[:] += K

    dt_max = float("inf")

    # Loop over each reaction and then each interior cell.
    # print('[_apply_reactions] rxns', rxns)
    for (rxn, reactant_index, product_index) in rxns:
        # print('[_apply_reactions] rxn', rxn)
        # print('[_apply_reactions] reactant_index', rxns)
        # print('[_apply_reactions] product_index', product_index)
        # print('[_apply_reactions] reactant_index', reactant_index)

        for i in range(1, ncells - 1):
            r = rate_coeff(rxn, ϵ[i])
            ρ_reactant = U[reactant_index - 1, i]
            ρdot = reaction_rate(r, ne[i], ρ_reactant)
            dt_max = min(dt_max, ρ_reactant / ρdot if ρdot != 0 else dt_max)
            ndot = ρdot * inv_m
            νiz[i] += ndot * inv_ne[i]
            inelastic_losses[i] += ndot * rxn.energy

            # Change in density due to ionization.
            # dU[reactant_index , i] -= ρdot
            # dU[product_index , i] += ρdot
            # Cyrus was here for  a llreactant_index, product_index
            dU[reactant_index - 1, i] -= ρdot
            dU[product_index - 1 , i] += ρdot

            if not landmark:
                # Cyrus was here
                # new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
                # print('[_apply_reactions] reactant_index', reactant_index)
                # if reactant_index == new_range:

                if reactant_index == index["ρn"]:
                    # print('[_apply_reactions] reactant_index == index["ρn"]:',reactant_index == index["ρn"])
                    reactant_velocity = un
                else:
                    reactant_velocity = U[reactant_index, i] / ρ_reactant if ρ_reactant != 0 else 0.0
                    dU[reactant_index, i] -= ρdot * reactant_velocity
                dU[product_index, i] += ρdot * reactant_velocity
    cache["dt_iz"][0] = dt_max
    return


# --- Apply Ion Acceleration ---
def apply_ion_acceleration(dU, U, params):
    index = params["index"]
    grid = params["grid"]
    cache = params["cache"]
    mi = params["mi"]
    ncharge = params["ncharge"]
    _apply_ion_acceleration(dU, U, grid, cache, index, mi, ncharge)
    return


def _apply_ion_acceleration(dU, U, grid, cache, index, mi, ncharge):
    inv_m = 1.0 / mi
    inv_e = 1.0 / e
    dt_max = float("inf")
    cell_centers = grid.cell_centers
    for i in range(1, len(cell_centers) - 1):
        E = -cache["∇ϕ"][i]
        Δz = grid.dz_cell[i]
        inv_E = 1.0 / abs(E) if abs(E) != 0 else float("inf")
        for Z in range(1, ncharge + 1):
            Q_accel = Z * e * U[index["ρi"][Z] - 1, i] * inv_m * E
            dt_candidate = math.sqrt(mi * Δz * inv_e * inv_E / Z) if inv_E != float("inf") else float("inf")
            dt_max = min(dt_max, dt_candidate)
            dU[index["ρiui"][Z] - 1, i] += Q_accel
    cache["dt_E"][0] = dt_max
    return


# --- Apply Ion Wall Losses ---
def apply_ion_wall_losses(dU, U, params):
    index = params["index"]
    ncharge = params["ncharge"]
    mi = params["mi"]
    thruster = params["thruster"]
    cache = params["cache"]
    grid = params["grid"]
    transition_length = params["transition_length"]
    wall_loss_scale = params["wall_loss_scale"]
    geometry = thruster.geometry
    L_ch = geometry.channel_length
    inv_Δr = 1.0 / (geometry.outer_radius - geometry.inner_radius) if (
                                                                              geometry.outer_radius - geometry.inner_radius) != 0 else float(
        "inf")
    e_inv_m = e / mi
    h = wall_loss_scale * edge_to_center_density_ratio()

    for i in range(1, len(grid.cell_centers) - 1):
        u_bohm = math.sqrt(e_inv_m * cache["Tev"][i])
        # in_channel is computed via linear_transition.
        in_channel = linear_transition(grid.cell_centers[i], L_ch, transition_length, 1.0, 0.0)
        νiw_base = in_channel * u_bohm * inv_Δr * h
        for Z in range(1, ncharge + 1):
            νiw = math.sqrt(Z) * νiw_base
            density_loss = U[index["ρi"][Z] - 1, i] * νiw
            momentum_loss = U[index["ρiui"][Z] - 1, i] * νiw
            dU[index["ρi"][Z] - 1, i] -= density_loss
            # dU[index["ρn"], i] += density_loss
            # cyrus was here
            # dU[0, i] += density_loss
            new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
            dU[new_range, i] += density_loss

            dU[index["ρiui"][Z] - 1, i] -= momentum_loss
    return


def apply_user_ion_source_terms(dU, U, params, source_neutrals, source_ion_continuity, source_ion_momentum):
    index = params["index"]
    grid = params["grid"]
    ncharge = params["ncharge"]
    cell_centers = grid.cell_centers
    for i in range(1, len(cell_centers) - 1):
        # dU[index["ρn"], i]+= source_neutrals(U, params, i)
        # Cyrus was here
        new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
        dU[new_range, i] += source_neutrals

        for Z in range(1, ncharge + 1):
            dU[index["ρi"][Z] - 1, i] += source_ion_continuity[Z - 1]
            dU[index["ρiui"][Z] - 1, i] += source_ion_momentum[Z - 1]


def excitation_losses(Q, cache, landmark, grid, excitation_reactions):
    nu_ex = cache["νex"]
    ϵ_arr = cache["ϵ"]
    nn_arr = cache["nn"]
    ne_arr = cache["ne"]
    K_arr = cache["K"]

    cell_centers = grid.cell_centers
    nu_ex.fill(0.0)

    for rxn in excitation_reactions:
        for i in range(1, len(cell_centers) - 1):
            # snippet => r= rate_coeff(rxn, ϵ[i])
            r_val = rate_coeff(rxn, ϵ_arr[i])
            # ndot= reaction_rate(r, ne[i], nn[i])
            n_dot = reaction_rate(r_val, ne_arr[i], nn_arr[i])
            nu_ex[i] += n_dot / ne_arr[i]
            # Q[i]+= n_dot* (rxn.energy - (!landmark)* K[i])
            # if not landmark => subtract K[i], else 0
            correction = K_arr[i] if (not landmark) else 0.0
            Q[i] += n_dot * (rxn.energy - correction)


def ohmic_heating(Q, cache, landmark):
    ne_arr = cache["ne"]
    ue_arr = cache["ue"]
    grad_phi = cache["∇ϕ"]
    K_arr = cache["K"]
    nu_e = cache["νe"]
    grad_pe = cache["∇pe"]

    if landmark:
        # snippet => @. Q= ne* ue* ∇ϕ
        for i in range(len(Q)):
            Q[i] = ne_arr[i] * ue_arr[i] * grad_phi[i]
    else:
        # snippet => Q= 2* ne*K* νe + ue* ∇pe
        for i in range(len(Q)):
            Q[i] = 2.0 * ne_arr[i] * K_arr[i] * nu_e[i] + ue_arr[i] * grad_pe[i]


# --- Source Electron Energy ---
def source_electron_energy(Q, params, wall_loss_model):
    cache = params["cache"]
    landmark = params["landmark"]
    grid = params["grid"]
    excitation_reactions = params["excitation_reactions"]

    ne_arr = cache["ne"]
    ohm_heat = cache["ohmic_heating"]
    wall_losses = cache["wall_losses"]
    inelastic_losses = cache["inelastic_losses"]

    # ohmic_heating
    ohmic_heating(ohm_heat, cache, landmark)

    # excitation_losses
    excitation_losses(inelastic_losses, cache, landmark, grid, excitation_reactions)

    # wall_power_loss
    wall_power_loss(wall_losses, wall_loss_model, params)

    # Q= ohmic_heating - ne* wall_losses - inelastic_losses
    for i in range(len(Q)):
        Q[i] = ohm_heat[i] - ne_arr[i] * wall_losses[i] - inelastic_losses[i]

    return Q
