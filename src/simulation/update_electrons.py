from ..physics.physicalconstants import e, me
from ..utilities.smoothing import linear_transition, smooth
from ..utilities.integration import cumtrapz, cumtrapz_inplace
from ..numerics.finite_differences import forward_difference, central_difference, backward_difference
from ..collisions.collision_frequencies import electron_mobility, freq_electron_ion_inplace, freq_electron_classical, \
    freq_electron_neutral_inplace
from .potential import anode_sheath_potential, compute_electric_field
from .current_control import apply_controller
from .electronenergy import update_pressure, update_temperature, update_electron_energy
from ..walls.wall_sheath import freq_electron_wall as freq_electron_wall_impl


def update_electrons(params, config, t=0.0):
    cache = params["cache"]

    nn = cache["nn"]
    Tev = cache["Tev"]
    pe = cache["pe"]
    ne = cache["ne"]
    nϵ = cache["nϵ"]
    nu_an = cache["νan"]
    nu_c = cache["νc"]
    nu_en = cache["νen"]
    nu_ei = cache["νei"]
    radial_loss_freq = cache["radial_loss_frequency"]
    Z_eff = cache["Z_eff"]
    nu_iz = cache["νiz"]
    nu_ex = cache["νex"]
    nu_ew_momentum = cache["νew_momentum"]
    kappa = cache["κ"]

    source_energy = config.source_energy
    wall_loss_model = config.wall_loss_model
    conductivity_model = config.conductivity_model
    anom_model = config.anom_model

    # Update electron temperature / pressure

    update_temperature(Tev, nϵ, ne, params["min_Te"])
    update_pressure(pe, nϵ, params["landmark"])

    # Update collision frequencies
    if params["electron_ion_collisions"]:
        freq_electron_ion_inplace(nu_ei, ne, Tev, Z_eff)

    freq_electron_neutral_inplace(nu_en, params['electron_neutral_collisions'], nn, Tev)
    freq_electron_classical(nu_c, nu_en, nu_ei, nu_iz, nu_ex, params['landmark'])
    freq_electron_wall(nu_ew_momentum, radial_loss_freq, wall_loss_model, params)

    # Update anomalous transport if t>0
    if t > 0.0:
        anom_model(nu_an, params, config)

    # Now update electrical vars
    update_electrical_vars(params)

    # update thermal conductivity
    conductivity_model(kappa, params)

    # update electron energy
    update_electron_energy(params, wall_loss_model, source_energy, params["dt"][0])


def freq_electron_wall(νew_momentum, radial_loss_freq, wall_loss_model, params):
    grid = params["grid"]
    L_ch = params["thruster"].geometry.channel_length
    transition_length = params["transition_length"]

    for i in range(len(radial_loss_freq)):
        freq_val = freq_electron_wall_impl(wall_loss_model, params, i)
        radial_loss_freq[i] = freq_val
        # now multiply by linear_transition
        zc = grid.cell_centers[i]
        radial_loss_freq[i] = freq_val
        νew_momentum[i] = freq_val * linear_transition(zc, L_ch, transition_length, 1.0, 0.0)


def update_electrical_vars(params):
    cache = params["cache"]
    anom_smoothing_iters = params["anom_smoothing_iters"]
    landmark = params["landmark"]
    grid = params["grid"]


    νan = cache["νan"]
    cell_cache_1 = cache["cell_cache_1"]
    νe = cache["νe"]
    νc = cache["νc"]
    μ = cache["μ"]
    B = cache["B"]
    νew_momentum = cache["νew_momentum"]
    anom_multiplier = cache["anom_multiplier"]
    Vs = cache["Vs"]
    ue = cache["ue"]
    ji = cache["ji"]
    channel_area = cache["channel_area"]
    ne_arr = cache["ne"]
    Id = cache["Id"]
    K = cache["K"]
    pe = cache["pe"]
    grad_pe = cache["∇pe"]
    phi = cache["ϕ"]
    grad_phi = cache["∇ϕ"]

    # smoothing
    if anom_smoothing_iters > 0:
        smooth(νan, cell_cache_1, anom_smoothing_iters)

    # multiply by anom_multiplier
    for i in range(len(νan)):
        νan[i] *= anom_multiplier[0]
        # total collision freq
        νe[i] = νc[i] + νan[i] + νew_momentum[i]
        μ[i] = electron_mobility(νe[i], B[i])

    # compute anode sheath potential
    Vs[0] = anode_sheath_potential(params)

    # compute discharge current
    V_L = params["discharge_voltage"] + Vs[0]
    V_R = params["cathode_coupling_voltage"]
    apply_drag = (not landmark) and (params["iteration"][0] > 5)

    Id[0] = integrate_discharge_current(grid, cache, V_L, V_R, params["neutral_velocity"], apply_drag)

    # update anom_multiplier
    from math import log, exp
    new_log_val = apply_controller(params["simulation"].current_control, Id[0], log(anom_multiplier[0]),
                                   params["dt"][0])
    anom_multiplier[0] = exp(new_log_val)

    # compute electron velocity, kinetic energy
    for i in range(len(ue)):
        # ue[i] = (ji[i] - Id/A)/ e / ne[i]
        ue[i] = (ji[i] - Id[0] / channel_area[i]) / (e * ne_arr[i])

    electron_kinetic_energy(cache["K"], νe, B, ue)

    # compute pressure gradient
    compute_pressure_gradient(grad_pe, pe, grid.cell_centers)

    # compute electric field
    compute_electric_field(grad_phi, cache, params["neutral_velocity"], apply_drag)

    # integrate to find potential
    cumtrapz_inplace(phi, grid.cell_centers, grad_phi, params["discharge_voltage"] + Vs[0])


def integrate_discharge_current(grid, cache, V_L, V_R, un, apply_drag):
    grad_pe = cache["∇pe"]
    μ_arr = cache["μ"]
    ne_arr = cache["ne"]
    ji_arr = cache["ji"]
    channel_area = cache["channel_area"]
    ncells = len(grid.cell_centers)

    if apply_drag:
        νei = cache["νei"]
        νen = cache["νen"]
        νan = cache["νan"]
        ui = cache["ui"]  # shape e.g. (ncharge, ncells)?

    int1 = 0.0
    int2 = 0.0
    for i in range(ncells - 1):
        dz = grid.dz_edge[i]


        int1_1 = (ji_arr[i] / (e * μ_arr[i]) + grad_pe[i]) / ne_arr[i]
        int1_2 = (ji_arr[i + 1] / (e * μ_arr[i + 1]) + grad_pe[i + 1]) / ne_arr[i + 1]

        if apply_drag:

            ion_drag_1 = ui[0, i] * (νei[i] + νan[i]) * me / e  # adjust index if you have 2D
            ion_drag_2 = ui[0, i + 1] * (νei[i + 1] + νan[i + 1]) * me / e
            neutral_drag_1 = un * νen[i] * me / e
            neutral_drag_2 = un * νen[i + 1] * me / e
            int1_1 -= (ion_drag_1 + neutral_drag_1)
            int1_2 -= (ion_drag_2 + neutral_drag_2)

        int1 += 0.5 * dz * (int1_1 + int1_2)

        int2_1 = 1.0 / (e * ne_arr[i] * μ_arr[i] * channel_area[i])
        int2_2 = 1.0 / (e * ne_arr[i + 1] * μ_arr[i + 1] * channel_area[i + 1])
        int2 += 0.5 * dz * (int2_1 + int2_2)

    deltaV = V_L - V_R
    I = (deltaV + int1) / int2
    return I


def electron_kinetic_energy(K_arr, νe_arr, B_arr, ue_arr):
    for i in range(len(K_arr)):
        if νe_arr[i] == 0.0:
            K_arr[i] = 0.0
        else:
            Omega = (e * B_arr[i]) / (me * νe_arr[i])
            factor = 1.0 + Omega * Omega
            K_arr[i] = 0.5 * me * factor * (ue_arr[i] * ue_arr[i]) / e


def compute_pressure_gradient(grad_pe, pe_arr, z_cell):
    n = len(z_cell)
    if n < 3:
        return

    # forward difference
    grad_pe[0] = forward_difference(pe_arr[0], pe_arr[1], pe_arr[2], z_cell[0], z_cell[1], z_cell[2])

    # center
    for j in range(1, n - 1):
        grad_pe[j] = central_difference(pe_arr[j - 1], pe_arr[j], pe_arr[j + 1], z_cell[j - 1], z_cell[j],
                                        z_cell[j + 1])

    # backward difference
    grad_pe[n - 1] = backward_difference(
        pe_arr[n - 3], pe_arr[n - 2], pe_arr[n - 1],
        z_cell[n - 3], z_cell[n - 2], z_cell[n - 1]
    )
