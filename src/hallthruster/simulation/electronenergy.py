import math
from ..utilities.utility_functions import right_edge, left_edge
from .sourceterms import source_electron_energy
from ..utilities.linearalgebra import tridiagonal_solve_inplace


def update_electron_energy(params, wall_loss_model, source_energy, dt):
    Te_L = params["Te_L"]
    Te_R = params["Te_R"]
    grid = params["grid"]
    cache = params["cache"]
    min_Te = params["min_Te"]
    implicit_energy = params["implicit_energy"]
    anode_bc = params["anode_bc"]
    landmark = params["landmark"]

    A_eps = cache["Aϵ"]
    b_eps = cache["bϵ"]
    n_eps = cache["nϵ"]
    ue = cache["ue"]
    ne = cache["ne"]
    Tev = cache["Tev"]
    pe = cache["pe"]
    Q = cache["cell_cache_1"]

    source_electron_energy(Q, params, wall_loss_model)

    for i in range(1, grid.num_cells - 1):
        Q[i] += source_energy(params, i)

    energy_boundary_conditions(A_eps, b_eps, Te_L, Te_R, ne, ue, anode_bc)

    # Set up the energy system (fill matrix A_eps and RHS b_eps).
    setup_energy_system(A_eps, b_eps, grid, cache, anode_bc, implicit_energy, dt)

    # Solve the tridiagonal system (update n_eps in place).
    tridiagonal_solve_inplace(n_eps, A_eps, b_eps)

    # Limit electron energy density and update temperature and pressure.
    limit_energy(n_eps, ne, min_Te)
    update_temperature(Tev, n_eps, ne, min_Te)
    update_pressure(pe, n_eps, landmark)
    return


def setup_energy_system(A_eps, b_eps, grid, cache, anode_bc, implicit, dt):
    ne = cache["ne"]
    ue = cache["ue"]
    kappa = cache["κ"]
    ji = cache["ji"]
    channel_area = cache["channel_area"]
    dA_dz = cache["dA_dz"]
    n_eps = cache["nϵ"]
    explicit = 1.0 - implicit
    Q = cache["cell_cache_1"]

    for i in range(1, grid.num_cells - 1):
        neL = ne[i - 1]
        ne0 = ne[i]
        neR = ne[i + 1]

        n_eps_L = n_eps[i - 1]
        n_eps_0 = n_eps[i]
        n_eps_R = n_eps[i + 1]

        ueL = ue[i - 1]
        ue0 = ue[i]
        ueR = ue[i + 1]

        kappa_L = kappa[i - 1]
        kappa_0 = kappa[i]
        kappa_R = kappa[i + 1]

        ΔzL = grid.dz_edge[left_edge(i)]
        ΔzR = grid.dz_edge[right_edge(i)]
        Δz = grid.dz_cell[i]

        # Weighted average of electron velocities.
        ue_avg = 0.25 * (ΔzL * (ueL + ue0) + ΔzR * (ue0 + ueR)) / Δz

        if ue_avg > 0:
            FR_factor_L = 0.0
            FR_factor_C = (5.0 / 3.0) * ue0 + kappa_0 / (ΔzR * ne0)
            FR_factor_R = - kappa_0 / (ΔzR * neR)

            FL_factor_L = (5.0 / 3.0) * ueL + kappa_L / (ΔzL * neL)
            FL_factor_C = - kappa_L / (ΔzL * ne0)
            FL_factor_R = 0.0
        else:
            if i == 1 and anode_bc == "sheath":
                Te0 = (2.0 / 3.0) * n_eps_0 / ne0
                # Assume cache["Id"] is a NumPy array; use its first element.
                jd = cache["Id"][0] / channel_area[0]
                ji_sheath_edge = ji[0]
                je_sheath_edge = jd - ji_sheath_edge
                ne_sheath_edge = ne[0]
                # Use elementary charge (you may want to import a constant instead of math.e).
                e_val = 1.602176634e-19
                ue_sheath_edge = - je_sheath_edge / ne_sheath_edge / e_val
                FL_factor_L = 0.0
                FL_factor_C = (4.0 / 3.0) * ue_sheath_edge * (1 + cache["Vs"][0] / Te0)
                FL_factor_R = 0.0
            elif i == 1:
                FL_factor_L = (5.0 / 3.0) * ueL + kappa_L / (ΔzL * neL)
                FL_factor_C = - kappa_L / (ΔzL * ne0)
                FL_factor_R = 0.0
            else:
                FL_factor_L = kappa_0 / (ΔzL * neL)
                FL_factor_C = (5.0 / 3.0) * ue0 - kappa_0 / (ΔzL * ne0)
                FL_factor_R = 0.0

            FR_factor_L = 0.0
            FR_factor_C = kappa_R / (ΔzR * ne0)
            FR_factor_R = (5.0 / 3.0) * ueR - kappa_R / (ΔzR * neR)

        FL = FL_factor_L * n_eps_L + FL_factor_C * n_eps_0 + FL_factor_R * n_eps_R
        FR = FR_factor_L * n_eps_L + FR_factor_C * n_eps_0 + FR_factor_R * n_eps_R

        A_eps._d[i] = (FR_factor_C - FL_factor_C) / Δz
        A_eps._dl[i - 1] = (FR_factor_L - FL_factor_L) / Δz
        A_eps._du[i] = (FR_factor_R - FL_factor_R) / Δz

        A_eps._d[i] = 1.0 + implicit * dt * A_eps._d[i]
        A_eps._dl[i - 1] = implicit * dt * A_eps._dl[i - 1]
        A_eps._du[i] = implicit * dt * A_eps._du[i]

        F_explicit = (FR - FL) / Δz

        dlnA_dz = dA_dz[i] / channel_area[i]
        flux_term = (5.0 / 3.0) * n_eps_0 * ue0

        b_eps[i] = n_eps[i] + dt * (Q[i] - explicit * F_explicit)
        b_eps[i] -= dt * flux_term * dlnA_dz
    return


def energy_boundary_conditions(A_eps, b_eps, Te_L, Te_R, ne, ue, anode_bc):
    A_eps._d[0] = 1.0
    A_eps._du[0] = 0.0
    A_eps._d[-1] = 1.0
    A_eps._dl[-1] = 0.0

    if anode_bc == "dirichlet" or ue[0] > 0:
        b_eps[0] = 1.5 * Te_L * ne[0]
    else:
        b_eps[0] = 0.0
        A_eps._d[0] = 1.0 / ne[0]
        A_eps._du[0] = -1.0 / ne[1]
    b_eps[-1] = 1.5 * Te_R * ne[-1]
    return


def limit_energy(n_eps, ne, min_Te):
    for i in range(len(n_eps)):
        if (not math.isfinite(n_eps[i])) or (n_eps[i] / ne[i] < 1.5 * min_Te):
            n_eps[i] = 1.5 * min_Te * ne[i]
    return


def update_temperature(Tev, n_eps, ne, min_Te):
    for i in range(len(Tev)):
        Tev[i] = (2.0 / 3.0) * max(min_Te, n_eps[i] / ne[i])

    return


def update_pressure(pe, n_eps, landmark):
    pe_factor = 1.0 if landmark else (2.0 / 3.0)
    for i in range(len(pe)):
        pe[i] = pe_factor * n_eps[i]

    return
