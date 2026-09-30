from ..physics.thermodynamics import velocity, sound_speed
from ..simulation.boundaryconditions import right_boundary_state, left_boundary_state
from ..utilities.utility_functions import right_edge, left_edge


def reconstruct(u_left, u_mid, u_right, limiter_fn):
    denom = (u_mid - u_left)
    if denom == 0.0:
        r = 0.0
    else:
        r = (u_right - u_mid) / denom

    phi = limiter_fn(r)
    delta_u = 0.25 * phi * (u_right - u_left)
    left_state = u_mid - delta_u
    right_state = u_mid + delta_u
    return left_state, right_state


def compute_edge_states(UL, UR, U, params, limiter, do_reconstruct,
                        apply_boundary_conditions=False):


    nvars, ncells = U.shape

    if do_reconstruct:
        for j in range(nvars):

            is_vel = params["is_velocity_index"][j]  # or something similar
            # print(f'[compute_edge_states] is_val {j}',is_vel)
            if is_vel:
                for i in range(1, ncells - 1):
                    # print(f'[compute_edge_states] j-1',j-1)

                    u_left = U[j, i - 1] / U[j - 1, i - 1] if U[j - 1, i - 1] != 0.0 else 0.0
                    u_mid = U[j, i] / U[j - 1, i] if U[j - 1, i] != 0.0 else 0.0
                    u_right = U[j, i + 1] / U[j - 1, i + 1] if U[j - 1, i + 1] != 0.0 else 0.0

                    uR, uL = reconstruct(u_left, u_mid, u_right, limiter)

                    left_edge_idx = left_edge(i)  # or i-1 or something, see your code's definition
                    right_edge_idx = right_edge(i)

                    rhoL = UL[j - 1, right_edge_idx]
                    rhoR = UR[j - 1, left_edge_idx]
                    UL[j, right_edge_idx] = uL * rhoL
                    UR[j, left_edge_idx] = uR * rhoR
            else:
                for i in range(1, ncells - 1):
                    u_left = U[j, i - 1]
                    u_mid = U[j, i]
                    u_right = U[j, i + 1]

                    uR, uL = reconstruct(u_left, u_mid, u_right, limiter)

                    left_edge_idx = left_edge(i)
                    right_edge_idx = right_edge(i)

                    UR[j, left_edge_idx] = uR
                    UL[j, right_edge_idx] = uL
    else:
        for j in range(nvars):
            for i in range(1, ncells - 1):
                left_edge_idx = left_edge(i)
                right_edge_idx = right_edge(i)
                UL[j, right_edge_idx] = U[j, i]
                UR[j, left_edge_idx] = U[j, i]

    # boundary conditions
    if apply_boundary_conditions:

        left_boundary_state(UL[:, 0], U, params)
        right_boundary_state(UR[:, -1], U, params)
    else:

        UL[:, 0] = U[:, 0]
        UR[:, -1] = U[:, -1]


def compute_wave_speeds(lmbda_global, dt_u, UL, UR, U, grid, fluids, index, ncharge):
    # print("[compute_wave_speeds] U", U)

    edges = grid.edges  # or grid.edges if grid is an object
    # print('[compute_wave_speeds]U:', U)

    for i, edge_val in enumerate(edges):
        neutral_fluid = fluids[0]



        # U_neutrals = (U[index["ρn"], i])
        # Cyrus was here
        # U_neutrals = (U[0, i])
        new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
        U_neutrals = (U[new_range, i])



        # print('U_neutrals',U_neutrals)
        u = velocity(U_neutrals, neutral_fluid)
        lmbda_global[0] = abs(u)
        # print('[compute_wave_speeds] ncharge:', ncharge)



        for Z in range(1, ncharge+1):
            fluid_ind = Z  # or Z+1 if you do 1-based
            fluid = fluids[fluid_ind]
            # print('[compute_wave_speeds] index["ρi"] ', index["ρi"])
            # print('[compute_wave_speeds] index["ρiui"] ', index["ρiui"])



            UL_ions = (UL[index["ρi"][Z]-1, i], UL[index["ρiui"][Z]-1, i])
            UR_ions = (UR[index["ρi"][Z]-1, i], UR[index["ρiui"][Z]-1, i])
            uL = velocity(UL_ions, fluid)
            uR = velocity(UR_ions, fluid)
            aL = sound_speed(UL_ions, fluid)
            aR = sound_speed(UR_ions, fluid)

            s_max = max(
                abs(uL + aL), abs(uL - aL),
                abs(uR + aR), abs(uR - aR)
            )
            if s_max != 0.0:
                dt_max = grid.dz_edge[i] / s_max
            else:
                dt_max = 1e9
            dt_u[i] = dt_max
            lmbda_global[fluid_ind] = max(s_max, lmbda_global[fluid_ind])


def compute_fluxes(F, UL, UR, flux_function, lmbda_global, grid, fluids, index, ncharge):
    edges = grid.edges
    for i, edge_val in enumerate(edges):

        # left_state_n = (UL[index["ρn"], i],)
        # right_state_n = (UR[index["ρn"], i],)

        # Cyrus was here
        # left_state_n = (UL[0, i],)
        # right_state_n = (UR[0, i],)
        new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
        left_state_n = (UL[new_range, i],)
        right_state_n = (UR[new_range, i],)

        flux_n = flux_function(left_state_n, right_state_n, fluids[0], lmbda_global[0])


        # F[index["ρn"], i] = flux_n[0]


        # Cyrus was here
        new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
        F[new_range, i] = flux_n[0]

        for Z in range(1, ncharge+1):
            fluid_ind = Z
            left_state_i = (UL[index["ρi"][Z]-1, i], UL[index["ρiui"][Z]-1, i])
            right_state_i = (UR[index["ρi"][Z]-1, i], UR[index["ρiui"][Z]-1, i])
            flux_mass, flux_momentum = flux_function(
                left_state_i, right_state_i, fluids[fluid_ind], lmbda_global[fluid_ind]
            )
            F[index["ρi"][Z]-1, i] = flux_mass
            F[index["ρiui"][Z]-1, i] = flux_momentum


def compute_fluxes_overall(F, UL, UR, U, params, scheme, apply_boundary_conditions=False):
    # print("[compute_fluxes_overall] U", U)

    # print('[compute_fluxes_overall] params["index"]: ', params["index"])
    index = params["index"]
    fluids = params["fluids"]
    grid = params["grid"]
    cache = params["cache"]
    ncharge = params["ncharge"]

    lmbda_global = cache["λ_global"]
    dt_u = cache["dt_u"]

    compute_edge_states(UL, UR, U, params, scheme.limiter, scheme.reconstruct,
                        apply_boundary_conditions)
    # print("[compute_fluxes_overall] U", U)


    compute_wave_speeds(lmbda_global, dt_u, UL, UR, U, grid, fluids, index, ncharge)
    compute_fluxes(F, UL, UR, scheme.flux_function, lmbda_global, grid, fluids, index, ncharge)

    return F
