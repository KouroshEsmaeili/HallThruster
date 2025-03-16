import math
from ..physics.physicalconstants import  e,kB
from ..utilities.utility_functions import myerf

def left_boundary_state(bc_state, U, params):
    index = params["index"]
    ncharge = params["ncharge"]
    mi = params["mi"]
    Ti = params["ion_temperature_K"]
    un = params["neutral_velocity"]
    mdot_a = params["anode_mass_flow_rate"]
    nn_B = params["background_neutral_density"]
    un_B = params["background_neutral_velocity"]
    neutral_ingestion_multiplier = params["neutral_ingestion_multiplier"]
    anode_bc = params["anode_bc"]
    cache = params["cache"]

    ingestion_density = nn_B * un_B / un * neutral_ingestion_multiplier

    left_boundary_state_inner(bc_state, U, index, ncharge, cache, mi, Ti, un,
                              ingestion_density, mdot_a, anode_bc)
    return


def left_boundary_state_inner(bc_state, U, index, ncharge, cache, mi, Ti, un, ingestion_density, mdot_a, anode_bc):
    if anode_bc == "sheath":
        Vs = cache["Vs"][0]
        if Vs > 0:
            Vs_norm = Vs / cache["Tev"][0]
            # Avoid division by zero:
            if Vs_norm > 0:
                chi = math.exp(-Vs_norm) / math.sqrt(math.pi * Vs_norm) / (1 + myerf(math.sqrt(Vs_norm)))
            else:
                chi = 0.0
            bohm_factor = 1.0 / math.sqrt(1 + chi)
        else:
            bohm_factor = 0.0
    else:
        bohm_factor = 1.0

    # bc_state[index["ρn"]] = mdot_a / cache["channel_area"][0] / un
    # # Add ingestion density.
    # bc_state[index["ρn"]] += ingestion_density

    # Cyrus was here
    new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
    bc_state[new_range] = mdot_a / cache["channel_area"][0] / un
    bc_state[new_range] += ingestion_density

    for Z in range(1, ncharge + 1):
        interior_density = U[index["ρi"][Z]-1, 1]
        interior_flux = U[index["ρiui"][Z]-1, 1]
        interior_velocity = interior_flux / interior_density if interior_density != 0 else 0.0


        sound_speed = math.sqrt((kB * Ti + Z * e * cache["Tev"][0]) / mi)

        boundary_velocity = -bohm_factor * sound_speed

        if interior_velocity <= -sound_speed:
            boundary_density = interior_density
            boundary_flux = interior_flux
        else:
            J_minus = interior_velocity - sound_speed * math.log(interior_density)
            J_plus = 2 * boundary_velocity - J_minus
            boundary_density = math.exp(0.5 * (J_plus - J_minus) / sound_speed)
            boundary_flux = boundary_velocity * boundary_density

        # Adjust neutral density at boundary.
        # bc_state[index["ρn"]] -= boundary_flux / un

        # Cyrus was here
        new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
        bc_state[new_range] -= boundary_flux / un

        bc_state[index["ρi"][Z]-1] = boundary_density
        bc_state[index["ρiui"][Z]-1] = boundary_flux
    return


def right_boundary_state(bc_state, U, params):
    index = params["index"]
    ncharge = params["ncharge"]

    # For neutrals: use second-to-last column.
    # bc_state[index["ρn"]] = U[index["ρn"], -2]


    # Cyrus was here
    new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
    bc_state[new_range] = U[new_range, -2]

    # For each ion charge state.
    for Z in range(1, ncharge + 1):
        bc_state[index["ρi"][Z]-1] = U[index["ρi"][Z]-1, -2]
        bc_state[index["ρiui"][Z]-1] = U[index["ρiui"][Z]-1, -2]
    return
