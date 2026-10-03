import math
from ..physics.physicalconstants import e, me


def compute_electric_field(grad_phi, cache, un, apply_drag):
    # Unpack necessary arrays from cache.
    ji = cache["ji"]  # Ion current density array

    Id = cache["Id"]  # Discharge current (assumed to be a one-element list)

    ne = cache["ne"]  # Electron number density
    # print('[compute_electric_field] cache["ne"]:', cache["ne"])

    mu = cache["μ"]  # Electron mobility (μ)

    grad_pe = cache["∇pe"]  # Electron pressure gradient
    channel_area = cache["channel_area"]

    ui = cache["ui"]  # Ion velocity array (2D array, use first row)
    nu_ei = cache["νei"]
    nu_en = cache["νen"]
    nu_an = cache["νan"]
    # print('[compute_electric_field] cache["∇ϕ"]:', cache["∇ϕ"])

    for i in range(len(grad_phi)):
        E = ((Id[0] / channel_area[i] - ji[i]) / e / mu[i] - grad_pe[i]) / ne[i]
        # print('[compute_electric_field] apply_drag:', apply_drag)

        if apply_drag:
            # print('[compute_electric_field] apply_drag:',apply_drag)
            ion_drag = ui[0, i] * (nu_ei[i] + nu_an[i]) * me / e
            neutral_drag = un * nu_en[i] * me / e
            E += ion_drag + neutral_drag
        grad_phi[i] = -E
    return grad_phi


def anode_sheath_potential(params):
    # print('[anode_sheath_potential] anode_sheath_potential is started')
    # print('[anode_sheath_potential] params["landmark"]', params['landmark'])
    if params['landmark']:
        return 0.0
    anode_bc = params["anode_bc"]
    cache = params["cache"]
    ne = cache["ne"]
    ji = cache["ji"]
    channel_area = cache["channel_area"]
    Tev = cache["Tev"]
    Id = cache["Id"]
    # print('[anode_sheath_potential] anode_bc',anode_bc)

    if anode_bc == "sheath":

        # print('[anode_sheath_potential] anode_bc == "sheath" is started')
        # Use first element of Tev, ne, etc. (0-indexed in Python)
        ce = math.sqrt(8 * e * Tev[0] / (math.pi * me))
        je_sheath = e * ne[0] * ce / 4
        jd = Id[0] / channel_area[0]
        ji_sheath_edge = ji[0]
        je_sheath_edge = jd - ji_sheath_edge
        current_ratio = je_sheath_edge / je_sheath
        # print('[anode_sheath_potential] current_ratio:',current_ratio)
        # print('[anode_sheath_potential] je_sheath:', je_sheath)
        # print('[anode_sheath_potential] ji_sheath_edge', ji_sheath_edge)
        # print('[anode_sheath_potential] current_ratio:', current_ratio)
        if current_ratio <= 0.0:
            Vs = 0.0
        else:
            # print('[anode_sheath_potential] current_ratio > 0.0')
            Vs = -Tev[0] * math.log(min(1.0, je_sheath_edge / je_sheath))
            # print('[anode_sheath_potential] Vs:', Vs)
    else:
        Vs = 0.0
    return Vs
