import os
import math


from ..HallThruster import REACTION_FOLDER
from ..utilities.interpolation import LinearInterpolation


class Reaction:
    pass


def rate_coeff_filename(reactant, product, reaction_type, folder=REACTION_FOLDER):
    if product is None:
        fname = os.path.join(folder, f"{reaction_type}_{repr(reactant)}.dat")
    else:
        fname = os.path.join(folder, f"{reaction_type}_{repr(reactant)}_{repr(product)}.dat")
    return '/' + fname


def load_rate_coeffs(reactant, product, reaction_type, folder=REACTION_FOLDER):
    filename = rate_coeff_filename(reactant, product, reaction_type, folder)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")

    e_data = []
    k_data = []
    ionization_energy = 0.0

    with open(filename, 'r') as f:
        line1 = f.readline().strip()
        if "Ionization energy" in line1:
            parts = line1.split(':', 1)
            ionization_energy = float(parts[1].strip())
        else:
            raise ValueError(f"Unexpected first line (no Ionization energy): '{line1}'")

        # 2) Read the second line => Header: "Energy (eV)\tRate coefficient (m/s)" or similar
        header_line = f.readline().strip()
        # We won't parse the header further, just skip it. Optionally, we can verify it's correct.

        for row in f:
            row = row.strip()
            if not row:
                continue
            parts = row.split()
            if len(parts) < 2:
                continue
            E_val = float(parts[0])
            k_val = float(parts[1])
            e_data.append(E_val)
            k_data.append(k_val)


    if len(e_data) < 2:
        raise ValueError(f"Insufficient data rows in {filename} to interpolate. Found {len(e_data)} rows.")

    interp_fn = LinearInterpolation(e_data, k_data)

    rate_coeffs = []
    for e_int in range(256):
        k_val = interp_fn(float(e_int))
        rate_coeffs.append(k_val)

    return ionization_energy, rate_coeffs


def lerp(a, b, t):

    return (1.0 - t) * a + t * b


def rate_coeff(rxn, energy):

    if not hasattr(rxn, "rate_coeffs"):
        raise AttributeError("Reaction object has no 'rate_coeffs' attribute.")


    # print('[rate_coeff] energy', energy)
    if math.isfinite(energy):
        ind = int(energy)
    else:
        ind = 0

    N = len(rxn.rate_coeffs) - 2
    if ind > N:
        ind = N
    elif ind < 0:
        ind = 0

    r1 = rxn.rate_coeffs[ind]
    r2 = rxn.rate_coeffs[ind + 1]  # careful with zero-based indexing

    frac = energy - ind

    return lerp(r1, r2, frac)


def _indices(symbol, reactions, species_range_dict):

    import numpy as np
    indices = np.zeros(len(reactions), dtype=int)

    for i, reaction in enumerate(reactions):
        spc_obj = getattr(reaction, symbol)
        spc_symbol = spc_obj.symbol  # e.g. "Xe+", "Xe0", etc.
        rng = species_range_dict[spc_symbol]
        print('[_indices] rng', rng)
        indices[i] = rng[0]  # or rng[1] if you stored 1-based
        # print('[_indices] indices', indices)
    return indices


def reactant_indices(reactions, species_range_dict):
    return _indices("reactant", reactions, species_range_dict)


def product_indices(reactions, species_range_dict):
    return _indices("product", reactions, species_range_dict)

