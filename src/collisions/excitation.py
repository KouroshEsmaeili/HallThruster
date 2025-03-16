import os
import math

from .reactions import Reaction, rate_coeff_filename, load_rate_coeffs
from ..physics.gas import Xenon
from ..utilities.interpolation import LinearInterpolation
from ..HallThruster import REACTION_FOLDER,LANDMARK_RATES_FILE



class ExcitationReaction(Reaction):
    def __init__(self, energy, reactant, rate_coeffs):
        super().__init__()
        self.energy = energy
        self.reactant = reactant
        self.rate_coeffs = rate_coeffs

    def __str__(self):
        electron_input = "e-"
        electron_output = "e-"
        react_str = str(self.reactant)
        product_str = str(self.reactant) + "*"
        return f"{electron_input} + {react_str} -> {electron_output} + {product_str}"

def LandmarkExcitationLookup():
    return "Landmark"

def ExcitationLookup():
    return "Lookup"

def ovs_rate_coeff_ex(energy):
    return 1e-12 * math.exp(-8.32 / energy)

def load_excitation_reactions(model, species, directories=None, **kwargs):
    if directories is None:
        directories = []

    reactions = []

    if model == "None":
        return []

    elif model == "Landmark":
        import pandas as pd
        rates_pd = pd.read_csv(LANDMARK_RATES_FILE)

        ϵ_array = rates_pd['Energy(ev)']
        k_iz = rates_pd['rate coefficient (m3/s)']
        k_loss = rates_pd['energy loss (eV/(m3 s))']

        ionization_energy = 12.12
        excitation_energy = 8.32
        k_excitation = []
        for i in range(len(ϵ_array)):
            val = (k_loss[i] - ionization_energy*k_iz[i]) / excitation_energy
            k_excitation.append(val)

        itp = LinearInterpolation(ϵ_array, k_excitation)
        xs = list(range(256))
        rate_coeffs = [itp(x) for x in xs]

        return [ExcitationReaction(excitation_energy, Xenon(0), rate_coeffs)]

    elif model == "Lookup":
        species_sorted = sorted(species, key=lambda sp: sp.Z)
        folders = directories + [REACTION_FOLDER]
        product = None

        for reactant in species_sorted:
            for folder in folders:
                filename = rate_coeff_filename(reactant, product, "excitation", folder)
                if os.path.exists(filename):
                    energy, rate_coeffs = load_rate_coeffs(
                        reactant, product, "excitation", folder
                    )
                    rxn = ExcitationReaction(energy, reactant, rate_coeffs)
                    reactions.append(rxn)
                    break
        return reactions

    elif model == "OVS":

        Es = range(256)
        ks = [ovs_rate_coeff_ex(float(e)) for e in Es]
        return [ExcitationReaction(8.32, Xenon(0), ks)]

    else:
        raise ValueError(f"Invalid excitation model {model}. "
                         "Choose 'None', 'Landmark', 'Lookup', or 'OVS'.")

    return reactions

