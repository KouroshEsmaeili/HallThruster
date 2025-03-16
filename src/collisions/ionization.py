
import os
from .reactions import Reaction, rate_coeff_filename, load_rate_coeffs
from ..physics.gas import Xenon  # or wherever Xenon is defined
from ..utilities.interpolation import LinearInterpolation
from ..HallThruster import REACTION_FOLDER, LANDMARK_RATES_FILE


class IonizationReaction(Reaction):
    def __init__(self, energy, reactant, product, rate_coeffs):
        super().__init__()
        self.energy = energy
        self.reactant = reactant
        self.product = product
        self.rate_coeffs = rate_coeffs

    def __str__(self):
        electron_input = "e-"

        delta_z = self.product.Z - self.reactant.Z + 1
        electron_output = f"{delta_z}e-"
        react_str = str(self.reactant)
        prod_str = str(self.product)

        rxn_str = f"{electron_input} + {react_str} -> {electron_output} + {prod_str}"
        # print('[IonizationReaction]rxn_str:', rxn_str)
        return rxn_str


def LandmarkIonizationLookup():
    return "Landmark"

def IonizationLookup():
    return "Lookup"

def load_ionization_reactions(model, species, directories=None, **kwargs):
    if directories is None:
        directories = []



    if model == "Landmark":
        if len(species) > 2 or not (species[0].Z == 0 and species[1].Z == 1):
            raise ValueError(f"Unsupported species {species} for LANDMARK ionization lookup.")
        import pandas as pd
        rates_pd = pd.read_csv(LANDMARK_RATES_FILE)
        if rates_pd.empty:
            print('[load_ionization_reactions]: failed to load rates file (rates_pd)')
        ϵ_array = rates_pd['Energy(ev)']
        k_array = rates_pd['rate coefficient (m3/s)']

        itp = LinearInterpolation(ϵ_array, k_array)
        xs = range(256)
        rate_coeffs = [itp(x) for x in xs]
        return [IonizationReaction(12.12, species[0], species[1], rate_coeffs)]

    elif model == "Lookup":
        species_sorted = sorted(species, key=lambda s: s.Z)
        reactions = []
        folders = directories + [REACTION_FOLDER]

        for i in range(len(species_sorted)):
            for j in range(i+1, len(species_sorted)):
                reactant = species_sorted[i]
                product = species_sorted[j]
                found = False
                for folder in folders:
                    print('folder', folder)
                    filename = rate_coeff_filename(reactant, product, "ionization", folder)
                    # print('[load_ionization_reactions]filename',filename)
                    # filename = '/' + filename
                    # print('[load_ionization_reactions]filename',filename)

                    if os.path.exists(filename):
                        energy, rate_coeff = load_rate_coeffs(reactant, product, "ionization", folder)
                        rxn = IonizationReaction(energy, reactant, product, rate_coeff)
                        reactions.append(rxn)
                        found = True
                        break
                if not found:
                    raise ValueError(f"No reactions including {reactant} and {product} "
                                     f"in provided directories {folders}.")
        return reactions

    elif model == "OVS":
        return [IonizationReaction(12.12, Xenon(0), Xenon(1), [0.0, 0.0, 0.0])]

    else:
        raise ValueError(f"Invalid ionization model {model}. Select 'Landmark' or 'Lookup' or 'OVS'.")
