
import os
from .reactions import Reaction, rate_coeff_filename, load_rate_coeffs
from ..HallThruster import REACTION_FOLDER
from ..physics.gas import Xenon


class ElasticCollision(Reaction):
    def __init__(self, species, rate_coeffs):
        super().__init__()
        self.species = species
        self.rate_coeffs = rate_coeffs

def NoElectronNeutral():
    return "None"

def ElectronNeutralLookup():
    return "Lookup"

def LandmarkElectronNeutral():
    return "Landmark"

def load_elastic_collisions(model, species, directories=None, **kwargs):
    if directories is None:
        directories = []

    if model == "None":
        return []

    elif model == "Landmark":
        return [ElasticCollision(Xenon(0), [2.5e-13, 2.5e-13, 2.5e-13])]

    elif model == "Lookup":
        species_sorted = sorted(species, key=lambda x: x.Z)
        reactions = []
        folders = directories + [REACTION_FOLDER]
        product = None
        collision_type = "elastic"

        for reactant in species_sorted:
            if reactant.Z > 0:
                break
            for folder in folders:
                filename = rate_coeff_filename(reactant, product, collision_type, folder)
                if os.path.exists(filename):
                    _, rate_coeff = load_rate_coeffs(reactant, product, collision_type, folder)
                    reaction = ElasticCollision(reactant, rate_coeff)
                    reactions.append(reaction)
                    break


        return reactions

    else:

        raise ValueError(f"Invalid elastic collision model {model}. Choose 'None', 'Landmark', or 'Lookup'.")
