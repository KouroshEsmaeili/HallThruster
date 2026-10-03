import math
from enum import IntEnum

from .physicalconstants import kB


class ConservationLawType(IntEnum):
    _ContinuityOnly = 1
    _IsothermalEuler = 2
    _EulerEquations = 3

class Fluid:
    def __init__(self, species, law_type, nvars, u=None, T=None, a=None):
        self.species = species               # e.g. Xenon(0)
        self.type = law_type                 # e.g. ConservationLawType._ContinuityOnly
        self.nvars = nvars                   # e.g. 1, 2, or 3
        self.u = u                           # float or None
        self.T = T                           # float or None
        self.a = a                           # float or None

    def __repr__(self):
        return (f"<Fluid species={self.species}, type={self.type.name}, "
                f"nvars={self.nvars}, u={self.u}, T={self.T}, a={self.a}>")

def nvars(fluid):
    return fluid.nvars

def ContinuityOnly(species, u, T):
    gamma = species.element.gamma
    mass = species.element.m
    a_val = math.sqrt(gamma * kB * T / mass)
    return Fluid(species, ConservationLawType._ContinuityOnly, 1,
                 u=float(u), T=float(T), a=float(a_val))

def IsothermalEuler(species, T):
    gamma = species.element.gamma
    mass = species.element.m
    a_val = math.sqrt(gamma * kB * T / mass)
    return Fluid(species, ConservationLawType._IsothermalEuler, 2,
                 u=None, T=float(T), a=float(a_val))

def EulerEquations(species):
    return Fluid(species, ConservationLawType._EulerEquations, 3,
                 u=None, T=None, a=None)

#
# The "dispatcher" function that in Julia was:
#   function Fluid(s; u=nothing, T=nothing)
#       if isnothing(u) && isnothing(T)
#           return EulerEquations(s)
#       elseif isnothing(u)
#           return Fluid(s, T)
#       else
#           return Fluid(s, u, T)
#       end
#   end
#

def fluid_factory(species, u=None, T=None):
    if u is None and T is None:
        return EulerEquations(species)
    elif u is None:
        # if T is not None
        return IsothermalEuler(species, T)
    else:
        return ContinuityOnly(species, u, T)

def ranges(fluids):
    fluid_ranges = []
    start_ind = 1
    for f in fluids:
        nf = nvars(f)
        last_ind = (start_ind - 1) + nf

        fluid_ranges.append(range(start_ind, last_ind+1))
        start_ind = last_ind + 1

    return fluid_ranges
