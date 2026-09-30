
from .fluid import Fluid


def gamma(fluid: Fluid):
    return fluid.species.element.gamma


def m(fluid: Fluid):
    return fluid.species.element.m


def R(fluid: Fluid):
    return fluid.species.element.R


def cp(fluid: Fluid):
    return fluid.species.element.cp


def cv(fluid: Fluid):
    return fluid.species.element.cv


def number_density(U, f: Fluid):

    return density(U, f) / m(f)


def density(U, f: Fluid):
    return U[0]


def velocity(U, f: Fluid):
    if len(U) == 1:
        # print('[velocity] U len(U) == 1', U)
        # print('[velocity] U[0] len(U) == 1', U[0])

        return f.u if hasattr(f, 'u') else 0.0

    elif len(U) >= 2:
        # print('[velocity] U len(U) >= 2', U)
        # print('[velocity] U[0] len(U) >= 2', U[0])
        # print('[velocity] U[1] len(U) >= 2', U[1])
        # print('[velocity] U[1] / U[0]', U[1] / U[0])
        return U[1] / U[0]

    # print('[velocity]', "return 0.0")
    return 0.0


def temperature(U, f: Fluid):
    g = gamma(f)
    if len(U) == 1:
        return f.T if hasattr(f, 'T') else 0.0
    elif len(U) == 2:
        return f.T if hasattr(f, 'T') else 0.0
    elif len(U) == 3:
        # T = (gamma - 1)*(U[2] - 0.5 * U[1]^2/U[0])/(U[0]*R(f))
        return (g - 1) * (U[2] - 0.5 * (U[1] ** 2) / U[0]) / (U[0] * R(f))
    else:
        print('[temperature]', "return 0.0")
        return 0.0


def pressure(U, f):
    g = gamma(f)
    if len(U) == 1:
        return U[0] * R(f) * f.T if hasattr(f, 'T') else 0.0
    elif len(U) == 2:
        return U[0] * R(f) * f.T if hasattr(f, 'T') else 0.0
    elif len(U) == 3:
        return (g - 1) * (U[2] - 0.5 * (U[1] ** 2) / U[0])
    print('[pressure]', "return 0.0")
    return 0.0


def sound_speed(U, f):
    import math
    g = gamma(f)
    if len(U) == 1:
        return f.a if hasattr(f, 'a') else 0.0
    elif len(U) == 2:
        return f.a if hasattr(f, 'a') else 0.0
    elif len(U) == 3:
        T = temperature(U, f)
        return math.sqrt(g * R(f) * T)
    return 0.0
