

import math


from ..physics.physicalconstants import e, me, kB

def freq_electron_neutral(collisions, nn, Tev):
    nu_en = 0.0
    for c in collisions:
        from ..collisions.reactions import rate_coeff  # if needed
        nu_en += rate_coeff(c, 1.5 * Tev) * nn
    return nu_en

def freq_electron_neutral_inplace(nu_en, collisions, nn, Tev):
    for i in range(len(nu_en)):
        nu_en[i] = freq_electron_neutral(collisions, nn[i], Tev[i])

def coulomb_logarithm(ne, Tev, Z=1.0):
    if Tev < 10 * (Z**2):
        ln_Λ = 23 - 0.5 * math.log(1e-6 * ne * (Z**2) / (Tev**3))
    else:
        ln_Λ = 24 - 0.5 * math.log(1e-6 * ne / (Tev**2))
    return ln_Λ

def freq_electron_ion(ne, Tev, Z):
    logC = coulomb_logarithm(ne, Tev, Z)
    return 2.9e-12 * (Z**2) * ne * logC / math.sqrt(Tev**3)

def freq_electron_ion_inplace(nu_ei, ne, Tev, Z):
    for i in range(len(nu_ei)):
        nu_ei[i] = freq_electron_ion(ne[i], Tev[i], Z[i])

def freq_electron_classical(nu_c, nu_en, nu_ei, nu_iz, nu_ex, landmark):
    for i in range(len(nu_c)):
        nu_c[i] = nu_en[i] + nu_ei[i]
    if landmark:
        return
    for i in range(len(nu_c)):
        nu_c[i] += nu_iz[i] + nu_ex[i]

def electron_mobility(nu_e, B):
    if nu_e == 0.0:
        return 0.0
    omega = e * B / (me * nu_e)
    return e / (me * nu_e * (1 + omega**2))

def electron_sound_speed(Tev):
    import math
    return math.sqrt(8 * e * Tev / (math.pi * me))
