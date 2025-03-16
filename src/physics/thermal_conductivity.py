import math
from ..utilities.interpolation import LinearInterpolation
from ..collisions.collision_frequencies import electron_mobility

class ThermalConductivityModel:

    def __call__(self, kappa, params):
        raise NotImplementedError("ThermalConductivityModel subclasses must override __call__.")

LOOKUP_ZS = [1, 2, 3, 4, 5]
LOOKUP_CONDUCTIVITY_COEFFS = [4.66, 4.0, 3.7, 3.6, 3.2]
ELECTRON_CONDUCTIVITY_LOOKUP = LinearInterpolation(LOOKUP_ZS, LOOKUP_CONDUCTIVITY_COEFFS)


class Braginskii(ThermalConductivityModel):

    def __call__(self, kappa, params):
        c = params["cache"]
        nu_c         = c["νc"]           # classical collision freq
        nu_ew        = c["νew_momentum"] # electron-wall momentum freq
        nu_an        = c["νan"]          # anomalous freq
        B_array      = c["B"]
        ne           = c["ne"]
        Tev          = c["Tev"]
        Z_eff        = c["Z_eff"]

        for i in range(len(kappa)):
            # get coefficient from 'charge states'
            k_coef = ELECTRON_CONDUCTIVITY_LOOKUP(Z_eff[i])
            nu = nu_c[i] + nu_ew[i] + nu_an[i]
            mu_val = electron_mobility(nu, B_array[i])
            kappa[i] = k_coef*mu_val* ne[i]* Tev[i]
        return kappa

class Mitchner(ThermalConductivityModel):

    def __call__(self, kappa, params):
        c = params["cache"]
        nu_c   = c["νc"]
        nu_ew  = c["νew_momentum"]
        nu_ei  = c["νei"]
        nu_an  = c["νan"]
        B_arr  = c["B"]
        ne     = c["ne"]
        Tev    = c["Tev"]

        for i in range(len(kappa)):
            nu = nu_c[i]+ nu_ew[i]+ nu_an[i]
            mu_val = electron_mobility(nu, B_arr[i])
            denom = 1.0
            if nu != 0.0:
                denom = 1.0 + nu_ei[i]/(math.sqrt(2.0)*nu)
            factor = 2.4/denom
            kappa[i] = factor*mu_val* ne[i]* Tev[i]
        return kappa

class LANDMARK_conductivity(ThermalConductivityModel):

    def __call__(self, kappa, params):
        c = params["cache"]
        mu_arr = c["μ"]
        ne     = c["ne"]
        Tev    = c["Tev"]

        for i in range(len(kappa)):
            kappa[i] = (10.0/9.0)* mu_arr[i]* 1.5* ne[i]* Tev[i]
        return kappa


thermal_conductivity_models = {
    "Braginskii": Braginskii,
    "Mitchner": Mitchner,
    "LANDMARK_conductivity": LANDMARK_conductivity
}
