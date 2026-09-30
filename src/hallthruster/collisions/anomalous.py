
from abc import ABC, abstractmethod

from ..physics.physicalconstants import e, me
from ..utilities.interpolation import LinearInterpolation
from ..utilities.smoothing import linear_transition


class AnomalousTransportModel(ABC):

    @abstractmethod
    def __call__(self, nu_an, params, config):
        pass


def anom_models():
    return {
        "NoAnom": NoAnom,
        "Bohm": Bohm,
        "TwoZoneBohm": TwoZoneBohm,
        "MultiLogBohm": MultiLogBohm,
        "GaussianBohm": GaussianBohm,
        "LogisticPressureShift": LogisticPressureShift,
        "SimpleLogisticShift": SimpleLogisticShift
    }


class NoAnom(AnomalousTransportModel):
    def __call__(self, nu_an, params, config):
        for i in range(len(nu_an)):
            nu_an[i] = 0.0
        return nu_an


class Bohm(AnomalousTransportModel):

    def __init__(self, c):
        self.c = c

    def __call__(self, nu_an, params, config):

        cache = params["cache"]
        grid = params["grid"]
        thruster = params["thruster"]

        B_array = cache["B"]
        iteration = params["iteration"][0]

        if iteration > 5:
            return nu_an

        L_ch = thruster.geometry.channel_length
        z_shift = pressure_shift(config.anom_model, config.background_pressure_Torr, L_ch)

        B_interp = LinearInterpolation(grid.cell_centers, B_array)

        for i, zc in enumerate(grid.cell_centers):
            _ = zc - z_shift
            B_val = B_interp(zc)
            omega_ce = e * B_val / me
            nu_an[i] = self.c * omega_ce

        return nu_an


class TwoZoneBohm(AnomalousTransportModel):
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def __call__(self, nu_an, params, config):
        cache = params["cache"]
        grid = params["grid"]
        thruster = params["thruster"]

        iteration = params["iteration"][0]
        if iteration > 5:
            return nu_an

        B_array = cache["B"]
        L_ch = thruster.geometry.channel_length
        L_trans = config.transition_length
        z_shift = pressure_shift(config.anom_model, config.background_pressure_Torr, L_ch)

        B_interp = LinearInterpolation(grid.cell_centers, B_array)

        for i, zc in enumerate(grid.cell_centers):
            z = zc - z_shift
            B_val = B_interp(zc)
            omega_ce = e * B_val / me
            c_val = linear_transition(z, L_ch, L_trans, self.c1, self.c2)
            nu_an[i] = c_val * omega_ce

        return nu_an


class MultiLogBohm(AnomalousTransportModel):


    def __init__(self, zs, cs):
        if len(zs) != len(cs):
            raise ValueError("Number of z values must be equal to number of c values")
        self.zs = zs
        self.cs = cs

    def __call__(self, nu_an, params, config):
        cache = params["cache"]
        grid = params["grid"]
        thruster = params["thruster"]

        iteration = params["iteration"][0]
        if iteration > 5:
            return nu_an

        B_array = cache["B"]
        L_ch = thruster.geometry.channel_length
        z_shift = pressure_shift(config.anom_model, config.background_pressure_Torr, L_ch)
        B_interp = LinearInterpolation(grid.cell_centers, B_array)

        import math
        log_cs = [math.log(val) for val in self.cs]
        log_itp = LinearInterpolation(self.zs, log_cs)

        for i, zc in enumerate(grid.cell_centers):
            _ = zc - z_shift
            B_val = B_interp(zc)
            omega_ce = e * B_val / me
            c_val = math.exp(log_itp(zc - z_shift))
            nu_an[i] = c_val * omega_ce

        return nu_an


class GaussianBohm(AnomalousTransportModel):

    def __init__(self, hall_min, hall_max, center, width):
        self.hall_min = hall_min
        self.hall_max = hall_max
        self.center = center
        self.width = width

    def __call__(self, nu_an, params, config):
        cache = params["cache"]
        grid = params["grid"]
        thruster = params["thruster"]

        iteration = params["iteration"][0]
        if iteration > 5:
            return nu_an

        B_array = cache["B"]
        L_ch = thruster.geometry.channel_length
        z_shift = pressure_shift(config.anom_model, config.background_pressure_Torr, L_ch)
        B_interp = LinearInterpolation(grid.cell_centers, B_array)

        import math

        for i, zc in enumerate(grid.cell_centers):
            z = zc - z_shift
            B_val = B_interp(zc)
            omega_ce = e * B_val / me
            exponent = -0.5 * ((z - self.center)/self.width)**2
            # c_val transitions from hall_max down to hall_min at the center
            c_val = self.hall_max * (1.0 - (1.0 - self.hall_min)*math.exp(exponent))
            nu_an[i] = c_val * omega_ce

        return nu_an


#=============================================================================
# Pressure Shift Base + Implementations
#=============================================================================

class PressureShift(AnomalousTransportModel):
    pass


def pressure_shift(model, pB, channel_length):
    if isinstance(model, LogisticPressureShift):
        return logistic_pressure_shift_impl(model, pB, channel_length)
    elif isinstance(model, SimpleLogisticShift):
        return simple_logistic_shift_impl(model, pB, channel_length)
    else:
        return 0.0


def logistic_pressure_shift_impl(model, pB, channel_length):
    z0 = model.z0
    dz = model.dz
    alpha = model.alpha
    pstar = model.pstar

    if pstar != 0.0:
        p_ratio = pB / pstar
    else:
        p_ratio = 0.0

    base = (alpha - 1.0)**(2.0*p_ratio - 1.0) if alpha != 1.0 else 1.0
    if base <= 0:
        denom = 1.0
    else:
        denom = 1.0 + base
    zstar = z0 + dz / denom
    return channel_length * zstar


def simple_logistic_shift_impl(model, pB, channel_length):
    shift_length = model.shift_length
    midpoint = model.midpoint_pressure
    slope = model.slope

    if midpoint == 0.0:
        return 0.0

    p_ratio = pB / midpoint

    import math
    val1 = 1.0 / (1.0 + math.exp(-slope*(p_ratio - 1.0)))
    val2 = 1.0 / (1.0 + math.exp(slope))
    zstar = shift_length * (val1 - val2)
    return -channel_length * zstar


class LogisticPressureShift(PressureShift):
    def __init__(self, model, z0, dz, pstar, alpha):
        self.model = model
        self.z0 = z0
        self.dz = dz
        self.pstar = pstar
        self.alpha = alpha

    def __call__(self, nu_an, params, config):
        return self.model(nu_an, params, config)


class SimpleLogisticShift(PressureShift):
    def __init__(self, model, shift_length, midpoint_pressure, slope=2.0):
        self.model = model
        self.shift_length = shift_length
        self.midpoint_pressure = midpoint_pressure
        self.slope = slope

    def __call__(self, nu_an, params, config):
        return self.model(nu_an, params, config)


def num_anom_variables(model):
    return 0
