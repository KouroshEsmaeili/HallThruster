import math
import numpy as np
from .geometry import Geometry1D
from .magnetic_field import MagneticField
from .thruster import Thruster



spt100_geometry = Geometry1D(
    channel_length=0.025,    # in meters
    inner_radius=0.0345,     # in meters
    outer_radius=0.05        # in meters
)

def spt100_analytic_field():
    L_ch = spt100_geometry.channel_length
    B_max = 0.015  # Tesla
    zs = np.linspace(0, 4 * L_ch, 256)
    Bs = np.zeros_like(zs)
    for i, z in enumerate(zs):
        if z < L_ch:
            Bs[i] = B_max * math.exp(-0.5 * ((z - L_ch) / 0.011) ** 2)
        else:
            Bs[i] = B_max * math.exp(-0.5 * ((z - L_ch) / 0.018) ** 2)
    return MagneticField(file="SPT-100 analytic field", z=zs, B=Bs)

SPT_100 = Thruster(
    name="SPT-100",
    geometry=spt100_geometry,
    magnetic_field=spt100_analytic_field(),
    shielded=False
)
