from dataclasses import dataclass
from src.thruster.geometry import Geometry1D, channel_perimeter, channel_width
from src.thruster.magnetic_field import MagneticField

@dataclass
class Thruster:
    name: str = "noname"
    geometry: Geometry1D = None
    magnetic_field: MagneticField = None
    shielded: bool = False

def channel_area_thruster(thruster: Thruster) -> float:
    return thruster.geometry.channel_area

def channel_perimeter_thruster(thruster: Thruster) -> float:
    return channel_perimeter(thruster.geometry.outer_radius, thruster.geometry.inner_radius)

def channel_width_thruster(thruster: Thruster) -> float:
    return channel_width(thruster.geometry.outer_radius, thruster.geometry.inner_radius)
