from dataclasses import dataclass, field
import math
from ..utilities.units import convert_to_float64, units


def compute_channel_area(outer_radius: float, inner_radius: float) -> float:
    return math.pi * (outer_radius ** 2 - inner_radius ** 2)


@dataclass
class Geometry1D:
    channel_length: float
    inner_radius: float
    outer_radius: float
    channel_area: float = field(init=False)

    def __post_init__(self):
        # Convert input parameters to float using provided units.
        self.channel_length = convert_to_float64(self.channel_length, units(":m"))
        self.inner_radius = convert_to_float64(self.inner_radius, units(":m"))
        self.outer_radius = convert_to_float64(self.outer_radius, units(":m"))
        # Compute the channel area.
        self.channel_area = compute_channel_area(self.outer_radius, self.inner_radius)



def channel_area_fn(outer_radius: float, inner_radius: float) -> float:
    return math.pi * (outer_radius ** 2 - inner_radius ** 2)


def channel_area_from_geometry(geometry: Geometry1D) -> float:
    return geometry.channel_area


def channel_perimeter(outer_radius: float, inner_radius: float) -> float:
    return 2 * math.pi * (outer_radius + inner_radius)


def channel_perimeter_from_geometry(geometry: Geometry1D) -> float:
    return channel_perimeter(geometry.outer_radius, geometry.inner_radius)


def channel_width(outer_radius: float, inner_radius: float) -> float:
    return outer_radius - inner_radius


def channel_width_from_geometry(geometry: Geometry1D) -> float:
    return channel_width(geometry.outer_radius, geometry.inner_radius)
