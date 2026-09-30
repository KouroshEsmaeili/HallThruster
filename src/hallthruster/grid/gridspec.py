import math

from .grid import Grid1D
from ..thruster.geometry import Geometry1D


class GridSpec:
    def __init__(self, grid_type, num_cells):
        self.type = grid_type
        self.num_cells = num_cells

    def __repr__(self):
        return f"<GridSpec type={self.type}, num_cells={self.num_cells}>"

def EvenGrid(n: int) -> GridSpec:
    return GridSpec(":EvenGrid", n)

def UnevenGrid(n: int) -> GridSpec:
    return GridSpec(":UnevenGrid", n)

def generate_grid(grid: GridSpec, geom: Geometry1D, domain):
    # print('[generate_grid] grid.type',grid.type)
    # print("[generate_grid] domain",domain)
    # print('[generate_grid] grid.num_cells',grid.num_cells)
    # print('[generate_grid] geom',geom)

    if grid.type == ":EvenGrid":
        return generate_even_grid(geom, domain, grid.num_cells)
    elif grid.type == ":UnevenGrid":
        return generate_uneven_grid(geom, domain, grid.num_cells)
    else:
        raise ValueError(f"Invalid grid type {grid.type}. Select ':EvenGrid' or ':UnevenGrid'.")

def generate_even_grid(geometry: Geometry1D, domain, num_cells: int) -> Grid1D:
    x0, x1 = domain
    num_edges = num_cells + 1
    step = (x1 - x0)/num_cells
    edges = []
    for i in range(num_edges):
        edges.append(x0 + i*step)
    # print("[generate_even_grid] len(edges)",len(edges))
    # print("[generate_even_grid] edges",edges)

    return Grid1D(edges)

def uneven_grid_density(z: float, Lch: float) -> float:
    center = 1.5 * Lch
    width = 0.5 * Lch
    if z < center:
        return 2.0
    else:
        return 1.0 + math.exp(-((z - center)/width)**2)

def points_from_density(density_fn, domain, N: int):
    import numpy as np
    x0, x1 = domain
    M = 2000  # for example
    xs = np.linspace(x0, x1, M)
    den = np.array([density_fn(float(z)) for z in xs],dtype=np.float64)
    cdf_array = np.cumsum(den)
    cdf_min, cdf_max = cdf_array[0], cdf_array[-1]

    # We want N edges from cdf_min..cdf_max
    cdf_linspace = np.linspace(cdf_min, cdf_max, N)

    # We'll invert via a search:
    result = []
    for cval in cdf_linspace:
        # find index in cdf_array
        idx = np.searchsorted(cdf_array, cval)
        idx = max(0, min(idx, M-1))
        result.append(xs[idx])
    return result

def generate_uneven_grid(geometry: Geometry1D, domain, num_cells: int) -> Grid1D:
    def density_fn(z):
        return uneven_grid_density(z, geometry.channel_length)

    num_edges = num_cells + 1
    edges = points_from_density(density_fn, domain, num_edges)
    return Grid1D(edges)

