import math
import numpy as np
from ..physics.physicalconstants import e



def clamp(x, a, b):
    return max(a, min(x, b))

def initialize_plume_geometry(params):
    cache = params["cache"]
    thruster = params["thruster"]
    geometry = thruster.geometry
    r_in = geometry.inner_radius
    r_out = geometry.outer_radius
    A_ch = geometry.channel_area

    # Fill cached arrays element-wise.
    cache["channel_area"].fill(A_ch)
    cache["inner_radius"].fill(r_in)
    cache["outer_radius"].fill(r_out)
    cache["channel_height"].fill(r_out - r_in)
    return

def update_plume_geometry(params):

    cache = params["cache"]
    grid = params["grid"]
    mi = params["mi"]
    thruster = params["thruster"]
    ncharge = params["ncharge"]

    # Unpack arrays from cache.
    channel_area = cache["channel_area"]
    inner_radius = cache["inner_radius"]
    outer_radius = cache["outer_radius"]
    channel_height = cache["channel_height"]
    dA_dz = cache["dA_dz"]
    tan_delta = cache["tanδ"]  # Use key "tanδ" as given.
    Tev = cache["Tev"]
    niui = cache["niui"]  # 2D array: shape (ncharge, ncells)
    ni = cache["ni"]      # 2D array: shape (ncharge, ncells)

    L_ch = thruster.geometry.channel_length
    # Find the exit plane index: first index where cell_centers >= L_ch.
    indices = np.where(grid.cell_centers >= L_ch)[0]
    exit_plane_index = indices[0] if len(indices) > 0 else 0
    Tev_exit = Tev[exit_plane_index]
    inv_mi = 1.0 / mi


    # print('[update_plume_geometry] exit_plane_index, grid.num_cells: ',exit_plane_index ,grid.num_cells)
    for i in range(exit_plane_index + 1, grid.num_cells):
        # print('[update_plume_geometry] i ', i)

        # Sum ion momentum and density for charge states.
        numerator = sum(niui[Z, i] for Z in range(ncharge))
        denominator = sum(ni[Z, i] for Z in range(ncharge))
        ui = numerator / denominator if denominator != 0 else 0.0

        thermal_speed = math.sqrt((5.0/3.0) * e * Tev_exit * inv_mi)
        ui = max(ui, thermal_speed)
        tan_delta[i] = clamp(thermal_speed / ui, 0.0, 1.0)

        avg_tan = 0.5 * (tan_delta[i] + tan_delta[i - 1])
        Δz = grid.cell_centers[i] - grid.cell_centers[i - 1]
        inner_radius[i] = max(0.0, inner_radius[i - 1] - avg_tan * Δz)
        outer_radius[i] = outer_radius[i - 1] + avg_tan * Δz
        channel_height[i] = outer_radius[i] - inner_radius[i]
        channel_area[i] = math.pi * (outer_radius[i]**2 - inner_radius[i]**2)
        dA_dz[i] = (channel_area[i] - channel_area[i - 1]) / Δz
    return
