import math
import numpy as np
import json
from ..utilities.interpolation import LinearInterpolation
from ..utilities.utility_functions import inlet_neutral_density
from ..utilities.smoothing import smooth_if
from ..physics.physicalconstants import e
from ..utilities.interpolation import lerp


class InitialCondition:
    pass


class DefaultInitialization(InitialCondition):
    def __init__(self, max_electron_temperature=-1.0,
                 min_ion_density: float = 2e17,
                 max_ion_density: float = 1e18):
        if max_electron_temperature is None:
            self.max_electron_temperature = -1.0
        # print('[DefaultInitialization]max_electron_temperature',self.max_electron_temperature)

        self.min_ion_density = min_ion_density
        self.max_ion_density = max_ion_density

    def __repr__(self):
        return (f"DefaultInitialization(max_electron_temperature={self.max_electron_temperature}, "
                f"min_ion_density={self.min_ion_density}, max_ion_density={self.max_ion_density})")


def initialize(U, params, config, init_condition=None):
    print('[initialize] initialize is started!!')
    # print('[initialize] params[index]',params['index'])
    # print('[initialize]:',init_condition)
    if init_condition is None:
        init_condition = DefaultInitialization(init_condition)
        # print('init_condition.max_electron_temperature',init_condition.max_electron_temperature)
    if not isinstance(init_condition, DefaultInitialization):
        raise ValueError(f"Initialization for {type(init_condition)} not yet implemented. "
                         "Only DefaultInitialization is supported.")

    # Unpack mandatory fields from config.
    anode_Tev = config.anode_Tev
    cathode_Tev = config.cathode_Tev
    domain = config.domain  # tuple (z_left, z_right)
    discharge_voltage = config.discharge_voltage
    anode_mass_flow_rate = config.anode_mass_flow_rate


    ρn = inlet_neutral_density(config)
    # print("[initialize] ρn :",ρn)


    un = config.neutral_velocity
    initialize_heavy_species_default(U, params, anode_Tev, domain, discharge_voltage,
                                     anode_mass_flow_rate, ρn, un,
                                     min_ion_density=init_condition.min_ion_density,
                                     max_ion_density=init_condition.max_ion_density)
    initialize_electrons_default(params, anode_Tev, cathode_Tev, domain, discharge_voltage,
                                 max_electron_temperature=init_condition.max_electron_temperature)

    # print('[initialize] U', U)

    return


def initialize_heavy_species_default(U, params, anode_Tev, domain, discharge_voltage, anode_mass_flow_rate, ρn_0, un, *,
                                     min_ion_density=2e17, max_ion_density=1e18):
    # print('[initialize_heavy_species_default] initialize_heavy_species_default is started')
    # print('[initialize_heavy_species_default] U',U)
    # print('[initialize_heavy_species_default] ρn_0: ',ρn_0)

    grid = params["grid"]
    index = params["index"]
    cache = params["cache"]
    ncharge = params["ncharge"]
    thruster = params["thruster"]
    mi = params["mi"]
    L_ch = thruster.geometry.channel_length
    z0 = domain[0]  # left boundary

    ni_center = L_ch / 2.0
    ni_width = L_ch / 3.0
    ni_min = min_ion_density
    ni_max = max_ion_density
    scaling_factor = math.sqrt(discharge_voltage / 300.0) * (anode_mass_flow_rate / 5e-6)

    def ion_density_function(z, Z):
        # print('[ion_density_function] Z:',Z)
        # print('[ion_density_function] ni_width:',ni_width)
        return mi * scaling_factor * (
                ni_min + (ni_max - ni_min) * math.exp(-(((z - z0) - ni_center) / ni_width) ** 2)) / (Z ** 2)

    def bohm_velocity(Z):
        return -math.sqrt(Z * e * anode_Tev / mi)

    def final_velocity(Z):
        return math.sqrt(2 * Z * e * discharge_voltage / mi)

    def scale(Z):
        return (2.0 / 3.0) * (final_velocity(Z) - bohm_velocity(Z))

    def ion_velocity_f1(z, Z):
        return bohm_velocity(Z) + scale(Z) * ((z - z0) / L_ch) ** 2

    def ion_velocity_f2(z, Z):
        # Interpolate linearly between ion_velocity_f1 at z = L_ch and final_velocity.
        return lerp(z, z0 + L_ch, domain[1], ion_velocity_f1(L_ch, Z), final_velocity(Z))

    def ion_velocity_function(z, Z):
        if (z - z0) < L_ch:
            return ion_velocity_f1(z, Z)
        else:
            return ion_velocity_f2(z, Z)

    # Subtract recombined neutrals.
    for Z in range(1, ncharge + 1):
        ρn_0 -= ion_velocity_function(0.0, Z) * ion_density_function(0.0, Z) / un
        # print('[initialize_heavy_species_default] ion_velocity_function(0.0, Z): ',ion_velocity_function(0.0, Z))
        # print('[initialize_heavy_species_default] ion_density_function(0.0, Z): ',ion_density_function(0.0, Z))
        # print('[initialize_heavy_species_default] un: ',un)


    ρn_1 = 0.01 * ρn_0  # Beam neutral density at outlet.
    neutral_function = lambda z: smooth_if(z - z0, L_ch / 2.0, ρn_0, ρn_1, L_ch / 6.0)
    # print('[initialize_heavy_species_default] z0, L_ch, ρn_0, ρn_1', z0, L_ch, ρn_0, ρn_1)

    def number_density_function(z):
        s = 0.0
        for Z in range(1, ncharge + 1):
            s += Z * ion_density_function(z, Z) / mi
        return s

    # print("U.shape",U)


    # print("[initialize_heavy_species_default] grid.cell_centers", grid.cell_centers)

    # Fill the state vector U.
    for i, z in enumerate(grid.cell_centers):
        # print('[initialize_heavy_species_default] i, z',i, z)
        # print('[initialize_heavy_species_default]:',len(grid.cell_centers))

        # print('[initialize_heavy_species_default] index:', index)
        # U[index["ρn"], i] = neutral_function(z)

        # Cyrus was here
        new_range = range(index["ρn"].start - 1, index["ρn"].stop - 1, index["ρn"].step)
        U[new_range, i] = neutral_function(z)

        # print("[initialize_heavy_species_default] neutral_function(z):", neutral_function(z))
        # print('[initialize_heavy_species_default] neutral_function:', neutral_function)
        # print('[initialize_heavy_species_default] U:', U)
        # print('index["ρi"]',index["ρi"])
        # print('ncharge',ncharge)
        # print('index["ρiui"]', index["ρiui"])


        for Z in range(1, ncharge + 1):
            # print('Z',Z)
            # print('[initialize_heavy_species_default]:',)

            U[index["ρi"][Z] - 1, i] = ion_density_function(z, Z)
            U[index["ρiui"][Z] - 1, i] = ion_density_function(z, Z) * ion_velocity_function(z, Z)
        cache["ne"][i] = number_density_function(z)
    # print('[initialize_heavy_species_default] U', U)
    return


def initialize_electrons_default(params, anode_Tev, cathode_Tev, domain, discharge_voltage, *,
                                 max_electron_temperature=-1.0):
    grid = params["grid"]
    cache = params["cache"]
    min_Te = params["min_Te"]
    thruster = params["thruster"]
    L_ch = thruster.geometry.channel_length
    z0 = domain[0]

    Te_baseline = lambda z: lerp(z, domain[0], domain[1], anode_Tev, cathode_Tev)
    # print('[initialize_electrons_default]max_electron_temperature:',max_electron_temperature)
    Te_max = max_electron_temperature if max_electron_temperature > 0.0 else discharge_voltage / 10.0
    Te_width = L_ch / 3.0

    energy_function = lambda z: (3.0 / 2.0) * (
            Te_baseline(z) + (Te_max - min_Te) * math.exp(-(((z - z0) - L_ch) / Te_width) ** 2))

    for i, z in enumerate(grid.cell_centers):
        cache["nϵ"][i] = cache["ne"][i] * energy_function(z)
        cache["Tev"][i] = energy_function(z) / 1.5
    return


def initialize_from_restart(U, params, restart_file_or_frame):
    # print('[initialize_from_restart] initialize_from_restart is started')

    if isinstance(restart_file_or_frame, str):
        with open(restart_file_or_frame, 'r') as f:
            restart = json.load(f)
        if "output" in restart:
            restart = restart["output"]
        if "frames" in restart:
            frame = restart["frames"][-1]
        elif "average" in restart:
            frame = restart["average"]
        else:
            raise ValueError(f"Restart file {restart_file_or_frame} has no key 'frames' or 'average'.")
    else:
        frame = restart_file_or_frame

    cache = params["cache"]
    grid = params["grid"]
    ncharge = params["ncharge"]
    mi = params["mi"]
    z = grid.cell_centers

    neutral_itp = LinearInterpolation(frame["z"], np.array(frame["nn"], dtype=np.float64) * mi)
    # Assume U uses 0-based indexing: row 0 for neutrals.
    U[0, :] = np.array([neutral_itp(z_val) for z_val in z], dtype=np.float64)

    ncharge_restart = len(frame["ni"])
    for Z in range(1, min(ncharge, ncharge_restart) + 1):
        ion_itp = LinearInterpolation(frame["z"], np.array(frame["ni"], dtype=np.float64)[Z - 1] * mi)
        U[Z, :] = np.array([ion_itp(z_val) for z_val in z], dtype=np.float64)

        ionui_itp = LinearInterpolation(frame["z"], np.array(frame["niui"], dtype=np.float64)[Z - 1] * mi)

        # Assume that ion momentum values are stored in rows (ncharge + Z).
        # U[Z + ncharge, :] = np.array([ionui_itp(z_val) for z_val in z],dtype=np.float64)


        # Cyrus was here
        U[Z + 1, :] = np.array([ionui_itp(z_val) for z_val in z], dtype=np.float64)

    ne_itp = LinearInterpolation(frame["z"], frame["ne"])
    cache["ne"][:] = np.array([ne_itp(z_val) for z_val in z], dtype=np.float64)

    Te = np.array([LinearInterpolation(frame["z"], frame["Tev"])(z_val) for z_val in z], dtype=np.float64)
    phi = np.array([LinearInterpolation(frame["z"], frame["potential"])(z_val) for z_val in z], dtype=np.float64)
    E = np.array([LinearInterpolation(frame["z"], frame["E"])(z_val) for z_val in z], dtype=np.float64)

    # print('[initialize_from_restart] cache["ϕ"]', len(cache["ϕ"]))
    # print('[initialize_from_restart] phi', len(phi))

    cache["nϵ"][:] = 1.5 * cache["ne"] * Te
    cache["Tev"][:] = Te
    cache["∇ϕ"][:] = -E
    cache["ϕ"][:] = phi
    return
