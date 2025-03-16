# simulation.py
"""
simulation.py

Provides the SimParams class and functions to set up and run a Hall thruster simulation.
It defines:
  - SimParams: a class holding grid, timestep, duration, and adaptive timestepping settings.
  - setup_simulation(config, sim, postprocess=None, include_dirs=None, restart=""):
      Creates the simulation state vector U and a parameters dictionary by:
         • Configuring fluids and indices.
         • Loading collision and reaction data.
         • Generating the computational grid and allocating U.
         • Loading and interpolating the magnetic field.
         • Adjusting the initial timestep for adaptive timestepping.
         • Merging parameters from config (via params_from_config) with additional values.
         • Initializing the state via initialize, then initializing anomalous frequencies,
           restart data (if provided), plume geometry, heavy species, and electrons.
      Returns a tuple (U, params).

  - run_from_setup(U, params, config):
      Defines the simulation time span and saving instants, calls the solver, and returns a Solution.

  - run_simulation(config, sim, **kwargs):
      Sets up the simulation and then runs it, returning a Solution object.

  - run_simulation_deprecated(config, **kwargs):
      (Deprecated) Run simulation using only a Config object.

All time values are converted using convert_to_float64 and units (assumed to be imported).
"""

import math
import numpy as np
import warnings

from numpy.f2py.symbolic import COUNTER

from ..simulation.solution import solve
from ..utilities.units import  convert_to_float64, units
from ..physics.physicalconstants import me  # Electron mass
from .allocation import allocate_arrays_from_grid
from ..grid.gridspec import generate_grid
from ..thruster.magnetic_field import load_magnetic_field_inplace
from ..utilities.interpolation import LinearInterpolation
from .initialization import initialize, initialize_from_restart
from .plume import initialize_plume_geometry
from .update_heavy_species import update_heavy_species as update_heavy_species_
from .update_electrons import update_electrons as update_electrons_
from ..collisions.anomalous import TwoZoneBohm
from ..collisions.ionization import load_ionization_reactions
from ..collisions.reactions import reactant_indices, product_indices
from ..collisions.elastic import load_elastic_collisions
from ..utilities.utility_functions import background_neutral_velocity, background_neutral_density
from ..physics.thermal_conductivity import LANDMARK_conductivity, Mitchner
from .current_control import NoController
from ..collisions.excitation import load_excitation_reactions

# --- SimParams Class ---
class SimParams:
    """
    Holds simulation parameters.

    Attributes:
      grid: A grid specifier (e.g., produced by EvenGrid or UnevenGrid).
      dt: Base timestep (seconds).
      duration: Total simulation duration (seconds).
      num_save: Number of simulation frames to save.
      verbose: If True, print simulation info.
      print_errors: If True, print errors in addition to storing them.
      adaptive: If True, use adaptive timestepping.
      CFL: CFL number for adaptive timestepping.
      min_dt: Minimum allowed timestep (seconds).
      max_dt: Maximum allowed timestep (seconds).
      max_small_steps: Maximum count of minimal timesteps.
      current_control: A current controller (object); if not provided, default NoController() is used.
    """

    def __init__(self, *, grid, dt, duration, num_save=1000, verbose=True, print_errors=True,
                 adaptive=True, CFL=0.799, min_dt=1e-10, max_dt=1e-7, max_small_steps=100, current_control=None):
        self.grid = grid
        self.dt = convert_to_float64(dt, units("s"))
        self.duration = convert_to_float64(duration, units("s"))
        self.num_save = num_save
        self.verbose = verbose
        self.print_errors = print_errors
        self.adaptive = adaptive
        self.CFL = CFL
        self.min_dt = convert_to_float64(min_dt, units("s"))
        self.max_dt = convert_to_float64(max_dt, units("s"))
        self.max_small_steps = max_small_steps
        if current_control is None:
            self.current_control = NoController()
        else:
            self.current_control = current_control


# --- Setup Simulation Function ---
def setup_simulation(config, sim, *, postprocess=None, include_dirs=None, restart=""):
    print('[setup_simulation] setup_simulation is started')
    """
    Set up the simulation state vector and parameters.

    Parameters:
      config: A configuration object/dictionary (from Config) containing thruster geometry, plasma properties, etc.
      sim: A SimParams instance containing grid and timestepping information.
      postprocess: Optional postprocessing settings.
      include_dirs: Optional list of directories to search for files.
      restart: Optional path to a restart JSON file.

    Returns:
      (U, params): U is the state vector (e.g., a NumPy array), and params is a dictionary with simulation parameters.
    """
    if include_dirs is None:
        include_dirs = []


    if config.LANDMARK:
        # print('[setup_simulation] isinstance(config.conductivity_model, LANDMARK_conductivity)',isinstance(config.conductivity_model, LANDMARK_conductivity))
        # print('[setup_simulation] type(config.conductivity_model)',type(config.conductivity_model))
        # print('[setup_simulation] type(LANDMARK_conductivity)',type(LANDMARK_conductivity))

        if not isinstance(config.conductivity_model, LANDMARK_conductivity):
            raise ValueError("LANDMARK configuration needs to use the LANDMARK thermal conductivity model.")


    # Cyrus was here
    # Check Landmark: if config.LANDMARK is True, conductivity_model must be LANDMARK_conductivity.
    # if not config.LANDMARK:
    #     print('[setup_simulation] isinstance(config.conductivity_model, LANDMARK_conductivity)',isinstance(config.conductivity_model, Mitchner))
    #     print('[setup_simulation] type(config.conductivity_model)',type(config.conductivity_model))
    #     print('[setup_simulation] type(LANDMARK_conductivity)',type(LANDMARK_conductivity))
    #
    #     if not isinstance(config.conductivity_model, Mitchner):
    #         raise ValueError("LANDMARK configuration needs to use the LANDMARK thermal conductivity model.")
    # if not config.LANDMARK:
    #     print('[setup_simulation] isinstance(config.conductivity_model, LANDMARK_conductivity)',isinstance(config.conductivity_model, LANDMARK_conductivity))
    #     print('[setup_simulation] type(config.conductivity_model)',type(config.conductivity_model))
    #     print('[setup_simulation] type(LANDMARK_conductivity)',type(LANDMARK_conductivity))
    #
    #     if not isinstance(config.conductivity_model, LANDMARK_conductivity):
    #         raise ValueError("LANDMARK configuration needs to use the LANDMARK thermal conductivity model.")

    from .configuration import configure_fluids, configure_index

    # Configure fluids and indices.
    fluids, fluid_ranges, species, species_range_dict, is_velocity_index = configure_fluids(config)
    index = configure_index(fluids, fluid_ranges)
    # print("index",index)

    ionization_reactions = load_ionization_reactions(config.ionization_model, np.unique(species),
                                                     directories=config.reaction_rate_directories)
    ionization_reactant_indices = reactant_indices(ionization_reactions, species_range_dict)
    ionization_product_indices = product_indices(ionization_reactions, species_range_dict)

    excitation_reactions = load_excitation_reactions(config.excitation_model, np.unique(species))
    # (If your module names it load_excitation_reactions, adjust accordingly)
    excitation_reactant_indices = reactant_indices(excitation_reactions, species_range_dict)

    electron_neutral_collisions = load_elastic_collisions(config.electron_neutral_model, np.unique(species))

    # Generate grid and allocate state.

    grid = generate_grid(sim.grid, config.thruster.geometry, config.domain)
    U, cache = allocate_arrays_from_grid(grid, config)
    # print('[setup_simulation] U', U)

    # print('[setup_simulation] cahse["ϕ"]:', cache["ϕ"])
    # print('[setup_simulation] U 1:', U)


    # Load magnetic field.
    thruster = config.thruster
    load_magnetic_field_inplace(thruster.magnetic_field, include_dirs=include_dirs)

    # Interpolate magnetic field onto grid cell centers.
    itp = LinearInterpolation(thruster.magnetic_field.z, thruster.magnetic_field.B)
    cache["B"] = np.array([itp(z) for z in grid.cell_centers],dtype=np.float64)

    # Adaptive timestep: if adaptive, set dt to a very small initial value.
    dt_val = sim.dt
    # print('dt_val',dt_val)
    if sim.adaptive:
        dt_val = 100 * np.finfo(float).eps
        if sim.CFL >= 0.8:
            if sim.print_errors:
                warnings.warn(
                    "CFL for adaptive timestepping set higher than stability limit of 0.8. Setting CFL to 0.799.")
            sim.CFL = 0.799
    cache["dt"][:] = dt_val
    from .configuration import params_from_config  # Converts config to a dict of concrete values
    # Build a simulation parameters dictionary.
    base_params = params_from_config(config)
    params_dict = {**base_params,
                   "simulation": sim,
                   "iteration": [-1],
                   "dt": np.array([dt_val],dtype=np.float64),
                   "grid": grid,
                   "postprocess": postprocess,
                   "index": index,
                   "cache": cache,
                   "fluids": fluids,
                   "fluid_ranges": fluid_ranges,
                   "species_range_dict": species_range_dict,
                   "is_velocity_index": is_velocity_index,
                   "ionization_reactions": ionization_reactions,
                   "ionization_reactant_indices": ionization_reactant_indices,
                   "ionization_product_indices": ionization_product_indices,
                   "excitation_reactions": excitation_reactions,
                   "excitation_reactant_indices": excitation_reactant_indices,
                   "electron_neutral_collisions": electron_neutral_collisions,
                   "background_neutral_velocity": background_neutral_velocity(config),
                   "background_neutral_density": background_neutral_density(config),
                   "γ_SEE_max": 1 - 8.3 * math.sqrt(me / config.propellant.m),
                   "min_Te": min(config.anode_Tev, config.cathode_Tev),
                   }

    # Initialize ion and electron variables.
    # print('[setup_simulation] U :', U)

    initialize(U, params_dict, config)
    # print('[setup_simulation] U 2:', U)
    # print('[setup_simulation] params.cache.ne:', params_dict['cache']['ne'])

    # Initialize anomalous collision frequency using TwoZoneBohm.
    TwoZoneBohm(1 / 160, 1 / 16)(params_dict["cache"]["νan"], params_dict, config)

    if restart != "":
        initialize_from_restart(U, params_dict, restart)

    initialize_plume_geometry(params_dict)
    # print("[setup_simulation] params_dict.cache['nn']",params_dict['cache']['nn'])


    update_heavy_species_(U, params_dict)
    update_electrons_(params_dict, config)

    return U, params_dict


def run_from_setup(U, params, config):
    # print('[run_from_setup] run_from_setup is started')
    # print('[run_from_setup] params.cache.ne:', params['cache']['ne'])

    t0 = 0.0
    t_end = params["simulation"].duration
    tspan = (t0, t_end)
    print('[run_from_setup] tspan',tspan)


    num_save = params["simulation"].num_save
    saveat = np.linspace(t0, t_end, num=num_save)

    # Assume solve(U, params, config, tspan, saveat) is implemented and returns a Solution object.

    sol = solve(U, params, config, tspan, saveat=saveat)

    # print('[run_from_setup] sol.retcode != "success"',sol.retcode != "Success")
    # print('[run_from_setup] sol.retcode', sol.retcode)
    if sol.retcode != "success" and params["simulation"].verbose:
        print(f"Simulation exited at t = {sol.t[-1]} with retcode {sol.retcode}")
    return sol
#

# --- run_simulation ---
def run_simulation(config, sim, **kwargs):
    # print('[run_simulation] run_simulation is started')

    U, params = setup_simulation(config, sim, **kwargs)

    # print('[run_simulation] params.cache.ne:', params['cache']['ne'])
    # print('[run_simulation] U: ', U)

    return run_from_setup(U, params, config)


def run_simulation_deprecated(config, **kwargs):
    # print('[run_simulation_deprecated] run_simulation_deprecated is started')

    U, params = setup_simulation(config, **kwargs)
    return run_from_setup(U, params, config)
