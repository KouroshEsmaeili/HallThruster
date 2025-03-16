from typing import List
import math
from ..utilities.units import convert_to_float64, units
from ..physics.fluid import ContinuityOnly, IsothermalEuler, ranges
from ..walls.wall_sheath import wall_loss_scale
from ..physics.gas import Xenon
from ..numerics.schemes import HyperbolicScheme

class Config:
    def __init__(self, *,
                 thruster,
                 domain,
                 discharge_voltage,
                 anode_mass_flow_rate,
                 ncharge: int = 1,
                 propellant = None,
                 cathode_coupling_voltage: float = 0.0,
                 anode_boundary_condition: str = "sheath",
                 anode_Tev: float = 2.0,
                 cathode_Tev: float = 2.0,
                 anom_model = None,
                 wall_loss_model = None,
                 conductivity_model = None,
                 electron_ion_collisions: bool = True,
                 neutral_velocity = None,
                 neutral_temperature_K = None,
                 ion_temperature_K: float = 1000.0,
                 ion_wall_losses: bool = False,
                 background_pressure_Torr: float = 0.0,
                 background_temperature_K: float = 150.0,
                 neutral_ingestion_multiplier: float = 1.0,
                 solve_plume: bool = False,
                 apply_thrust_divergence_correction: bool = False,
                 electron_plume_loss_scale: float = 1.0,
                 magnetic_field_scale: float = 1.0,
                 transition_length = None,
                 scheme = None,
                 initial_condition = None,
                 implicit_energy: float = 1.0,
                 reaction_rate_directories: List[str] = None,
                 anom_smoothing_iters: int = 0,
                 LANDMARK: bool = False,
                 ionization_model: str = "Lookup",
                 excitation_model: str = "Lookup",
                 electron_neutral_model: str = "Lookup",
                 source_neutrals = None,
                 source_ion_continuity = None,
                 source_ion_momentum = None,
                 source_energy = None,
                 ):
        if reaction_rate_directories is None:
            reaction_rate_directories = []
        self.thruster = thruster
        self.reaction_rate_directories = reaction_rate_directories
        self.source_ion_continuity = ion_source_terms(ncharge, source_ion_continuity, "continuity")
        # print('source_ion_continuity',self.source_ion_continuity)
        self.source_ion_momentum = ion_source_terms(ncharge, source_ion_momentum, "momentum")
        if source_neutrals is None:
            source_neutrals = 0.0
        self.source_neutrals = source_neutrals

        if scheme is None:
            scheme = HyperbolicScheme()
        self.scheme = scheme

        self.discharge_voltage = convert_to_float64(discharge_voltage, units("V"))
        self.cathode_coupling_voltage = convert_to_float64(cathode_coupling_voltage, units("V"))
        self.anode_Tev = convert_to_float64(anode_Tev, units("eV"))
        self.cathode_Tev = convert_to_float64(cathode_Tev, units("eV"))

        default_neutral_velocity = 150.0
        default_neutral_temp = 500.0
        if neutral_velocity is None and neutral_temperature_K is None:
            neutral_velocity = default_neutral_velocity
            neutral_temperature_K = default_neutral_temp
        elif neutral_temperature_K is None:
            neutral_temperature_K = default_neutral_temp
            neutral_velocity = convert_to_float64(neutral_velocity, units("m")/units("s"))
        elif neutral_velocity is None:
            neutral_temperature_K = convert_to_float64(neutral_temperature_K, units("K"))
            neutral_velocity = 0.25 * math.sqrt(8 * 1.380649e-23 * neutral_temperature_K / math.pi / propellant.m)
        else:
            neutral_velocity = convert_to_float64(neutral_velocity, units("m")/units("s"))
            neutral_temperature_K = convert_to_float64(neutral_temperature_K, units("K"))
        self.neutral_velocity = neutral_velocity
        self.neutral_temperature_K = neutral_temperature_K

        self.ion_temperature_K = convert_to_float64(ion_temperature_K, units("K"))
        self.domain = (convert_to_float64(domain[0], units("m")), convert_to_float64(domain[1], units("m")))
        self.anode_mass_flow_rate = convert_to_float64(anode_mass_flow_rate, units("kg")/units("s"))
        self.ncharge = ncharge
        if propellant is None:
            propellant = Xenon
        self.propellant = propellant
        self.anode_boundary_condition = anode_boundary_condition
        if anom_model is None:
            from ..collisions.anomalous import TwoZoneBohm
            self.anom_model = TwoZoneBohm(1/160, 1/16)

        if wall_loss_model is None:
            from ..walls.wall_sheath import WallSheath
            from ..walls.materials import BNSiO2
            self.wall_loss_model = WallSheath(BNSiO2, 1.0)

        if conductivity_model is None:
            from ..physics.thermal_conductivity import  Mitchner  # adjust import path as needed
            conductivity_model = Mitchner()
        self.conductivity_model = conductivity_model
        self.electron_ion_collisions = electron_ion_collisions
        self.ion_wall_losses = ion_wall_losses
        self.background_pressure_Torr = convert_to_float64(background_pressure_Torr, units("Pa"))
        self.background_temperature_K = convert_to_float64(background_temperature_K, units("K"))
        self.neutral_ingestion_multiplier = neutral_ingestion_multiplier
        self.solve_plume = solve_plume
        self.apply_thrust_divergence_correction = apply_thrust_divergence_correction
        self.electron_plume_loss_scale = electron_plume_loss_scale
        self.magnetic_field_scale = magnetic_field_scale
        if transition_length is None:
            transition_length = 0.1 * self.thruster.geometry.channel_length
        self.transition_length = convert_to_float64(transition_length, units("m"))
        self.scheme = scheme
        self.initial_condition = initial_condition
        self.implicit_energy = implicit_energy
        self.anom_smoothing_iters = anom_smoothing_iters
        self.LANDMARK = LANDMARK
        self.ionization_model = ionization_model
        self.excitation_model = excitation_model
        self.electron_neutral_model = electron_neutral_model
        if source_energy is None:
            source_energy = lambda params, i: 0.0
        self.source_energy = source_energy
    def __repr__(self):
        return (f"Config(discharge_voltage={self.discharge_voltage}, domain={self.domain}, "
                f"thruster={self.thruster})")

# --- Serialization Helper ---
def exclude_config():
    return ("source_neutrals", "source_ion_continuity", "source_ion_momentum", "source_potential", "source_energy")

# --- Ion Source Terms Helper ---
def ion_source_terms(ncharge, source, typ):
    # print('[ion_source_terms]', ncharge, source)
    if source is None:
        return [0.0 for _ in range(ncharge)]
    if len(source) != ncharge:
        raise ValueError(f"Number of ion {typ} source terms must match number of charges")
    return source


# --- Configure Fluids ---
def configure_fluids(config):
    propellant = config.propellant
    neutral_fluid = ContinuityOnly(propellant(0), u=config.neutral_velocity, T=config.neutral_temperature_K)
    ion_fluids = [IsothermalEuler(propellant(Z), T=config.ion_temperature_K) for Z in range(1, config.ncharge + 1)]
    fluids = [neutral_fluid] + ion_fluids
    species = [fluid.species for fluid in fluids]
    fluid_ranges = ranges(fluids)
    species_range_dict = {str(fluid.species): None for fluid in fluids}
    for fluid, frange in zip(fluids, fluid_ranges):
        species_range_dict[str(fluid.species)] = frange
    last_index = fluid_ranges[-1][-1]

    # print('[configure_fluids] last_index:', last_index)
    # is_velocity_index = [False] * (last_index + 1)

    # Cyrus was here
    is_velocity_index = [False] * last_index


    # print('[configure_fluids] len(is_velocity_index):', len(is_velocity_index))

    for i in range(3, last_index + 1, 2):
        # print('[configure_fluids] i:',i)
        # is_velocity_index[i] = True

        # Cyrus was here
        is_velocity_index[i-1] = True
    return fluids, fluid_ranges, species, species_range_dict, is_velocity_index

# --- Configure Index ---
def configure_index(fluids, fluid_ranges):
    first_ion_index = next((i for i, f in enumerate(fluids) if f.species.Z > 0), None)
    if first_ion_index is None:
        first_ion_index = len(fluids)
    # print('[configure_index] fluid_ranges',fluid_ranges)
    index_dict = {}
    index_dict["ρn"] = fluid_ranges[0]
    ion_ranges = fluid_ranges[first_ion_index:]
    # print('[configure_index] ion_ranges',ion_ranges)

    index_dict["ρi"] = {i + 1: ion_ranges[i][0] for i in range(len(ion_ranges))}
    index_dict["ρiui"] = {i + 1: ion_ranges[i][1] for i in range(len(ion_ranges))}

    return index_dict

# --- Parameters from Config ---
def params_from_config(config):
    # print(config.wall_loss_model)
    return {
        "thruster": config.thruster,
        "ncharge": config.ncharge,
        "mi": config.propellant.m,
        "anode_bc": config.anode_boundary_condition,
        "landmark": config.LANDMARK,
        "transition_length": config.transition_length,
        "Te_L": config.anode_Tev,
        "Te_R": config.cathode_Tev,
        "implicit_energy": config.implicit_energy,
        "ion_temperature_K": config.ion_temperature_K,
        "neutral_velocity": config.neutral_velocity,
        "anode_mass_flow_rate": config.anode_mass_flow_rate,
        "neutral_ingestion_multiplier": config.neutral_ingestion_multiplier,
        "ion_wall_losses": config.ion_wall_losses,
        "wall_loss_scale": wall_loss_scale(config.wall_loss_model),
        "plume_loss_scale": config.electron_plume_loss_scale,
        "anom_smoothing_iters": config.anom_smoothing_iters,
        "discharge_voltage": config.discharge_voltage,
        "cathode_coupling_voltage": config.cathode_coupling_voltage,
        "electron_ion_collisions": config.electron_ion_collisions,
    }