import os

from ._paths import (
    LANDMARK_FOLDER,
    LANDMARK_RATES_FILE,
    MIN_NUMBER_DENSITY,
    PACKAGE_ROOT,
    PYTHON_PATH,
    REACTION_FOLDER,
    TEST_DIR,
)

from .physics import thermal_conductivity, thermodynamics
from .walls.materials import BoronNitride
from .simulation.simulation import SimParams, run_simulation
from .simulation.postprocess import time_average, discharge_current, thrust, mass_eff, current_eff, divergence_eff, \
    voltage_eff
from .thruster import spt100
from .walls import wall_sheath, constant_sheath_potential
from .collisions.anomalous import MultiLogBohm, GaussianBohm
from .numerics.flux_functions import global_lax_friedrichs, HLLE



def get_python_path():
    """Return the absolute path to the HallThruster Python code."""
    return os.fspath(PYTHON_PATH)


# --- Example Simulation Function ---
def example_simulation(*, ncells, duration, dt, nsave):
    from .simulation.configuration import Config
    from .grid.gridspec import EvenGrid

    from .numerics.limiters import van_albada, minmod
    from .numerics.schemes import HyperbolicScheme

    # Configuration 1.
    config_1 = Config(
        thruster=spt100.SPT_100,
        domain=(0.0, 0.08),
        discharge_voltage=300.0,
        anode_mass_flow_rate=5e-6,
        wall_loss_model=wall_sheath.WallSheath(material=BoronNitride),
        neutral_temperature_K=500,
        scheme=HyperbolicScheme(
            flux_function=global_lax_friedrichs,
            limiter=van_albada,
        ),
    )
    sim = SimParams(
        grid=EvenGrid(ncells),
        duration=duration,
        dt=dt,
        num_save=nsave,
        verbose=False,
        adaptive=False,
    )
    sol_1 = run_simulation(config_1, sim)
    if sol_1.retcode != "success":
        raise Exception("Simulation 1 failed")
    from .walls.constant_sheath_potential import ConstantSheathPotential
    from .numerics.schemes import HyperbolicScheme

    # Configuration 2.
    config_2 = Config(
        thruster=spt100.SPT_100,
        domain=(0.0, 0.08),
        discharge_voltage=300.0,
        anode_mass_flow_rate=5e-6,
        anom_model=MultiLogBohm([0.02, 0.025, 0.03], [0.0625, 0.00625, 0.0625]),
        wall_loss_model=ConstantSheathPotential(20.0, 1.0, 1.0),
        LANDMARK=True,
        conductivity_model=thermal_conductivity.LANDMARK_conductivity(),
        neutral_temperature_K=500,
        ion_wall_losses=True,
        solve_plume=True,
        scheme=HyperbolicScheme(
            flux_function=HLLE,
            limiter=minmod,
        ),
    )
    sim_2 = SimParams(
        grid=sim.grid, duration=duration, dt=dt, num_save=nsave,
        adaptive=True, CFL=0.75, verbose=False,
    )
    sol_2 = run_simulation(config_2, sim_2)
    if sol_2.retcode != "success":
        raise Exception("Simulation 2 failed")

    # Configuration 3.
    config_3 = Config(
        thruster=spt100.SPT_100,
        domain=(0.0, 0.08),
        discharge_voltage=300.0,
        anode_mass_flow_rate=5e-6,
        anom_model=GaussianBohm(
            hall_min=0.00625, hall_max=0.0625, center=0.025, width=0.002,
        ),

        wall_loss_model=constant_sheath_potential.ConstantSheathPotential(20.0, 1.0, 1.0),
        conductivity_model=thermal_conductivity.Braginskii(),
        neutral_temperature_K=500,
        ion_wall_losses=False,
        solve_plume=False,
    )
    sim_3 = SimParams(
        grid=sim.grid, duration=duration, dt=dt, num_save=nsave,
        adaptive=True, CFL=0.75, verbose=False,
    )
    sol_3 = run_simulation(config_3, sim_3)
    if sol_3.retcode != "success":
        raise Exception("Simulation 3 failed")

    # Call some postprocessing functions to exercise that code.
    time_average(sol_1)
    discharge_current(sol_1)
    thrust(sol_1)
    mass_eff(sol_1)
    current_eff(sol_1)
    divergence_eff(sol_1)
    voltage_eff(sol_1)

    return sol_1


# --- Precompile Workload ---
def precompile_workload():
    from .simulation.json import run_simulation as run_json_simulation
    sol = example_simulation(ncells=20, duration=1e-7, dt=1e-8, nsave=2)
    precompile_dir = os.path.join(os.fspath(TEST_DIR), "precompile")
    for file in os.listdir(precompile_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(precompile_dir, file)
        try:
            _ = run_json_simulation(file_path)
        except Exception:
            continue
    output_file = "__output.json"
    if os.path.exists(output_file):
        os.remove(output_file)


# --- Precompile Block (Executed When Module is Run Directly) ---
if __name__ == "__main__":
    precompile_workload()
