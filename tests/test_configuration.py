import math

import pytest

from hallthruster import Config, EvenGrid, SPT_100, SimParams, Xenon
from hallthruster.collisions.anomalous import NoAnom, TwoZoneBohm
from hallthruster.physics.physicalconstants import kB
from hallthruster.walls.no_wall_losses import NoWallLosses


def make_config(**overrides):
    values = {
        "thruster": SPT_100,
        "domain": (0.0, 0.08),
        "discharge_voltage": 300.0,
        "anode_mass_flow_rate": 5e-6,
    }
    values.update(overrides)
    return Config(**values)


def test_config_defaults_and_operating_point():
    config = make_config(ncharge=3)

    assert config.propellant is Xenon
    assert isinstance(config.anom_model, TwoZoneBohm)
    assert config.neutral_velocity == 150.0
    assert config.neutral_temperature_K == 500.0
    assert config.ncharge == 3
    assert config.domain == (0.0, 0.08)
    assert config.anode_mass_flow_rate == 5e-6
    assert config.discharge_voltage == 300.0
    assert config.background_temperature_K == 100.0
    assert all(source(None, None, 0) == 0.0 for source in config.source_ion_continuity)
    assert config.source_neutrals(None, None, 0) == 0.0


def test_config_preserves_custom_models():
    anomalous = NoAnom()
    walls = NoWallLosses()
    config = make_config(anom_model=anomalous, wall_loss_model=walls)

    assert config.anom_model is anomalous
    assert config.wall_loss_model is walls


def test_neutral_temperature_computes_velocity_after_defaulting_to_xenon():
    temperature = 600.0
    config = make_config(neutral_temperature_K=temperature)
    expected = 0.25 * math.sqrt(8 * kB * temperature / math.pi / Xenon.m)

    assert config.neutral_velocity == pytest.approx(expected)
    assert config.neutral_temperature_K == temperature


def test_anode_temperature_defaults_to_cathode_temperature():
    config = make_config(cathode_Tev=3.5)

    assert config.anode_Tev == 3.5
    assert config.cathode_Tev == 3.5


def test_simparams_defaults_and_explicit_values():
    sim = SimParams(grid=EvenGrid(20), dt=1e-8, duration=1e-7, num_save=3)

    assert sim.grid.num_cells == 20
    assert sim.dt == 1e-8
    assert sim.duration == 1e-7
    assert sim.num_save == 3
    assert sim.adaptive is True
    assert sim.min_dt == 1e-10
    assert sim.max_dt == 1e-7
    assert sim.CFL == 0.799
