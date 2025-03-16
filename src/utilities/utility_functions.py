from math import sqrt, pi, exp
from ..physics.physicalconstants import kB, e


def left_edge(i):
    return i - 1


def right_edge(i):
    return i


def electron_density(U, p, i):
    return sum(Z * U[p.index["ρi"][Z]-1, i] for Z in range(1, p.ncharge + 1)) / p.mi


def inlet_neutral_density(config):
    # print("[inlet_neutral_density0] config.anode_mass_flow_rate: ", config.anode_mass_flow_rate)
    # print("[inlet_neutral_density0] config.neutral_velocity: ", config.neutral_velocity)
    # print("[inlet_neutral_density0] config.thruster.geometry.channel_area: ", config.thruster.geometry.channel_area)

    return config.anode_mass_flow_rate / config.neutral_velocity / config.thruster.geometry.channel_area


def background_neutral_density(config):
    return config.propellant.m * config.background_pressure_Torr / kB / config.background_temperature_K


def background_neutral_velocity(config):
    return 0.25 * sqrt(8 * kB * config.background_temperature_K / (pi * config.propellant.m))


def ion_current_density(U, p, i):
    # print('[ion_current_density] ion_current_density is started')
    return sum(Z * e * U[p.index["ρiui"][Z]-1, i] for Z in range(1, p.config.ncharge + 1)) / p.config.propellant.m


def myerf(x):
    if x < 0:
        return -myerf(-x)
    x_sqrt_pi = x * sqrt(pi)
    x_squared = x ** 2
    h = x_sqrt_pi + (pi - 2) * x_squared
    g = h / (1 + h)
    return 1 - exp(-x_squared) / x_sqrt_pi * g


# Example usage:
if __name__ == "__main__":
    # Simple tests for left/right edge functions
    for i in range(5):
        print(f"Index {i}: left_edge = {left_edge(i)}, right_edge = {right_edge(i)}")

    # Test myerf for a few values
    for x in [0, 0.5, 1, 2, -1]:
        print(f"myerf({x}) = {myerf(x)}")

