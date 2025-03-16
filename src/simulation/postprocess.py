from dataclasses import dataclass
from collections import OrderedDict
import math
import copy
import numpy as np
from ..physics.physicalconstants import e


@dataclass
class Postprocess:
    output_file: str = ""
    average_start_time: float = -1.0
    save_time_resolved: bool = False



def time_average(sol, start_time):
    start_time = float(start_time)
    # Find the first index where time >= start_time.
    start_frame = next((i for i, t in enumerate(sol.t) if t >= start_time), 0)
    return time_average_from_frame(sol, start_frame)


def time_average_from_frame(sol, start_frame=0):
    # Deep copy the last frame.
    avg_frame = copy.deepcopy(sol.frames[-1])
    # Assume each frame is a dictionary.
    for key, value in avg_frame.items():
        if key == "anom_variables":
            if isinstance(value, (list, np.ndarray)):
                avg_frame[key] = [0.0 for _ in range(len(value))]
            else:
                avg_frame[key] = 0.0
        else:
            if isinstance(value, np.ndarray):
                avg_frame[key] = np.zeros_like(value)
            elif isinstance(value, (int, float)):
                avg_frame[key] = 0.0
            elif isinstance(value, list):
                avg_frame[key] = [0.0 for _ in value]
            else:
                avg_frame[key] = 0.0

    num_frames = len(sol.t)
    Δt = num_frames - start_frame
    for i in range(start_frame, num_frames):
        frame = sol.frames[i]
        for key, value in frame.items():
            if key == "anom_variables":
                if isinstance(value, (list, np.ndarray)):
                    for j in range(len(value)):
                        avg_frame[key][j] += value[j] / Δt
            else:
                if isinstance(value, np.ndarray):
                    avg_frame[key] += value / Δt
                elif isinstance(value, (int, float)):
                    avg_frame[key] += value / Δt
                elif isinstance(value, list):
                    avg_frame[key] = [avg + (val / Δt) for avg, val in zip(avg_frame[key], value)]

    from ..simulation.solution import Solution

    new_sol = Solution(
        t=sol.t[-1:],
        frames=[avg_frame],
        params=sol.params,
        config=sol.config,
        retcode=sol.retcode,
        error=sol.error
    )
    return new_sol


# --- Frame Dictionary Conversion ---
def frame_dict(sol, frame):
    print('[frame_dict] frame_dict is started')

    grid = sol.params["grid"]  # Assume grid is a dict with key "cell_centers"
    ncharge = config.ncharge
    f = sol.frames[frame - 1]  # Convert from 1-indexed to 0-indexed.
    d = OrderedDict()
    d["thrust"] = thrust(sol, frame)
    d["discharge_current"] = discharge_current(sol, frame)
    d["ion_current"] = ion_current(sol, frame)
    d["mass_eff"] = mass_eff(sol, frame)
    d["voltage_eff"] = voltage_eff(sol, frame)
    d["current_eff"] = current_eff(sol, frame)
    d["divergence_eff"] = divergence_eff(sol, frame)
    d["anode_eff"] = anode_eff(sol, frame)
    d["t"] = sol.t[frame - 1]
    d["z"] = sol.params["grid"]["cell_centers"]
    d["nn"] = f.get("nn")
    d["ni"] = [f["ni"][Z] for Z in range(ncharge)]
    d["ui"] = [f["ui"][Z] for Z in range(ncharge)]
    d["niui"] = [f["niui"][Z] for Z in range(ncharge)]
    d["B"] = sol.params["cache"]["B"]
    d["ne"] = f.get("ne")
    d["ue"] = f.get("ue")
    d["potential"] = f.get("ϕ")
    d["E"] = -f.get("∇ϕ", 0)
    d["Tev"] = f.get("Tev")
    d["pe"] = f.get("pe")
    d["grad_pe"] = f.get("∇pe")
    d["nu_en"] = f.get("νen")
    d["nu_ei"] = f.get("νei")
    d["nu_anom"] = f.get("νan")
    d["nu_class"] = f.get("νc")
    d["mobility"] = f.get("μ")
    d["channel_area"] = f.get("channel_area")
    return d


def thrust(sol, frame):
    mi = sol.params["mi"]
    # print('[thrust] len(sol.frames)',type(sol.frames))
    # print('sol.frames',sol.frames)
    # print('[thrust] frame',frame)

    f = sol.frames[frame]
    left_area = f["channel_area"][0]
    right_area = f["channel_area"][-1]
    thrust_val = 0.0
    ncharge = sol.config.ncharge

    for Z in range(ncharge):
        # print('[thrust] f["niui"]:',f["niui"])
        thrust_val += right_area * mi * (f["niui"][Z, -1] ** 2 / f["ni"][Z, -1])
        thrust_val -= left_area * mi * (f["niui"][Z, 0] ** 2 / f["ni"][Z, 0])
    if sol.config.apply_thrust_divergence_correction:
        return thrust_val * math.sqrt(divergence_eff(sol, frame))
    else:
        return thrust_val


def thrust_all(sol):
    return [thrust(sol, i) for i in range(len(sol.frames))]


def discharge_current(sol, frame):
    f = sol.frames[frame - 1]
    return f["Id"][0] if "Id" in f and len(f["Id"]) > 0 else 0.0


def discharge_current_all(sol):
    return [discharge_current(sol, i + 1) for i in range(len(sol.frames))]


def anode_eff(sol, frame):
    T_val = thrust(sol, frame)
    current = discharge_current(sol, frame)
    Vd = sol.config["discharge_voltage"]
    mdot_a = sol.config["anode_mass_flow_rate"]
    return 0.5 * T_val ** 2 / current / Vd / mdot_a


def anode_eff_all(sol):
    return [anode_eff(sol, i + 1) for i in range(len(sol.frames))]


def voltage_eff(sol, frame):
    Vd = sol.config["discharge_voltage"]
    mi = sol.config["propellant"].m
    f = sol.frames[frame - 1]
    # For the first ion species, assume row index 0.
    ui = f["niui"][0, -1] / f["ni"][0, -1]
    return 0.5 * mi * ui ** 2 / e / Vd


def voltage_eff_all(sol):
    return [voltage_eff(sol, i + 1) for i in range(len(sol.frames))]


def divergence_eff(sol, frame):
    f = sol.frames[frame - 1]
    tan_delta = f["tanδ"][-1]
    delta = math.atan(float(tan_delta))
    return math.cos(delta) ** 2


def divergence_eff_all(sol):
    return [divergence_eff(sol, i + 1) for i in range(len(sol.frames))]


def ion_current(sol, frame):
    Ii = 0.0
    f = sol.frames[frame - 1]
    right_area = f["channel_area"][-1]
    ncharge = sol.config.get("ncharge", 1)
    for Z in range(ncharge):
        Ii += (Z + 1) * e * f["niui"][Z, -1] * right_area
    return Ii


def ion_current_all(sol):
    return [ion_current(sol, i + 1) for i in range(len(sol.frames))]


def electron_current(sol, frame):
    return discharge_current(sol, frame) - ion_current(sol, frame)


def electron_current_all(sol):
    return [electron_current(sol, i + 1) for i in range(len(sol.frames))]


def current_eff(sol, frame):
    dc = discharge_current(sol, frame)
    return ion_current(sol, frame) / dc if dc != 0 else 0.0


def current_eff_all(sol):
    return [current_eff(sol, i + 1) for i in range(len(sol.frames))]


def mass_eff(sol, frame):
    mass_eff_val = 0.0
    f = sol.frames[frame - 1]
    right_area = f["channel_area"][-1]
    mi = sol.params["mi"]
    mdot = sol.config["anode_mass_flow_rate"]
    ncharge = config.ncharge
    for Z in range(ncharge):
        mass_eff_val += mi * f["niui"][Z, -1] * right_area / mdot
    return mass_eff_val


def mass_eff_all(sol):
    return [mass_eff(sol, i + 1) for i in range(len(sol.frames))]

