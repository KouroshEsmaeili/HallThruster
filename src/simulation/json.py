import os
import json
from collections import OrderedDict
from .postprocess import thrust, time_average, anode_eff, divergence_eff, ion_current, mass_eff, voltage_eff,current_eff,divergence_eff
from ..utilities.serialization import deserialize, serialize


def discharge_current(sol, frame):
    return 0.0




def run_simulation(json_file: str, restart: str = ""):
    if not os.path.exists(json_file):
        json_file = os.path.join(os.path.dirname(__file__), json_file)

    # Check extension.
    _, ext = os.path.splitext(json_file)
    if ext.lower() != ".json":
        raise ValueError(f"{json_file} is not a valid JSON file")

    with open(json_file, "r") as f:
        # Allow inf values.
        content = f.read()
    # Read JSON (using object_pairs_hook=OrderedDict to preserve order).
    obj = json.loads(content, object_pairs_hook=OrderedDict)

    # Extract input. If "input" key exists, use it.
    input_data = obj.get("input", obj)

    # Deserialize configuration and simulation parameters.
    # These functions must be provided elsewhere.
    cfg = deserialize("Config", input_data["config"])
    sim = deserialize("SimParams", input_data["simulation"])

    postprocess = None
    if "postprocess" in input_data and "output_file" in input_data["postprocess"] and input_data["postprocess"][
        "output_file"]:
        postprocess = deserialize("Postprocess", input_data["postprocess"])
    from ..simulation.simulation import run_simulation as run_simulation_internal
    sol = run_simulation_internal(cfg, sim, postprocess, os.path.dirname(json_file), restart)

    if postprocess is not None:
        average_start_time = postprocess.get("average_start_time", -1)
        save_time_resolved = postprocess.get("save_time_resolved", True)
        write_to_json(postprocess["output_file"], sol, average_start_time=average_start_time,
                      save_time_resolved=save_time_resolved)

    return sol


def frame_dict(sol, frame: int):
    print('[frame_dict] frame_dict is started')
    grid = sol.params["grid"]
    ncharge = sol.config.ncharge
    f = sol.frames[frame - 1]  # Convert 1-based Julia index to 0-based Python index.
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
    d["z"] = grid["cell_centers"]  # Assume grid is a dict with cell_centers.
    d["nn"] = f.get("nn")
    # For ions, assume f contains arrays "ni", "ui", and "niui" for each charge state.
    d["ni"] = [f["ni"][Z - 1] for Z in range(1, ncharge + 1)]
    d["ui"] = [f["ui"][Z - 1] for Z in range(1, ncharge + 1)]
    d["niui"] = [f["niui"][Z - 1] for Z in range(1, ncharge + 1)]
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


def serialize_sol(sol, average_start_time: float = -1, save_time_resolved: bool = True):
    output = OrderedDict()
    output["retcode"] = str(sol.retcode)
    output["error"] = sol.error

    if average_start_time >= 0:
        # Find first frame where time >= average_start_time.
        first_frame = next((i for i, t in enumerate(sol.t) if t >= average_start_time), 0)
        # If no such frame exists, default to frame 0.
        if first_frame is None:
            first_frame = 0
        avg = time_average(sol, first_frame)
        output["average"] = frame_dict(avg, 1)
    if save_time_resolved:
        output["frames"] = [frame_dict(sol, i + 1) for i in range(len(sol.frames))]
    result = OrderedDict({
        "input": OrderedDict({
            "config": serialize(sol.config),
            "simulation": serialize(sol.params["simulation"]),
            "postprocess": serialize(sol.params.get("postprocess", {}))
        }),
        "output": output
    })
    return result


def write_to_json(file: str, sol, average_start_time: float = -1.0, save_time_resolved: bool = True):
    _, ext = os.path.splitext(file)
    if ext.lower() != ".json":
        raise ValueError(f"{file} is not a JSON file.")
    output = serialize_sol(sol, average_start_time=average_start_time, save_time_resolved=save_time_resolved)
    with open(file, "w") as f:
        # Use json.dump with allow_nan=True.
        json.dump(output, f, allow_nan=True, indent=2)
    return

