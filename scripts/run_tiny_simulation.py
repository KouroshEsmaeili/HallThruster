"""Run the M1 controlled direct-API simulation smoke case."""

import numpy as np

from hallthruster import Config, EvenGrid, SPT_100, SimParams, run_simulation


config = Config(
    thruster=SPT_100,
    domain=(0.0, 0.08),
    discharge_voltage=300.0,
    anode_mass_flow_rate=5e-6,
    neutral_temperature_K=500.0,
)
sim = SimParams(
    grid=EvenGrid(20),
    dt=1e-8,
    duration=1e-7,
    num_save=3,
    adaptive=False,
    verbose=False,
    print_errors=True,
)
solution = run_simulation(config, sim)

finite = all(
    np.all(np.isfinite(value))
    for frame in solution.frames
    for value in frame.values()
    if isinstance(value, np.ndarray)
)

print(f"retcode={solution.retcode}")
print(f"requested_end_time={sim.duration}")
print(f"actual_end_time={solution.t[-1]}")
print(f"saved_frames={len(solution.frames)}")
print(f"finite_state_arrays={finite}")
if solution.error:
    print(solution.error)
