import copy
import numpy as np
import time, traceback
from ..physics.physicalconstants import e, me
from .update_heavy_species import integrate_heavy_species_stage
from ..collisions.anomalous import num_anom_variables


class Solution:

    def __init__(self, t, frames, params, config, retcode, error):
        self.t = list(t)
        self.frames = list(frames)
        self.params = params
        self.config = config
        self.retcode = retcode
        self.error = error

    def __str__(self):
        num_frames = len(self.frames)
        retcode_str = str(self.retcode)
        if len(self.t) == 0:
            end_time = 0.0
        else:
            end_time = self.t[-1]
        plural = "" if num_frames == 1 else "s"
        return (f"Hall thruster solution with {num_frames} saved frame{plural} "
                f"(retcode: {retcode_str}, end time: {end_time} seconds)")

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_solution_by_frame(key)

        if isinstance(key, slice) or isinstance(key, list):
            return self._get_solution_by_slice(key)

        if isinstance(key, tuple) and len(key) == 2:
            field, charge = key
            if isinstance(field, str):
                # interpret it as symbol
                return self._get_ion_quantity(field, charge)
            else:
                raise ValueError("If indexing with a tuple, the first must be a string (field).")

        if isinstance(key, str):
            return self._get_field(key)

        raise KeyError(f"Invalid index {key} for Solution.")

    def _get_solution_by_frame(self, frame):
        # If your code is 1-based in Julia, you might do frame-1 here. We'll assume Python is 0-based.
        if frame < 0 or frame >= len(self.frames):
            raise IndexError("Frame index out of range.")
        new_t = [self.t[frame]]
        new_frames = [self.frames[frame]]
        return Solution(new_t, new_frames,
                        self.params, self.config,
                        self.retcode, self.error)

    def _get_solution_by_slice(self, frames):
        new_t = [self.t[i] for i in range(len(self.t))][frames]
        new_frames = [self.frames[i] for i in range(len(self.frames))][frames]
        return Solution(new_t, new_frames,
                        self.params, self.config,
                        self.retcode, self.error)

    def _get_field(self, field):
        # 1) transform alternate name if needed
        alts = alternate_field_names()
        if field in alts:
            field = alts[field]  # e.g. "potential" => "ϕ"

        # 2) check if field in saved_fields => return [frame[field] for frame in frames]
        if field in saved_fields():
            return [getattr(frm, field) for frm in self.frames]

        # 3) special cases
        if field == "B":
            return self.params.cache.B  # e.g. a 1D array
        elif field in ["ωce", "cyclotron_freq", "omega_ce"]:
            B_arr = self.params.cache.B
            return [e * bval / me for bval in B_arr]
        elif field == "E":
            grad_phi_list = self._get_field("∇ϕ")

            return [[-val for val in frame_vals] for frame_vals in grad_phi_list]
        elif field == "z":
            # e.g. cell centers
            return self.params['grid'].cell_centers

        # If none matched:
        raise ValueError(f"Field :{field} not found! Valid fields are {valid_fields()}")

    def _get_ion_quantity(self, field, charge):
        if field not in _saved_fields_matrix():
            raise ValueError("Indexing a solution by [field, charge] only for ion quantities. "
                             f"Given {field} not in {list(_saved_fields_matrix())}.")

        if charge <= 0 or charge > self.config.ncharge:
            raise ValueError(f"No ions of charge state {charge}. Maximum is {self.config.ncharge}.")

        out = []
        for frm in self.frames:
            mat = getattr(frm, field)  # shape [ncharge, ncells]

            row = mat[charge - 1, :] if hasattr(mat, "__getitem__") else None
            out.append(row)
        return out


def _saved_fields_vector():
    return ("μ", "Tev", "ϕ", "∇ϕ", "ne", "pe", "ue", "∇pe", "νan", "νc", "νen",
            "νei", "radial_loss_frequency", "νew_momentum", "νiz", "νex",
            "νe", "Id", "ji", "nn", "anom_multiplier", "ohmic_heating", "wall_losses",
            "inelastic_losses", "Vs", "channel_area", "inner_radius", "outer_radius",
            "dA_dz", "tanδ", "anom_variables", "dt")


def _saved_fields_matrix():
    return ("ni", "ui", "niui")


def saved_fields():
    return _saved_fields_vector() + _saved_fields_matrix()


def alternate_field_names():
    return {
        "mobility": "μ",
        "potential": "ϕ",
        "thermal_conductivity": "κ",
        "grad_pe": "∇pe",
        "nu_anom": "νan",
        "nu_class": "νc",
        "nu_wall": "νew_momentum",
        "nu_ei": "νei",
        "nu_en": "νen",
        "nu_iz": "νiz",
        "nu_ex": "νex",
        "tan_divergence_angle": "tanδ",
    }


def valid_fields():
    special = ("z", "B", "E", "E", "ωce", "cyclotron_freq")
    # note that "E" appears twice in the snippet, might be a small duplication
    # We'll unify them or keep them. We'll keep them here for fidelity:
    return special + saved_fields() + tuple(alternate_field_names().keys())


def solve(U, params, config, tspan, saveat):

    # Initialize
    iteration = params["iteration"]
    iteration[0] = 1
    t = tspan[0]
    yield_interval = 100
    errstring = ""
    retcode = "success"  # or :success in Julia

    fields_to_save = saved_fields()
    # first_saveval => NamedTuple{fields_to_save}(params.cache) in Julia
    # We'll do a dictionary or copy the relevant fields. For demonstration, we'll do a shallow approach:
    first_saveval = {field: params["cache"][field] for field in fields_to_save if field in params["cache"]}
    frames = [copy.deepcopy(first_saveval) for _ in saveat]  # one for each save time

    small_step_count = 0
    uniform_steps = False
    sim = params["simulation"]  # or sim = params.get("simulation")
    # user-provided sources from config
    source_neutrals = config.source_neutrals
    source_ion_continuity = config.source_ion_continuity
    source_ion_momentum = config.source_ion_momentum
    scheme = config.scheme
    sources = {"source_neutrals": source_neutrals,
               "source_ion_continuity": source_ion_continuity,
               "source_ion_momentum": source_ion_momentum}
    save_ind = 1  # Python 0-indexing: we start saving at the second save time index
    num_save = len(saveat)
    start_time = time.time()
    try:
        while t < tspan[1]:
            if sim.adaptive:
                # print('flag1')
                print('[solve] t, tspan[1]:',t, tspan[1])
                if uniform_steps:
                    params["dt"][0] = sim.dt
                    small_step_count -= 1
                else:
                    dt_val = params["cache"]["dt"][0]
                    params["dt"][0] = max(sim.min_dt, min(dt_val, sim.max_dt))

            # print('[solve] params["dt"]',params["dt"])
            # cyrus was here
            # t += params["dt"]*100
            t += params["dt"]

            # count how many times we've used min_dt
            if params["dt"][0] == sim.min_dt:
                small_step_count += 1
            elif not uniform_steps:
                small_step_count = 0

            if small_step_count >= sim.max_small_steps:
                uniform_steps = True
            elif small_step_count == 0:
                uniform_steps = False

            # update heavy species
            # e.g.: integrate_heavy_species(U, params, scheme, sources, dt_current)
            # update_heavy_species(U, params)
            # placeholder:
            # if any not isfinite in U => fail
            from ..simulation.update_heavy_species import integrate_heavy_species, update_heavy_species
            # print('[solve]scheme',scheme)
            # print("[solve] U", U)
            integrate_heavy_species_stage(U, params, scheme, sources, params["dt"][0])
            # print("[solve] U", U)

            update_heavy_species(U, params)
            # print("[solve] U", U)


            if not np.all(np.isfinite(U)):
                if sim.print_errors:
                    print(f"Warning: NaN or Inf detected in heavy species solver at time {t}")
                retcode = "failure"
                break
            from ..simulation.update_electrons import update_electrons
            update_electrons(params, config, t)

            # update plume geometry if needed
            # if config["solve_plume"]:
            #    update_plume_geometry(params)

            # update electrons
            # update_electrons(params, config, t)
            # Update plume geometry if configured.
            if config.solve_plume:
                from ..simulation.plume import update_plume_geometry
                update_plume_geometry(params)

            iteration[0] += 1

            if iteration[0] % yield_interval == 0:
                # Yield control if needed.
                time.sleep(0)

            if save_ind < num_save and t > saveat[save_ind]:
                # Save vector fields.
                for field in _saved_fields_vector():
                    if field == "anom_variables":
                        for i in range(num_anom_variables(config.anom_model)):
                            frames[save_ind][field][i] = params["cache"][field][i]
                    else:
                        frames[save_ind][field] = np.copy(params["cache"][field])
                # Save matrix fields.
                for field in _saved_fields_matrix():
                    frames[save_ind][field] = np.copy(params["cache"][field])
                save_ind += 1

    except Exception as e:
        errstring = traceback.format_exc()
        retcode = "error"
        if sim.print_errors:
            # print("flag")
            print("Warning: Error detected in solution:")
            print(errstring)


    # Cyrus was here total_time if needed!!
    # total_time = time.time() - start_time

    ind = min(save_ind, num_save)
    # Create and return the Solution object.
    sol = Solution(saveat[:ind], frames[:ind], params, config, retcode, errstring)
    return sol
