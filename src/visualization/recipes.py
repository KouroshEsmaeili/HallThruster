import numpy as np
import matplotlib.pyplot as plt
from ..collisions.reactions import rate_coeff


def get_alternate_field_names():
    # Alternate names for fields with special characters.
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


def saved_fields():
    # Returns a tuple of field names that are saved per frame.
    vector_fields = ("μ", "Tev", "ϕ", "∇ϕ", "ne", "pe", "ue", "∇pe",
                     "νan", "νc", "νen", "νei", "radial_loss_frequency",
                     "νew_momentum", "νiz", "νex", "νe", "Id", "ji", "nn",
                     "anom_multiplier", "ohmic_heating", "wall_losses",
                     "inelastic_losses", "Vs", "channel_area", "inner_radius",
                     "outer_radius", "dA_dz", "tanδ", "anom_variables", "dt")
    matrix_fields = ("ni", "ui", "niui")
    return vector_fields + matrix_fields


def valid_fields():
    # Returns a tuple of valid fields for indexing a solution.
    alts = tuple(get_alternate_field_names().keys())
    return ("z", "B", "E") + saved_fields() + alts + ("E", "ωce", "cyclotron_freq")




def plot_solution(sol, frame=None, label_user=""):
    # Default to last frame if not provided.
    if frame is None:
        frame = len(sol.frames) - 1
    # Extract frame data (assume frames is a list of dicts).
    fr = sol.frames[frame]

    # Extract common parameters.
    z_cell = np.array(sol.params["z_cell"],dtype=np.float64)
    L_ch = sol.params["L_ch"]
    z_normalized = z_cell / L_ch

    ncharge = sol.config.get("ncharge", 1)

    # Generate charge labels; assume sol.config["propellant"] is callable.
    try:
        charge_labels = [str(sol.config["propellant"](Z-1)) for Z in range(1, ncharge + 1)]
    except Exception:
        charge_labels = [f"Ion {Z}" for Z in range(1, ncharge + 1)]

    # Set up figure layout.
    subplot_width = 500  # in pixels
    subplot_height = 500
    nrows, ncols = 2, 4
    # Assume 100 dpi, so figure size in inches.
    fig_width = (subplot_width * ncols) / 100
    fig_height = (subplot_height * nrows) / 100
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axs = axs.flatten()  # Flatten to 1D array for easier indexing.

    # Common plotting attributes.
    lw = 2
    # Use identity y-scale (i.e. linear).

    # Subplot 1: Neutral density.
    axs[0].plot(z_normalized, fr.get("nn", np.full_like(z_normalized, np.nan)),
                label=f"{label_user}, Total" if label_user else "Total", color="C0", linewidth=lw)
    axs[0].set_ylabel("Density (m⁻³)")
    axs[0].set_title("Neutral density")

    # Subplot 2: Plasma density.
    # Plot ion density for each charge state.
    print('[recipes] flag recipes')

    for Z in range(1, ncharge + 1):
        # Assume fr["ni"] is a 2D array with shape (ncharge, ncells) (0-indexed rows).
        ion_density = fr.get("ni", None)
        if ion_density is not None:
            axs[1].plot(z_normalized, ion_density[Z - 1, :],
                        label=f"{label_user}, {charge_labels[Z - 1]}", color=f"C{Z - 1}", linewidth=lw)
    # If more than one charge state, also plot electron density.
    if ncharge > 1:
        electron_density = fr.get("ne", None)
        if electron_density is not None:
            axs[1].plot(z_normalized, electron_density,
                        label=f"{label_user}, ne", color=f"C{ncharge}", linewidth=lw)
    axs[1].set_ylabel("Density (m⁻³)")
    axs[1].set_title("Plasma density")

    # Subplot 3: Ion velocity (km/s).
    for Z in range(1, ncharge + 1):
        ion_velocity = fr.get("ui", None)
        if ion_velocity is not None:
            # Assume ion_velocity is a 2D array with shape (ncharge, ncells)
            axs[2].plot(z_normalized, ion_velocity[Z - 1, :] / 1000,
                        label=f"{charge_labels[Z - 1]}", color=f"C{Z - 1}", linewidth=lw)
    axs[2].set_ylabel("Ion velocity (km/s)")
    axs[2].set_title("Ion velocity")
    axs[2].legend(loc="lower right")

    # Subplot 4: Potential.
    potential = fr.get("ϕ", None)
    if potential is not None:
        axs[3].plot(z_normalized, potential, label=label_user, color="C0", linewidth=lw)
    axs[3].set_ylabel("Potential (V)")
    axs[3].set_title("Potential")

    # Subplot 5: Ionization rate.
    # Compute ionization rate for each ion species.
    # Create an array of small epsilon values.
    ionization_rate = np.full_like(z_normalized, np.finfo(float).eps)
    ionization_reactions = sol.params.get("ionization_reactions", [])
    ne_arr = fr.get("ne", np.full_like(z_normalized, np.nan))
    Tev = fr.get("Tev", np.full_like(z_normalized, np.nan))
    # For each charge state, add contributions.
    for Z in range(1, ncharge + 1):
        # For each reaction in ionization_reactions:
        if not ionization_reactions:
            ionization_rate[:] = np.nan
        else:
            for rxn in ionization_reactions:
                # Assume rxn is a dict with keys "product" and "reactant", and these have attribute "Z".
                if rxn["product"].Z == Z:
                    if rxn["reactant"].Z == 0:
                        reactant_density = fr.get("nn", np.full_like(z_normalized, np.nan))
                    else:
                        # Use ion density for the appropriate charge state.
                        reactant_density = fr.get("ni", np.full((ncharge, len(z_normalized)), np.nan))[
                                           rxn["reactant"].Z - 1, :]
                    # Compute a rate coefficient array using the provided ionization model.
                    # Assume sol.params.config.ionization_model exists and rate_coeff is available.
                    rate_array = np.array(
                        [rate_coeff(sol.params.config["ionization_model"], rxn, 1.5 * T) for T in Tev],dtype=np.float64)
                    ionization_rate += reactant_density * ne_arr * rate_array
        # Plot ionization rate for this charge state on subplot 5.
        axs[4].plot(z_normalized, ionization_rate, label=f"{label_user}, {charge_labels[Z - 1]}", color=f"C{Z - 1}",
                    linewidth=lw)
    axs[4].set_ylabel("Ionization rate (m⁻³/s)")
    axs[4].set_title("Ionization rate")

    # Subplot 6: Electron temperature.
    electron_temperature = fr.get("Tev", None)
    if electron_temperature is not None:
        axs[5].plot(z_normalized, electron_temperature, label=label_user, color="C0", linewidth=lw)
    axs[5].set_ylabel("Electron temperature (eV)")
    axs[5].set_title("Electron temperature")

    # Subplot 7: Electron velocity.
    electron_velocity = fr.get("ue", None)
    if electron_velocity is not None:
        axs[6].plot(z_normalized, electron_velocity / 1000, label=label_user, color="C0", linewidth=lw)
    axs[6].set_ylabel("Electron velocity (km/s)")
    axs[6].set_title("Cross-field electron velocity")

    # Subplot 8: Electric field.
    electric_field = fr.get("∇ϕ", None)
    if electric_field is not None:
        axs[7].plot(z_normalized, -electric_field, label=label_user, color="C0", linewidth=lw)
    axs[7].set_ylabel("Electric field (V/m)")
    axs[7].set_title("Electric field")

    # Set common x-axis label for bottom row.
    for ax in axs[-4:]:
        ax.set_xlabel("z / L")

    # Adjust layout.
    fig.tight_layout()
    return fig, axs



def plot_collision(sol, frame=None, freqs=None):
    if frame is None:
        frame = 0  # Default to first frame.
    if freqs is None:
        freqs = ["ωce", "νan", "νen", "νei", "νiz", "νex", "νw"]

    # Set up plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale("log")
    ax.set_xlabel("z / L")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.set_title("Collision Frequencies")

    # Get y-limits from kwargs or use default.
    ylims = (1e5, 1e10)
    ax.set_ylim(ylims)

    # Compute normalized z from sol.params.
    z_cell = np.array(sol.params["z_cell"],dtype=np.float64)
    L_ch = sol.params["L_ch"]
    zs = z_cell / L_ch

    # Plot each frequency series.
    # For special key "ωce", plot electron cyclotron frequency.
    if "ωce" in freqs:
        # Assume sol["ωce"] returns a list of arrays; use the first array.
        ωce = sol["ωce"][0] if sol["ωce"] else np.full_like(zs, np.nan)
        ax.plot(zs, ωce, label="Electron cyclotron frequency", linestyle="--", linewidth=2)

    for key in freqs:
        if key == "ωce":
            continue
        data = sol[key][frame] if key in sol else None
        if data is None:
            continue
        # Replace zeros with NaN.
        data = np.where(data == 0, np.nan, data)
        # Use key as label.
        ax.plot(zs, data, label=key, linewidth=2)

    if "νe" in sol:
        total = sol["νe"][frame]
        ax.plot(zs, total, label="Total collision freq", color="black", linewidth=2)

    return fig, ax

