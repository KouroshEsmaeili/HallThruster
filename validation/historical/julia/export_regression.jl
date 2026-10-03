#!/usr/bin/env julia

using HallThruster
using JSON3
using Statistics

const ht = HallThruster
const REFERENCE_COMMIT = "014a12fb193af6927cb10f77da5e7baf215b5bc0"
const CHECKPOINT_TARGETS = [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]

vector_data(x) = collect(x)
matrix_data(x) = [collect(row) for row in eachrow(x)]

function option(name, default, convert)
    index = findfirst(==(name), ARGS)
    index === nothing && return default
    index == length(ARGS) && error("missing value after $name")
    return convert(ARGS[index + 1])
end

function normalized_config(config, sim)
    geometry = config.thruster.geometry
    anom = config.anom_model
    gaussian = anom.model
    wall = config.wall_loss_model
    scheme = config.scheme
    flux_name = scheme.flux_function === ht.rusanov ? "rusanov" : string(scheme.flux_function)
    limiter_name = scheme.limiter === ht.van_leer ? "van_leer" : string(scheme.limiter)
    return Dict(
        "ncharge" => config.ncharge,
        "propellant" => repr(config.propellant),
        "thruster" => Dict(
            "name" => config.thruster.name,
            "shielded" => config.thruster.shielded,
            "geometry" => Dict(
                "channel_length" => geometry.channel_length,
                "inner_radius" => geometry.inner_radius,
                "outer_radius" => geometry.outer_radius,
                "channel_area" => geometry.channel_area,
            ),
            "magnetic_field_file" => config.thruster.magnetic_field.file,
        ),
        "domain" => collect(config.domain),
        "anode_mass_flow_rate" => config.anode_mass_flow_rate,
        "discharge_voltage" => config.discharge_voltage,
        "cathode_coupling_voltage" => config.cathode_coupling_voltage,
        "neutral_velocity" => config.neutral_velocity,
        "neutral_temperature_K" => config.neutral_temperature_K,
        "ion_temperature_K" => config.ion_temperature_K,
        "anode_Tev" => config.anode_Tev,
        "cathode_Tev" => config.cathode_Tev,
        "background_pressure_Torr" => config.background_pressure_Torr,
        "background_temperature_K" => config.background_temperature_K,
        "transition_length" => config.transition_length,
        "neutral_ingestion_multiplier" => config.neutral_ingestion_multiplier,
        "solve_plume" => config.solve_plume,
        "apply_thrust_divergence_correction" => config.apply_thrust_divergence_correction,
        "ion_wall_losses" => config.ion_wall_losses,
        "electron_ion_collisions" => config.electron_ion_collisions,
        "electron_plume_loss_scale" => config.electron_plume_loss_scale,
        "magnetic_field_scale" => config.magnetic_field_scale,
        "anom_smoothing_iters" => config.anom_smoothing_iters,
        "anom_model" => Dict(
            "type" => "LogisticPressureShift",
            "z0" => anom.z0,
            "dz" => anom.dz,
            "pstar" => anom.pstar,
            "alpha" => anom.alpha,
            "model" => Dict(
                "type" => "GaussianBohm",
                "hall_min" => gaussian.hall_min,
                "hall_max" => gaussian.hall_max,
                "center" => gaussian.center,
                "width" => gaussian.width,
            ),
        ),
        "wall_loss_model" => Dict(
            "type" => "WallSheath",
            "material" => wall.material.name,
            "loss_scale" => wall.loss_scale,
        ),
        "scheme" => Dict(
            "flux_function" => flux_name,
            "limiter" => limiter_name,
            "reconstruct" => scheme.reconstruct,
        ),
        "ionization_model" => string(config.ionization_model),
        "excitation_model" => string(config.excitation_model),
        "electron_neutral_model" => string(config.electron_neutral_model),
        "simulation" => Dict(
            "grid_type" => "EvenGrid",
            "num_cells" => sim.grid.num_cells,
            "dt" => sim.dt,
            "adaptive" => sim.adaptive,
            "CFL" => sim.CFL,
            "min_dt" => sim.min_dt,
            "max_dt" => sim.max_dt,
            "max_small_steps" => sim.max_small_steps,
            "duration" => sim.duration,
            "num_save" => sim.num_save,
        ),
    )
end

function stored_baseline()
    return Dict(
        "thrust_mN" => 87.304,
        "discharge_current_A" => 4.614,
        "ion_current_A" => 3.922,
        "max_electron_temperature_eV" => 24.832,
        "max_electric_field_V_per_m" => 6.665e4,
        "max_neutral_density_per_m3" => 2.088e19,
        "max_ion_density_per_m3" => 9.651e17,
        "mass_efficiency" => 0.954,
        "current_efficiency" => 0.873,
        "divergence_efficiency" => 0.949,
        "voltage_efficiency" => 0.661,
        "anode_efficiency" => 0.5656,
    )
end

function frame_data(frame)
    return Dict(
        "nn" => vector_data(frame.nn),
        "ni" => matrix_data(frame.ni),
        "niui" => matrix_data(frame.niui),
        "ui" => matrix_data(frame.ui),
        "ne" => vector_data(frame.ne),
        "Tev" => vector_data(frame.Tev),
        "potential" => vector_data(frame.ϕ),
        "electric_field" => vector_data(-frame.∇ϕ),
        "nu_ionization" => vector_data(frame.νiz),
        "nu_collisions" => vector_data(frame.νc),
    )
end

function scalar_metrics(sol)
    nsave = length(sol.frames)
    avg_start = nsave ÷ 3
    n_avg = nsave - avg_start
    tail = avg_start:nsave
    thrust_values = ht.thrust(sol) .* 1000
    discharge_values = ht.discharge_current(sol)
    ion_values = ht.ion_current(sol)
    mass_values = ht.mass_eff(sol)
    current_values = ht.current_eff(sol)
    divergence_values = ht.divergence_eff(sol)
    voltage_values = ht.voltage_eff(sol)
    anode_values = ht.anode_eff(sol)
    averaged = ht.time_average(sol, avg_start)
    averaged_frame = averaged.frames[1]

    stderr(values) = std(values) / sqrt(n_avg)
    return Dict(
        "avg_start_frame_one_based" => avg_start,
        "averaged_frame_count" => length(tail),
        "historical_standard_error_denominator_count" => n_avg,
        "thrust_mN" => Dict("value" => mean(thrust_values[tail]), "standard_error" => stderr(thrust_values[tail])),
        "discharge_current_A" => Dict("value" => mean(discharge_values[tail]), "standard_error" => stderr(discharge_values[tail])),
        "ion_current_A" => Dict("value" => mean(ion_values[tail]), "standard_error" => stderr(ion_values[tail])),
        "mass_efficiency" => Dict("value" => mean(mass_values), "standard_error" => stderr(mass_values)),
        "current_efficiency" => Dict("value" => mean(current_values), "standard_error" => stderr(current_values)),
        "divergence_efficiency" => Dict("value" => mean(divergence_values), "standard_error" => stderr(divergence_values)),
        "voltage_efficiency" => Dict("value" => mean(voltage_values), "standard_error" => stderr(voltage_values)),
        "anode_efficiency" => Dict("value" => mean(anode_values), "standard_error" => stderr(anode_values)),
        "max_electron_temperature_eV" => Dict("value" => maximum(averaged_frame.Tev)),
        "max_electric_field_V_per_m" => Dict("value" => maximum(-averaged_frame.∇ϕ)),
        "max_neutral_density_per_m3" => Dict("value" => maximum(averaged_frame.nn)),
        "max_ion_density_per_m3" => Dict("value" => maximum(averaged_frame.ni)),
    )
end

function collect_run(sol, runtime_seconds)
    nsave = length(sol.frames)
    avg_start = nsave ÷ 3
    averaged = ht.time_average(sol, avg_start).frames[1]
    checkpoints = Any[]
    for target in CHECKPOINT_TARGETS
        index = argmin(abs.(sol.t .- target))
        push!(checkpoints, Dict(
            "target_time" => target,
            "frame_index_zero_based" => index - 1,
            "time" => sol.t[index],
            "fields" => frame_data(sol.frames[index]),
            "discharge_current_A" => ht.discharge_current(sol, index),
            "ion_current_A" => ht.ion_current(sol, index),
        ))
    end
    saved_dt = [frame.dt[] for frame in sol.frames]
    return Dict(
        "status" => Dict(
            "retcode" => string(sol.retcode),
            "error" => sol.error,
            "reported_final_time" => sol.t[end],
            "saved_frames" => nsave,
            "accepted_steps" => sol.params.iteration[] - 1,
            "runtime_seconds" => runtime_seconds,
            "warnings" => String[],
        ),
        "adaptive" => Dict(
            "initial_requested_dt" => sol.params.simulation.dt,
            "initial_internal_dt" => 100 * eps(),
            "minimum_saved_dt" => minimum(saved_dt),
            "maximum_saved_dt" => maximum(saved_dt),
            "saved_dt" => saved_dt,
            "accepted_steps" => sol.params.iteration[] - 1,
            "save_times" => vector_data(sol.t),
        ),
        "grid" => Dict(
            "z" => vector_data(sol.params.grid.cell_centers),
            "magnetic_field" => vector_data(sol.params.cache.B),
        ),
        "metrics" => scalar_metrics(sol),
        "histories" => Dict(
            "time" => vector_data(sol.t),
            "dt" => saved_dt,
            "discharge_current_A" => ht.discharge_current(sol),
            "ion_current_A" => ht.ion_current(sol),
            "thrust_mN" => ht.thrust(sol) .* 1000,
        ),
        "checkpoints" => checkpoints,
        "averaged_profiles" => Dict(
            "nn" => vector_data(averaged.nn),
            "ni" => matrix_data(averaged.ni),
            "niui" => matrix_data(averaged.niui),
            "ui" => matrix_data(averaged.ui),
            "ne" => vector_data(averaged.ne),
            "Tev" => vector_data(averaged.Tev),
            "potential" => vector_data(averaged.ϕ),
            "electric_field" => vector_data(-averaged.∇ϕ),
            "magnetic_field" => vector_data(sol.params.cache.B),
        ),
    )
end

function summary(run)
    return Dict(
        "status" => run["status"],
        "adaptive" => Dict(
            "minimum_saved_dt" => run["adaptive"]["minimum_saved_dt"],
            "maximum_saved_dt" => run["adaptive"]["maximum_saved_dt"],
            "accepted_steps" => run["adaptive"]["accepted_steps"],
        ),
        "metrics" => run["metrics"],
    )
end

isempty(ARGS) && error("usage: export_regression.jl OUTPUT.json [options]")
output_path = abspath(ARGS[1])
baseline_file = abspath(option("--baseline", joinpath(ht.TEST_DIR, "regression", "baseline.json"), String))
repeat_count = option("--repeat", 2, x -> parse(Int, x))
duration_override = option("--duration", nothing, x -> parse(Float64, x))
num_save_override = option("--num-save", nothing, x -> parse(Int, x))
num_cells_override = option("--num-cells", nothing, x -> parse(Int, x))

input = JSON3.read(read(baseline_file))
config = ht.deserialize(ht.Config, input.config)
base_sim = ht.deserialize(ht.SimParams, input.simulation)
sim = ht.SimParams(
    grid = isnothing(num_cells_override) ? base_sim.grid : ht.EvenGrid(num_cells_override),
    dt = base_sim.dt,
    duration = isnothing(duration_override) ? base_sim.duration : duration_override,
    num_save = isnothing(num_save_override) ? base_sim.num_save : num_save_override,
    verbose = base_sim.verbose,
    print_errors = base_sim.print_errors,
    adaptive = base_sim.adaptive,
    CFL = base_sim.CFL,
    min_dt = base_sim.min_dt,
    max_dt = base_sim.max_dt,
    max_small_steps = base_sim.max_small_steps,
    current_control = base_sim.current_control,
)

repeat_summaries = Any[]
final_run = nothing
for run_index in 1:repeat_count
    timed = @timed ht.run_simulation(config, sim; include_dirs = [dirname(baseline_file)])
    run = collect_run(timed.value, timed.time)
    push!(repeat_summaries, summary(run))
    global final_run = run
    timed = nothing
    run_index < repeat_count && GC.gc()
end

payload = Dict(
    "metadata" => Dict(
        "language" => "julia",
        "reference_commit" => REFERENCE_COMMIT,
        "hallthruster_version" => string(pkgversion(HallThruster)),
        "julia_version" => string(VERSION),
        "baseline_file" => baseline_file,
        "repeat_count" => repeat_count,
    ),
    "configuration" => normalized_config(config, sim),
    "stored_baseline" => stored_baseline(),
    "repeatability" => repeat_summaries,
    "run" => final_run,
)

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.write(io, payload; allow_inf = true)
    write(io, '\n')
end
