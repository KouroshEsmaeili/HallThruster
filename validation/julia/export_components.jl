#!/usr/bin/env julia

# Export deterministic validation cases from HallThruster.jl@014a12f.
# Run this with a Julia 1.10 environment whose active project is an extracted
# checkout of that exact commit; see validation/README.md.

using HallThruster
using JSON3
using LinearAlgebra

const ht = HallThruster
const REFERENCE_COMMIT = "014a12fb193af6927cb10f77da5e7baf215b5bc0"

vector_data(value) = collect(value)
matrix_data(value) = [collect(@view value[row, :]) for row in axes(value, 1)]
finite_data(value) = isnan(value) ? "NaN" : isinf(value) ? (value > 0 ? "Inf" : "-Inf") : value

function validation_config(; kwargs...)
    defaults = (;
        thruster = ht.SPT_100,
        domain = (0.0, 0.08),
        discharge_voltage = 300.0,
        anode_mass_flow_rate = 5e-6,
        neutral_temperature_K = 500.0,
    )
    options = merge(defaults, (; kwargs...))
    return ht.Config(; options...)
end

function grid_data(spec)
    grid = ht.generate_grid(spec, ht.SPT_100.geometry, (0.0, 0.08))
    return Dict(
        "num_cells" => grid.num_cells,
        "edges" => vector_data(grid.edges),
        "cell_centers" => vector_data(grid.cell_centers),
        "dz_edge" => vector_data(grid.dz_edge),
        "dz_cell" => vector_data(grid.dz_cell),
    )
end

function reaction_data()
    energies = [0.0, 1.0, 5.5, 10.0, 20.0, 50.0, 100.0, 255.0]
    output = Dict{String, Any}()
    cases = [
        ("elastic", nothing),
        ("excitation", nothing),
        ("ionization", ht.Xenon(1)),
    ]
    for (reaction_type, product) in cases
        threshold, coefficients = ht.load_rate_coeffs(
            ht.Xenon(0), product, reaction_type,
        )
        reaction = if reaction_type == "elastic"
            ht.ElasticCollision(ht.Xenon(0), coefficients)
        elseif reaction_type == "excitation"
            ht.ExcitationReaction(threshold, ht.Xenon(0), coefficients)
        else
            ht.IonizationReaction(threshold, ht.Xenon(0), product, coefficients)
        end
        output[reaction_type] = Dict(
            "threshold" => threshold,
            "energies" => energies,
            "coefficients" => [ht.rate_coeff(reaction, energy) for energy in energies],
        )
    end
    return output
end

function config_data()
    config = validation_config(; ncharge = 3)
    return Dict(
        "thruster" => config.thruster.name,
        "propellant" => repr(config.propellant),
        "ncharge" => config.ncharge,
        "domain" => collect(config.domain),
        "discharge_voltage" => config.discharge_voltage,
        "anode_mass_flow_rate" => config.anode_mass_flow_rate,
        "cathode_coupling_voltage" => config.cathode_coupling_voltage,
        "anode_Tev" => config.anode_Tev,
        "cathode_Tev" => config.cathode_Tev,
        "neutral_velocity" => config.neutral_velocity,
        "neutral_temperature_K" => config.neutral_temperature_K,
        "ion_temperature_K" => config.ion_temperature_K,
        "background_pressure_Torr" => config.background_pressure_Torr,
        "background_temperature_K" => config.background_temperature_K,
        "transition_length" => config.transition_length,
        "solve_plume" => config.solve_plume,
        "ion_wall_losses" => config.ion_wall_losses,
        "electron_ion_collisions" => config.electron_ion_collisions,
        "anom_model" => string(nameof(typeof(config.anom_model))),
        "wall_loss_model" => string(nameof(typeof(config.wall_loss_model))),
        "conductivity_model" => string(nameof(typeof(config.conductivity_model))),
        "ionization_model" => string(config.ionization_model),
        "excitation_model" => string(config.excitation_model),
        "electron_neutral_model" => string(config.electron_neutral_model),
        "source_neutrals_zero" => config.source_neutrals(nothing, nothing, 1),
        "source_energy_zero" => config.source_energy(nothing, 1),
        "source_ion_continuity_zero" => [source(nothing, nothing, 1) for source in config.source_ion_continuity],
        "source_ion_momentum_zero" => [source(nothing, nothing, 1) for source in config.source_ion_momentum],
    )
end

function index_data()
    output = Dict{String, Any}()
    for ncharge in (1, 2, 3)
        fluids, fluid_ranges, species, species_ranges, is_velocity = ht.configure_fluids(
            validation_config(; ncharge),
        )
        index = ht.configure_index(fluids, fluid_ranges)
        output[string(ncharge)] = Dict(
            "species" => repr.(species),
            "fluid_ranges_one_based_inclusive" => [[first(item), last(item)] for item in fluid_ranges],
            "species_ranges_one_based_inclusive" => Dict(
                string(key) => [first(value), last(value)] for (key, value) in species_ranges
            ),
            "neutral" => index.ρn,
            "ion_density" => Dict(string(charge) => index.ρi[charge] for charge in 1:ncharge),
            "ion_momentum" => Dict(string(charge) => index.ρiui[charge] for charge in 1:ncharge),
            "velocity_mask" => vector_data(is_velocity),
        )
    end
    return output
end

function initialization_data()
    config = validation_config(;
        discharge_voltage = 500.0,
        anode_mass_flow_rate = 3e-6,
        ncharge = 3,
        cathode_Tev = 5.0,
        anode_Tev = 3.0,
        neutral_velocity = 300.0,
        neutral_temperature_K = 100.0,
        ion_temperature_K = 300.0,
        initial_condition = ht.DefaultInitialization(max_electron_temperature = 10.0),
    )
    grid = ht.generate_grid(ht.EvenGrid(4), config.thruster.geometry, config.domain)
    fluids, fluid_ranges, _, _, _ = ht.configure_fluids(config)
    index = ht.configure_index(fluids, fluid_ranges)
    state, cache = ht.allocate_arrays(grid, config)
    params = (;
        ht.params_from_config(config)...,
        grid,
        index,
        cache,
        min_Te = min(config.anode_Tev, config.cathode_Tev),
    )
    ht.initialize!(state, params, config)
    return Dict(
        "cell_centers" => vector_data(grid.cell_centers),
        "state" => matrix_data(state),
        "electron_density" => vector_data(cache.ne),
        "electron_energy_density" => vector_data(cache.nϵ),
        "electron_temperature_ev" => vector_data(cache.Tev),
    )
end

function setup_data()
    config = validation_config()
    sim = ht.SimParams(;
        grid = ht.EvenGrid(20), dt = 1e-8, duration = 1e-7, num_save = 3,
        adaptive = false, verbose = false, print_errors = true,
    )
    state, params = ht.setup_simulation(config, sim)
    cache = params.cache
    return Dict(
        "state" => matrix_data(state),
        "cell_centers" => vector_data(params.grid.cell_centers),
        "edges" => vector_data(params.grid.edges),
        "magnetic_field" => vector_data(cache.B),
        "neutral_density" => vector_data(cache.nn),
        "ion_density" => matrix_data(cache.ni),
        "ion_flux" => matrix_data(cache.niui),
        "electron_density" => vector_data(cache.ne),
        "electron_temperature_ev" => vector_data(cache.Tev),
        "electron_energy_density" => vector_data(cache.nϵ),
        "potential" => vector_data(cache.ϕ),
        "potential_gradient" => vector_data(cache.∇ϕ),
        "channel_area" => vector_data(cache.channel_area),
        "inner_radius" => vector_data(cache.inner_radius),
        "outer_radius" => vector_data(cache.outer_radius),
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
        "∇ϕ" => vector_data(frame.∇ϕ),
        "Id" => vector_data(frame.Id),
        "νiz" => vector_data(frame.νiz),
        "νc" => vector_data(frame.νc),
    )
end

function short_simulation_data()
    config = validation_config()
    sim = ht.SimParams(;
        grid = ht.EvenGrid(20), dt = 1e-8, duration = 1e-7, num_save = 3,
        adaptive = false, verbose = false, print_errors = true,
    )
    solution = ht.run_simulation(config, sim)
    return Dict(
        "retcode" => string(solution.retcode),
        "times" => vector_data(solution.t),
        "frames" => [frame_data(frame) for frame in solution.frames],
    )
end

function collect_data()
    geometry = ht.SPT_100.geometry
    magnetic = ht.LinearInterpolation(ht.SPT_100.magnetic_field.z, ht.SPT_100.magnetic_field.B)
    fd_points = (0.0, 0.5, 2.0)
    fd_values = fd_points .^ 2
    matrix = Tridiagonal([1.0, -1.0, 2.0], [4.0, 5.0, 6.0, 7.0], [2.0, 3.0, -2.0])
    expected_solution = [1.0, -2.0, 3.0, 0.5]
    rhs = matrix * expected_solution
    ratios = [-2.0, 0.0, 0.25, 1.0, 10.0, Inf, NaN]
    fluid = ht.Fluid(ht.Xenon(1))
    left = (1.0, 300.0, ht.Xenon.cv * 300.0 + 0.5 * 300.0^2)
    right = (0.5, 50.0, 0.5 * (ht.Xenon.cv * 600.0 + 0.5 * 100.0^2))

    return Dict(
        "meta" => Dict("language" => "julia", "julia_reference_commit" => REFERENCE_COMMIT),
        "physical_constants" => Dict("e" => ht.e, "me" => ht.me, "kB" => ht.kB, "NA" => ht.NA, "R0" => ht.R0),
        "gas_species" => Dict(
            "name" => repr(ht.Xenon), "M" => ht.Xenon.M, "m" => ht.Xenon.m,
            "R" => ht.Xenon.R, "cp" => ht.Xenon.cp, "cv" => ht.Xenon.cv,
            "gamma" => ht.Xenon.γ,
            "species" => repr.([ht.Xenon(charge) for charge in (0, 1, 2, 3)]),
        ),
        "geometry" => Dict(
            "channel_length" => geometry.channel_length,
            "inner_radius" => geometry.inner_radius,
            "outer_radius" => geometry.outer_radius,
            "channel_area" => geometry.channel_area,
            "channel_width" => ht.channel_width(geometry),
            "channel_perimeter" => ht.channel_perimeter(geometry),
        ),
        "even_grid" => Dict(string(count) => grid_data(ht.EvenGrid(count)) for count in (4, 10, 20)),
        "uneven_grid" => Dict(string(count) => grid_data(ht.UnevenGrid(count)) for count in (4, 10, 20)),
        "interpolation" => Dict(
            "x" => [-1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 9.0],
            "y" => [ht.LinearInterpolation([0.0, 2.0, 5.0], [0.0, 4.0, 10.0])(x)
                    for x in (-1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 9.0)],
        ),
        "finite_differences" => Dict(
            "forward_coefficients" => collect(ht.forward_diff_coeffs(fd_points...)),
            "central_coefficients" => collect(ht.central_diff_coeffs(fd_points...)),
            "backward_coefficients" => collect(ht.backward_diff_coeffs(fd_points...)),
            "second_coefficients" => collect(ht.second_deriv_coeffs(fd_points...)),
            "upwind_coefficients" => collect(ht.upwind_diff_coeffs(fd_points...)),
            "downwind_coefficients" => collect(ht.downwind_diff_coeffs(fd_points...)),
            "forward_quadratic" => ht.forward_difference(fd_values..., fd_points...),
            "central_quadratic" => ht.central_difference(fd_values..., fd_points...),
            "backward_quadratic" => ht.backward_difference(fd_values..., fd_points...),
            "second_quadratic" => ht.second_deriv_central_diff(fd_values..., fd_points...),
        ),
        "integration" => vector_data(ht.cumtrapz([0.0, 0.5, 2.0, 3.0], [1.0, 2.0, -1.0, 4.0], 3.0)),
        "linear_algebra" => Dict("rhs" => vector_data(rhs), "solution" => vector_data(ht.tridiagonal_solve(matrix, rhs))),
        "limiters" => Dict(
            string(name) => Dict(
                "ratios" => finite_data.(ratios),
                "values" => finite_data.([limiter(ratio) for ratio in ratios]),
            ) for (name, limiter) in pairs(ht.slope_limiters)
        ),
        "flux_functions" => Dict(
            "left_state" => collect(left),
            "right_state" => collect(right),
            "physical_left" => collect(ht.flux(left, fluid)),
            "physical_right" => collect(ht.flux(right, fluid)),
            "rusanov" => collect(ht.rusanov(left, right, fluid)),
            "global_lax_friedrichs" => collect(ht.global_lax_friedrichs(left, right, fluid, 500.0)),
            "hlle" => collect(ht.HLLE(left, right, fluid)),
        ),
        "reaction_tables" => reaction_data(),
        "magnetic_field" => Dict(
            "positions" => [0.0, 0.0125, 0.025, 0.05, 0.08, 0.1],
            "values" => [magnetic(z) for z in (0.0, 0.0125, 0.025, 0.05, 0.08, 0.1)],
        ),
        "config_defaults" => config_data(),
        "fluid_index_mapping" => index_data(),
        "initialization" => initialization_data(),
        "setup" => setup_data(),
        "short_simulation" => short_simulation_data(),
    )
end

if isempty(ARGS)
    error("usage: julia export_components.jl OUTPUT.json")
end

output_path = abspath(ARGS[1])
mkpath(dirname(output_path))
open(output_path, "w") do io
    write(io, JSON3.write(collect_data()))
    write(io, '\n')
end
