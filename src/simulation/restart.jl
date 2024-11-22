"""
    write_restart(path::AbstractString, sol)

Write a JLD2 restart file to `path``.

This can be reloaded to resume a simulation.
"""
function write_restart(path::AbstractString, sol)
    JLD2.save(path,
        Dict(
            "t" => sol.t,
            "u" => sol.u,
            "savevals" => sol.savevals,
            "params" => sol.params,
            "retcode" => sol.retcode,
        ),)
end

"""
    read_restart(path::AbstractString)

Load a JLD2 restart file from `path`.
"""
function read_restart(path::AbstractString)
    dict = JLD2.load(path)

    retcode = if haskey(dict, "retcode")
        dict["retcode"]
    else
        :Restart
    end

    return Solution(
        dict["t"], dict["u"], dict["savevals"], dict["params"], retcode, "",
    )
end

function load_restart(grid, config, path::AbstractString)
    sol = read_restart(path)
    U, cache = load_restart(grid, config, sol)
    return U, cache
end

function load_restart(grid, config, sol::Solution)
    U, cache = HallThruster.allocate_arrays(grid, config)
    z_cell = sol.params.grid.cell_centers

    # Interpolate neutrals
    itp = LinearInterpolation(z_cell, sol.u[end][1, :])
    U[1, :] = itp.(grid.cell_centers)

    # Interpolate ions
    for Z in 1:min(sol.params.config.ncharge, config.ncharge)
        ind = 2 * Z
        itp_density = LinearInterpolation(z_cell, sol.u[end][ind, :])
        itp_flux = LinearInterpolation(z_cell, sol.u[end][ind + 1, :])

        U[ind, :] .= itp_density.(grid.cell_centers)
        U[ind + 1, :] .= itp_flux.(grid.cell_centers)
    end

    # If we have more charges in the new solution than in the restart, initialize the other charges to default values
    initial_density = config.min_number_density * config.propellant.m
    initial_velocity = 0.0

    if config.ncharge > sol.params.config.ncharge
        for Z in (sol.params.onfig.ncharge + 1):(config.ncharge)
            ind = 2 * Z
            U[ind] .= initial_density
            U[ind + 1] .= initial_velocity
        end
    end

    # Interpolate electron energy
    itp = LinearInterpolation(z_cell, sol.u[end][end, :])
    U[end, :] = itp.(grid.cell_centers)

    sv = sol.savevals[end]
    # Interpolate cell-centered cache variables
    for field in fieldnames(typeof(sv))
        if field == :Id || field == :Vs || field == :anom_multiplier ||
           field == :anom_variables || field == :dt
            cache[field] .= sv[field]
        elseif field == :ni
            for Z in 1:(config.ncharge)
                @. @views cache[field][Z, :] = U[2 * Z, :] / config.propellant.m
            end
        elseif field == :ui
            for Z in 1:(config.ncharge)
                @. @views cache[field][Z, :] = U[2 * Z + 1, :] / U[2 * Z, :]
            end
        elseif field == :niui
            for Z in 1:(config.ncharge)
                @. @views cache[field][Z, :] = U[2 * Z + 1, :] / config.propellant.m
            end
        elseif field == :nn
            @. @views cache[field] = U[1, :] / config.propellant.m
        else
            itp = LinearInterpolation(z_cell, sv[field])
            cache[field] .= itp.(grid.cell_centers)
        end
    end

    return U, cache
end
