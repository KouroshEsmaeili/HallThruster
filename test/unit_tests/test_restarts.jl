@testset "Restarts" begin
    test_path = joinpath(HallThruster.PACKAGE_ROOT, "test", "unit_tests")
    restart_path = joinpath(test_path, "test_restart.jld2")

    config = HallThruster.Config(;
        thruster = HallThruster.SPT_100,
        domain = (0.0u"cm", 8.0u"cm"),
        discharge_voltage = 300.0u"V",
        anode_mass_flow_rate = 5u"mg/s",
        wall_loss_model = HallThruster.WallSheath(HallThruster.BoronNitride),
        transition_length = 1e-3,
    )

    ncells = 50
    grid = HallThruster.generate_grid(
        config.thruster.geometry, config.domain, EvenGrid(ncells),)
    fluids, fluid_ranges, species, species_range_dict = HallThruster.configure_fluids(config)

    #make a new restart from the current configuration
    duration = 1e-7
    dt = 1e-8
    nsave = 2
    sol = HallThruster.run_simulation(config; ncells, duration, dt, nsave, verbose = false)
    HallThruster.write_restart(restart_path, sol)

    # Check that writing a restart of a restart does not lose any information
    # i.e. restarts are perfectly reconstructable
    restart = HallThruster.read_restart(restart_path)
    @test sol.retcode == :success
    @test sol.u == restart.u
    @test sol.savevals == restart.savevals
    @test sol.t == restart.t

    # test loading restart, generating cache, U, etc, without interpolation, changing number of charges, or changing domain
    U, cache = HallThruster.load_restart(grid, config, sol)
    U2, cache2 = HallThruster.load_restart(grid, config, restart_path)

    # Check that we get the same result by loading solution directly or by loading filename
    @test U == U2
    @test cache == cache2

    # Check that the result is correct
    @test U ≈ sol.u[end]
    @test grid.cell_centers ≈ sol.params.grid.cell_centers
    @test cache.νan ≈ sol[:νan][end]
    @test cache.ne ≈ sol[:ne][end]
    @test cache.ϕ ≈ sol[:ϕ][end]
    @test cache.Tev ≈ sol[:Tev][end]
    @test cache.νc ≈ sol[:νc][end]

    #test that a simulation can be run from the restart
    restart_sol = HallThruster.run_simulation(
        config; dt = 5e-9, duration = 4e-9, grid = HallThruster.EvenGrid(ncells),
        nsave = 10, restart = restart_path,)
    @test restart_sol.retcode == :success

    # Clean up restart
    rm(restart_path, force = true)
end

@testset "Landmark data loading" begin
    landmark_1_1 = HallThruster.load_landmark_data(1, "fluid_1")
    landmark_1_2 = HallThruster.load_landmark_data(1, "fluid_2")
    landmark_1_hybrid = HallThruster.load_landmark_data(1, "hybrid")

    landmark_2_1 = HallThruster.load_landmark_data(2, "fluid_1")
    landmark_2_2 = HallThruster.load_landmark_data(2, "fluid_2")
    landmark_2_hybrid = HallThruster.load_landmark_data(2, "hybrid")

    landmark_3_1 = HallThruster.load_landmark_data(3, "fluid_1")
    landmark_3_2 = HallThruster.load_landmark_data(3, "fluid_2")
    landmark_3_hybrid = HallThruster.load_landmark_data(3, "hybrid")

    @test length(landmark_1_1.u) == 1
    @test size(landmark_1_1.u[1]) == (4, 100)
    @test all(isnan, landmark_1_1.u[1][3, :])

    landmark_1 = HallThruster.load_landmark_data(1)

    @test landmark_1[1].u[1][1, :] ≈ landmark_1_1.u[1][1, :]
    @test landmark_1[1].u[1][2, :] ≈ landmark_1_1.u[1][2, :]
    @test landmark_1[1].u[1][4, :] ≈ landmark_1_1.u[1][4, :]

    @test landmark_1[2].u[1][1, :] ≈ landmark_1_2.u[1][1, :]
    @test landmark_1[2].u[1][2, :] ≈ landmark_1_2.u[1][2, :]
    @test landmark_1[2].u[1][4, :] ≈ landmark_1_2.u[1][4, :]

    @test landmark_1[3].u[1][1, :] ≈ landmark_1_hybrid.u[1][1, :]
    @test landmark_1[3].u[1][2, :] ≈ landmark_1_hybrid.u[1][2, :]
    @test landmark_1[3].u[1][4, :] ≈ landmark_1_hybrid.u[1][4, :]
end
