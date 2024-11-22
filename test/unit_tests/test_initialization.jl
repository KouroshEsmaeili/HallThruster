@testset "Initialization" begin
    struct TestAnomModel <: HallThruster.AnomalousTransportModel end

    HallThruster.num_anom_variables(::TestAnomModel) = 3

    domain = (0.0, 0.08)
    thruster = HallThruster.SPT_100
    config = HallThruster.Config(;
        anom_model = TestAnomModel(),
        ncharge = 3,
        propellant = HallThruster.Xenon,
        thruster,
        domain,
        cathode_Te = 5.0,
        anode_Te = 3.0,
        neutral_velocity = 300.0,
        neutral_temperature = 100.0,
        ion_temperature = 300.0,
        initial_condition = HallThruster.DefaultInitialization(;
            max_electron_temperature = 10.0),
        anode_mass_flow_rate = 3e-6,
        discharge_voltage = 500.0,
    )

    mi = config.propellant.m

    ncells = 100
    fluids, fluid_ranges, species, species_range_dict, is_velocity_index = HallThruster.configure_fluids(config)
    grid = HallThruster.generate_grid(config.thruster.geometry, domain, EvenGrid(ncells))
    U, cache = HallThruster.allocate_arrays(grid, config)
    index = HallThruster.configure_index(fluids, fluid_ranges)

    params = (;
        index,
        cache,
        grid,
        config,
    )

    HallThruster.initialize!(U, params)

    @test abs((U[index.ρn, 1] / U[index.ρn, end]) - 100) < 1

    ne = [HallThruster.electron_density(U, params, i) for i in 1:(ncells + 2)]
    ϵ = params.cache.nϵ ./ ne

    max_Te = 2 / 3 * maximum(ϵ)
    @test 9 <= max_Te <= 11

    ui = [U[index.ρiui[Z], :] ./ U[index.ρi[Z], :] for Z in 1:(config.ncharge)]

    @test ui[1][1] ≈ -sqrt(HallThruster.e * config.anode_Te / mi)
    @test ui[2][1] ≈ -sqrt(2 * HallThruster.e * config.anode_Te / mi)
    @test ui[3][1] ≈ -sqrt(3 * HallThruster.e * config.anode_Te / mi)

    @test abs(2 / 3 * ϵ[1] - config.anode_Te) < 0.1
    @test abs(2 / 3 * ϵ[end] - config.cathode_Te) < 0.1

    @test cache.anom_variables == [zeros(102) for _ in 1:3]

    struct NewInitialization <: HallThruster.InitialCondition end
    @test_throws ArgumentError HallThruster.initialize!(U, params, NewInitialization())
end

@testset "Anom initialization" begin
    anom_model = HallThruster.NoAnom()

    config = HallThruster.Config(;
        thruster = HallThruster.SPT_100,
        domain = (0.0u"cm", 8.0u"cm"),
        discharge_voltage = 300.0u"V",
        anode_mass_flow_rate = 5u"mg/s",
        wall_loss_model = HallThruster.WallSheath(HallThruster.BoronNitride, 0.15),
        anom_model,
    )

    ncells = 10
    sim_options = (; ncells, nsave = 2, verbose = false)
    initial_model = HallThruster.TwoZoneBohm(1 // 160, 1 / 16)

    # Check that anomalous transport is initialized to a two-zone Bohm approximation instead of the prescribed NoAnom.
    sol = HallThruster.run_simulation(
        config; ncells, dt = 0.0, duration = 0.0, sim_options...,)
    @test initial_model(zeros(ncells + 2), sol.params) == sol.params.cache.νan
    @test anom_model(zeros(ncells + 2), sol.params) != sol.params.cache.νan

    # Check that after one iteration, the anomalous transport is the correct value
    dt = 1e-8
    sol = HallThruster.run_simulation(config; ncells, dt, duration = dt, sim_options...)
    @test initial_model(zeros(ncells + 2), sol.params) != sol.params.cache.νan
    @test anom_model(zeros(ncells + 2), sol.params) == sol.params.cache.νan
end

@testset "Configuration" begin
    common_opts = (;
        ncharge = 3,
        discharge_voltage = 300u"V",
        anode_mass_flow_rate = 5u"mg/s",
        thruster = HallThruster.SPT_100,
        domain = (0.0u"cm", 5.0u"cm"),
    )

    config = HallThruster.Config(;
        background_pressure = 0.0u"Torr",
        background_neutral_temperature = 0.0u"K",
        common_opts...,
    )

    fluids, fluid_ranges, species, species_range_dict, is_velocity_index = HallThruster.configure_fluids(config)

    @test fluid_ranges == [1:1, 2:3, 4:5, 6:7]
    @test species == [Xenon(0), Xenon(1), Xenon(2), Xenon(3)]
    @test species_range_dict == Dict(
        Symbol("Xe") => 1:1,
        Symbol("Xe+") => 2:3,
        Symbol("Xe2+") => 4:5,
        Symbol("Xe3+") => 6:7,
    )

    @test fluids[1] == HallThruster.ContinuityOnly(
        species[1], config.neutral_velocity, config.neutral_temperature,)
    @test fluids[2] == HallThruster.IsothermalEuler(species[2], config.ion_temperature)
    @test fluids[3] == HallThruster.IsothermalEuler(species[3], config.ion_temperature)
    @test fluids[4] == HallThruster.IsothermalEuler(species[4], config.ion_temperature)
    @test is_velocity_index == [false, false, true, false, true, false, true]

    index = HallThruster.configure_index(fluids, fluid_ranges)
    @test keys(index) == (:ρn, :ρi, :ρiui)
    @test values(index) == (1, [2, 4, 6], [3, 5, 7])

    # load collisions and reactions
    ionization_reactions = HallThruster._load_reactions(
        config.ionization_model, unique(species),)
    ionization_reactant_indices = HallThruster.reactant_indices(
        ionization_reactions, species_range_dict,)
    @test ionization_reactant_indices == [1, 1, 1, 2, 2, 4]

    ionization_product_indices = HallThruster.product_indices(
        ionization_reactions, species_range_dict,)
    @test ionization_product_indices == [2, 4, 6, 4, 6, 6]

    excitation_reactions = HallThruster._load_reactions(
        config.excitation_model, unique(species),)
    excitation_reactant_indices = HallThruster.reactant_indices(
        excitation_reactions, species_range_dict,)
    @test excitation_reactant_indices == [1]

    # Test that initialization and configuration works properly when background neutrals are included

    pB = 5e-6u"Torr"
    TB = 120u"K"

    config_bg = HallThruster.Config(;
        background_pressure = pB,
        background_neutral_temperature = TB,
        common_opts...,
    )

    fluids, fluid_ranges, species, species_range_dict = HallThruster.configure_fluids(config_bg)
    @test fluid_ranges == [1:1, 2:3, 4:5, 6:7]
    @test species == [Xenon(0), Xenon(1), Xenon(2), Xenon(3)]
    @test species_range_dict == Dict(
        Symbol("Xe") => 1:1,
        Symbol("Xe+") => 2:3,
        Symbol("Xe2+") => 4:5,
        Symbol("Xe3+") => 6:7,
    )

    @test fluids[1] == HallThruster.ContinuityOnly(
        species[1], config.neutral_velocity, config.neutral_temperature,)
    @test fluids[2] == HallThruster.IsothermalEuler(species[2], config.ion_temperature)
    @test fluids[3] == HallThruster.IsothermalEuler(species[3], config.ion_temperature)
    @test fluids[4] == HallThruster.IsothermalEuler(species[4], config.ion_temperature)
    @test is_velocity_index == [false, false, true, false, true, false, true]

    index = HallThruster.configure_index(fluids, fluid_ranges)
    @test keys(index) == (:ρn, :ρi, :ρiui)
    @test values(index) == (1, [2, 4, 6], [3, 5, 7])

    # load collisions and reactions
    ionization_reactions = HallThruster._load_reactions(
        config.ionization_model, unique(species),)
    ionization_reactant_indices = HallThruster.reactant_indices(
        ionization_reactions, species_range_dict,)
    @test ionization_reactant_indices == [1, 1, 1, 2, 2, 4]

    ionization_product_indices = HallThruster.product_indices(
        ionization_reactions, species_range_dict,)
    @test ionization_product_indices == [2, 4, 6, 4, 6, 6]

    excitation_reactions = HallThruster._load_reactions(
        config.excitation_model, unique(species),)
    excitation_reactant_indices = HallThruster.reactant_indices(
        excitation_reactions, species_range_dict,)
    @test excitation_reactant_indices == [1]
end
