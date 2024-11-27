# Configuration

The `Config` struct contains all of the options you need to run a simulation. On this page, we will explain what options are available and what they do. Note that all arguments must be provided as keywords.

There are four absolutely mandatory arguments. These are:

- `discharge_voltage`: The difference in potential between the anode and cathode, in Volts. This is used to set the left boundary condition. If the cathode potential is zero, then the anode potential is equal to the discharge voltage.
- `thruster`: This is a `Thruster` object containing important geometric and magnetic information about the thruster being simulated. See the page about [Thrusters](@ref) for more.
- `domain`: This is a Tuple containing the locations of the left and right boundaries of the simulation domain, in meters. For instance, if your simulation domain starts at z = 0.0 and is 5 cm long, you would write `domain = (0.0, 0.05)`.
- `anode_mass_flow_rate`: The propellant mass flow rate at the anode, in kg/s

Aside from these arguments, all others have  default values provided. These are detailed below:

- `initial_condition`: A function used for initializing the simulation. See the page about [Initialization](@ref) for more information.
- `ncharge`: Number of charge states to simulate. Defaults to `1`.
- `propellant`: Propellant gas. Defaults to `Xenon`. Other options are described on the [Propellants](@ref) page.
- `scheme`: Numerical scheme to employ for integrating the ion equations. This is a `HyperbolicScheme` struct with fields `flux_function`, `limiter`, and `reconstruct`. Defaults to `HyperbolicScheme(flux_function = rusanov, limiter = minmod, reconstruct = false)`. For more information, see [Fluxes](@ref).
- `cathode_potential`: The potential at the right boundary of the simulation. Defaults to `0.0`
- `anode_Te`: The electron temperature at the anode, in eV. Acts as a Dirichlet boundary condition for the energy equation. Defaults to `3.0`.
- `cathode_Te`: The electron temperature at the cathode, in eV. Acts as a Dirichlet boundary condition for the energy equation. Defaults to `3.0`.
- `wall_loss_model`: How radial losses due to sheaths are computed. Defaults to `ConstantSheathPotential(sheath_potential=-20.0, inner_loss_coeff = 1.0, outer_loss_coeff = 1.0)`, which is the loss term from LANDMARK case 1. Other wall loss models are described on the [Wall Loss Models](@ref) page.
- `anom_model`: Model for computing the anomalous collision frequency. Defaults to `TwoZoneBohm(1/160, 1/16)`. Further details on the [Anomalous Transport](@ref) page.
- `transition_length`: Distance over which the transition between inside and outside the channel is smoothed. Affects wall losses as well as two-zone Bohm-like transport models.
- `conductivity_model`: Model for the perpendicular electron thermal conductivity. Defaults to `Mitchner()`. Further details can be found on the [Electron Thermal Conductivity](@ref) page.
- `ionization_model`: Model for ionization. Defaults to `IonizationLookup()`, which uses a lookup table to compute ionization rate coefficients as a function of electron energy. Other options are described on the [Collisions and Reactions](@ref) page.
- `excitation_model`: Model for excitation reactions. Defaults to `ExcitationLookup()`, which uses a lookup table to compute excitation rate coefficients as a function of electron energy.. Other models are described on the [Collisions and Reactions](@ref) page.
- `electron_neutral_model`: Model for elastic scattering collisions between electrons and neutral atoms. Defaults to `ElectronNeutralLookup()`, which uses a lookup table to compute the elastic scattering rate coefficient. Other models are described on the [Collisions and Reactions](@ref) page.
- `electron_ion_collisions`: Whether to include electron-ion collisions. Defaults to `true`. More information on the [Collisions and Reactions](@ref) page.
- `neutral_velocity`: Neutral velocity in m/s. Defaults to `300.0`. Note: If this is not set, the `neutral_temperature` is used to compute it using a one-sided maxwellian flux approximation.
- `neutral_temperature`: Neutral temperature in Kelvins. Defaults to `500.0`. 
- `ion_temperature`: Ion temperature in Kelvins. Defaults to 100.0
- `implicit_energy`: The degree to which the energy is solved implicitly. `0.0` is a fully-explicit forward Euler, `0.5` is Crank-Nicholson, and `1.0` is backward Euler. Defaults to `1.0`.
- `min_number_density`: Minimum allowable number density for any species. Defaults to `1e6`
- `min_electron_temperature`: Minimum allowable electron temperature. Defaults to `1.0`.
- `magnetic_field_scale`: Factor by which the magnetic field is increased or decreased compared to the one in the provided `Thruster` struct. Defaults to `1.0`.
- `source_neutrals`: Extra user-provided neutral source term. Can be an arbitrary function, but must take `(U, params, i)` as arguments. Defaults to `Returns(0.0)`. See [User-Provided Source Terms](@ref source_terms_md) for more information.
- `source_ion_continuity`: Vector of extra source terms for ion continuity, one for each charge state. Defaults to `fill(Returns(0.0), ncharge)` . See [User-Provided Source Terms](@ref source_terms_md) for more information.
- `source_ion_momentum`: Vector of extra source terms for ion momentum, one for each charge state. Defaults to `fill(Returns(0.0), ncharge)` . See [User-Provided Source Terms](@ref source_terms_md) for more information.
- `source_potential`: Extra source term for potential equation. Defaults to `Returns(0.0)`. See [User-Provided Source Terms](@ref source_terms_md) for more information.
- `source_electron_energy`: Extra source term for electron energy equation. Defaults to `Returns(0.0)`. See [User-Provided Source Terms](@ref source_terms_md) for more information.
- `LANDMARK`: Whether we are using the LANDMARK physics model. This affects whether certain terms are included in the equations, such as electron and heavy species momentum transfer due to ionization and the form of the electron thermal conductivity. Also affects whether we use an anode sheath model. Defaults to `false`.
- `ion_wall_losses`: Whether we model ion losses to the walls. Defaults to `false`.
- `background_pressure`: The pressure of the background neutrals, in Pascals. These background neutrals are injected at the anode to simulate the ingestion of facility neutrals.
- `background_neutral_temperature`: The temperature of the background neutrals, in K
- `anode_boundary_condition`: Can be either `:sheath` or `:dirichlet`
- `anom_smoothing_iters`: How many times to smooth the anomalous transport profile. Defaults to zero
- `solve_plume`: Whether quasi-1D beam expansion should be modelled outside of the channel.
- `apply_thrust_divergence_correction`: Whether the thrust output by HallThruster.jl should include a divergence correction factor of cos(δ)^2
- `electron_plume_loss_scale`: The degree to which radial electron losses are applied in the plume. Defaults to 1. See [Wall Loss Models](@ref) for more information.
