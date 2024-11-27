# Physics model

HallThruster.jl solves the quasineutral plasma equations of motion for a Hall Thruster along the thruster's channel centerline (the z-axis). We solve seperate models for neutral particles, ions, and electrons. Neutrals are assumed to have constant velocity and temperature and are tracked by a single continuity equation. Ions are assumed isothermal and unmagnetized. Multiple ion species with different charge states are supported, and each is tracked by a continuity equation and a momentum equation. We employ the drift-diffusion approximation for electrons, which reduces the electron momentum equation to a generalized Ohm's law. Charge conservation is then used to solve for the electrostatic potential. The electron temperature is determined by solving an equation for the conservation of electron internal energy. The model is based upon the work presented in [K. Hara, *Non-oscillatory quasineutral fluid model of cross-field discharge plasmas*, Physics of Plasmas 25, 123508, 2018](https://aip.scitation.org/doi/pdf/10.1063/1.5055750). See [Configuration](@ref) for supported gases. Xenon is the standard, Krypton is fully supported and in theory any monoatomic gas can be used as propellant in the simulation.

## Neutrals

For neutrals, the continuity equation is solved:

```math
    \frac{\partial n_n}{\partial t} + \frac{\partial}{\partial z} (n_n u_n) = \dot{n}_n
```

Here, ``n_n`` is the neutral number density in m``^{-3}``, ``\mathbf{u_n}`` is the neutral velocity vector in m/s, and ``\dot{n}_n`` is the rate of neutral depletion due to ionization in  m``^{-3}``s``^{-1}``, which is given by

```math
    \dot{n}_n = -\sum_{j = 1}^3 n_e n_n k_{nj}(T_e)
```

where ``n_e`` is the electron number density ``j`` represents the ion charge state (i.e. ``j = 1`` represents singly-charged ions, and so on), ``T_e`` is the electron temperature, and ``k_{nj}`` is the rate coefficient of the ionization reaction
 
```math   
A + e^- \rightarrow A^{j+} + (j + 1) e^-
```

where A represents the gas species being simulated. Currently, the code is compatible with Xenon and Krypton. The reaction rate coefficients are generated as a function of electron temperature using the [BOLSIG+ code](http://www.bolsig.laplace.univ-tlse.fr).
We read in a table of these rate coefficients with electron temperature and use the Interpolations.jl to generate transform this data into a continuous function. 

The neutrals are assumed to have a constant velocity in the axial direction and a constant temperature, and are thus approximated monoenergetic and not Maxwellian. The neutral momentum and energy equations are not solved for. 

## Ions

We solve continuity and momentum for each ion species. We may have the option for an ion energy equation, but for now they are treated as isothermal. The ion continuity equation for ions with charge ``j`` is

```math
    \frac{\partial n_{ij}}{\partial t} + \frac{\partial}{\partial z} (n_{ij} u_{ij}) = \dot{n}_{ij}
```

Here ``n_{ij}``, ``u_{ij}``, and ``\dot{n}_{ij}`` are the number density, velocity, and net rate of production of ions with charge state ``j``. The production rate ``\dot{n}_{ij}`` is given by:

```math
    \dot{n}_{ij} = n_e n_n k_{nj}(Te) - \sum_{\ell = j + 1}^3 n_e n_{ij} k_{j\ell}(T_e)
```

The first term here represents the rate of production of ions with charge state ``j`` and the second term represents the rate at which these ions are further ionized to become ions of charge state ``\ell``. In all, the following six reactions are modelled:

```math
\begin{aligned}
    A + e^- &\rightarrow A^{+} + 2 e^-\\
    A + e^- &\rightarrow A^{2+} + 3 e^-\\
    A + e^- &\rightarrow A^{3+} + 4 e^-\\
    A+ + e^- &\rightarrow A^{2+} + 2 e^-\\
    A+ + e^- &\rightarrow A^{3+} + 3 e^-\\
    A^{2+} + e^- &\rightarrow A^{3+} + 2 e^-
\end{aligned}
```

The currently-specified model does not include ion losses to the radial walls, but this could be included at a later date. Likewise, we could also include momentum-transfer collisions between ions and neutrals and between ions of different charge states at a future date, but neglect these for now. Future updates may also add the ability to model molecular propellants, not just monatomic ones, in which case we would need to add significantly more reaction equations, species, and model rotational and vibrational modes.

The one-dimensional momentum equation for ions of charge state ``j`` is obtained by assuming the ions are unmagentized and that the momentum transfer due to collisions is negligible. The momentum equation in conservative form is

```math
    \frac{\partial}{\partial t} (n_{ij} u_{ij}) + \frac{\partial}{\partial z} (n_{ij} u_{ij}^2 + \frac{p_{ij}}{m_i}) = \frac{j e}{m_i} n_{ij} E_z
```

In this equation, ``p_{ij} = n_{ij} k_B T_{i}`` is the partial pressure of ions with charge ``j``, ``T_i`` is the ion temperature, ``e`` is the fundamental charge, ``m_i`` is the ion mass, and ``E_z`` is the axial electric field. 

## Electrons

We assume that the plasma is quasineutral, which means that the local charge density is zero everywhere. This means that

```math
    n_e = \sum_{j=1}^3 j\;n_{ij}.
```

In addition, the electrons are assumed to be massless. This yields a generalized Ohm's law, also known as the Quasineutral Drift Diffusion (QDD) model. The electron momentum equation becomes:

```math
    \nu_e \frac{m_e}{e}\mathbf{j}_e = e n_e \mathbf{E} +\nabla p_e - \mathbf{j}_e \times \mathbf{B}
```

Here, ``\nu_e`` is the total electron momentum transfer collision frequency, ``\mathbf{j}_e = -e n_e \mathbf{u_e}`` is the electron current vector, ``p_e = n_e k_B T_e`` is the electron pressure, and ``B`` is the magnetic field. We want to model the electron velocity in both the axial (``\hat{z}``) and azimuthal (``\theta``) directions. Making the assumption that ``B`` is purely radial and that the plasma is axisymmetric, we arrive at the following two equations after some algebraic manipulations.

```math
    j_{ez} = \frac{e^2 n_e}{m_e \nu_e}\frac{1}{1 + \Omega_e^2}\left(E_z + \frac{1}{e n_e}\frac{\partial p_e}{\partial z}\right)\\
    j_{e\theta} = \Omega_e j_{ez}
```

In this expression, ``\Omega_e = \omega_{ce}/\nu_e = e |B| / m_e \nu_e`` is the Hall parameter, or the ratio of the electron cyclotron frequency to the total electron momentum transfer collision frequency, and measures how well-magnetized the electrons are. Finally, we introduce the anomalous collision frequency (``\nu_{AN}``):

```math
    \nu_e = \nu_c + \nu_{AN}
```

In Hall thrusters, the observed axial/cross-field electron current is significantly higher than that which would result from classical collisions alone (here, ``\nu_c`` represents the classical electron momentum transfer collision frequency, see [Collisions and Reactions](@ref)). We model this enhanced transport in a fluid framework as an additional anomalous collision frequency, see [Anomalous Transport](@ref). The purpose of this code is to facilitate the development and testing of models for this important parameter.

## Discharge current and electric field

 To determine the electric field and dischrage current we generally follow the method from [V. Giannetti, et. al *Numerical and experimental investigation of longitudinal oscillations in Hall thrusters*, Aerospace 8, 148, 2021](https://www.mdpi.com/2226-4310/8/6/148). We outline the main steps here. By writing the discharge current as ``\frac{I_d}{A_{ch}} = j_e + j_i`` where ``A_{ch}`` is the channel area, plugging in Ohm's law for the electron current density, and integrating over the domain, we can write the discharge current as

```math
    I_d = \frac{\Delta V + \int_{z_c}^{z_a} \frac{1}{en_e}\frac{\partial p_e}{\partial z} + \frac{j_{iz}}{en_e\mu_{\perp}} dz}{\int_{z_c}^{z_a} en_e \mu_{\perp} A dz}
```

Where A is the cross section area of either the channel or plume, ``z_c`` and ``z_a`` are the cathode and anode positions, and ``\Delta V, j_{iz},`` and ``\mu_{\perp}`` are given by 

```math
\Delta V = V_d + V_s \\
j_{iz} = \Sigma_{j=1}^{3} j\;n_{ij} u_{ij} \\
\mu_{\perp} = \frac{e}{m_e \nu_e} \frac{1}{1+\Omega_e^2}
```

With the discharge current known, the axial electric field is computed locally as

```math
E_z = \frac{I_d}{en_e\mu_{\perp}A_{ch}} - \frac{1}{en_e} \frac{\partial p_e}{\partial z} - \frac{j_{iz}}{en_e\mu_{\perp}} 
```
As the electric field is the negative gradient of the electrostatic potential, the potential is finally calculted by integrating the negative electric field using the trapezoid rule.


## Electron energy equation

The electron internal energy equation in one dimension is

```math
    \frac{\partial}{\partial t}\left(\frac{3}{2} n_e k_B T_e\right) + \frac{\partial}{\partial z}\left(\frac{5}{2} n_e k_B T_e u_{ez} + q_{ez}\right) = 
    n_e u_{ez} \frac{\partial\phi}{\partial z} - W_{loss} - S_{coll}
```


Here, ``q_ez`` is the electron heat conduction in one dimension and ``S_{wall}``, see [Wall Loss Models](@ref),  represents the loss of electron energy to the thruster walls and ``S_{coll}``, see [Collisions and Reactions](@ref) captures the loss of energy due to inelastic collisions. The heat conduction is defined by Fourier's Law:


```math
\begin{aligned}
    q_{ez} &= -\kappa_{e\perp} \nabla_{\perp} T_e\\ 
\end{aligned}
```

In this expression, ``\kappa_{e\perp}`` is the cross-field (axial) electron thermal conductivity, for which various forms exist. More details can be found on the [Electron Thermal Conductivity](@ref) page. 

The heat transfer terms slightly change when considering the [Landmark case study](https://www.landmark-plasma.com/test-case-3), while the different wall and inelastic collision loss models are described in [Wall Loss Models](@ref) and [Collisions and Reactions](@ref). 

## Sheath considerations

HallThruster.jl, being a fluid globally quasineutral model, is not designed to resolve plasma sheaths. However, the sheath and presheath are important to model Hall Thruster discharges accurately. As this is a 1D axial solver, we do not have any direct fluxes towards the walls, the energy losses can however be taken into account by a source term in the energy equation. This term and the boundary conditions implemented at the anode employ the following presheath approximations and assumptions. They are absolutely critical to replicate experimental Hall Thruster behaviour. 

In the following, potential differences ``e\phi`` are assumed to be on the order of the electron temperature ``k T_e``. Furthermore, assume that cold ions fall through an arbitrary potential of ``\phi_0`` while they move towards the wall. Through conservation of energy, their arrival velocity at the sheath edge can be related to the potential difference. 

```math
    \frac{1}{2} m_i v_0^2 = e \phi_0
```

Additionally, the ion flux during acceleration toward the wall is conserved. 

```math
    n_i v = n_0 v_0
```
        
The relation for ion velocity as a function of position in the sheath can be written as 


```math
    \frac{1}{2} m_i v^2 = \frac{1}{2} m_i v_0^2 - e\phi (x)
```

Rewriting both energy conservation and above expression for ``v_0`` and ``v``, and dividing gives

```math
    \frac{v_0}{v} = \sqrt{\frac{\phi_0}{\phi_0 - \phi}}
```

which by applying flux conservation results in 

```math
    n_i = n_0 \sqrt{\frac{\phi_0}{\phi_0 - \phi}}
```

Close to the sheath edge the density equation can be expanded as a Taylor series, as ``\phi`` is small compared to ``\phi_0``.


```math
    n_i = n_0 \left(1 - \frac{1}{2}\frac{\phi}{\phi_0} + ...\right)
```

In one dimension, neglecting collisions with other species and assuming isentropic temperature and pressure terms, no convection and no electron inertia, the electrons can be described by the Boltzmann relation.


```math
    n_e = n_0 exp\left(\frac{e \phi}{k T_e}\right)
```

In this regime, the electron density is diffusion dominated and dictated by the electrostatic field. This assumption is generally valid along magnetic field lines and across weak magnetic fields with sufficient electron electron collisions. The Boltzmann relation can be expanded by assuming that the change in potential at the sheath edge is small compared to the electron temperature. 

```math
    n_e = n_0 \left(1 - \frac{e\phi}{k T_e} + ... \right)
```

Taking Poisson's equation of the form 

```math
    \nabla^2 \phi = - \frac{e}{k Te_0}(n_i - n_e)
```

and substituting expanded Boltzmann relation and expanded ion density leads after rearranging to 

```math
    \nabla^2 \phi = \frac{e n_0 \phi}{\epsilon_0}\left(\frac{1}{2\phi_0} - \frac{e}{kT_e}\right)
```

As the sheath is assumed to be ion attracting, it can by definition not slow or repell ions. As a result, the right hand side of \autoref{eq:poisson_sub_expanded} has to always be positive, which leads to the following requirement. 

```math
    \phi_0 > \frac{kT_e}{2e}
```

By substituting energy conservation equation, the ion Bohm speed can be recovered. This condition is applied to the anode boundary and will be discussed in the boundary conditions. 

```math
    v_0 > \sqrt{\frac{kT_e}{m_i}}
```
