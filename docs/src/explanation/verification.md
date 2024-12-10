# Verification

Tests can be found in the `test` folder, and are split in `unit_tests` and `order_verification` tests. We verify that the PDEs are discretized correctly  using the [Method of Manufactured Solutions (MMS)](https://asmedigitalcollection.asme.org/fluidsengineering/article/124/1/4/462791/Code-Verification-by-the-Method-of-Manufactured) and perform order verification studies in order to ensure that the actual order of accuracy matches the predicted order.  For more details on the discretization, see [Fluxes](@ref) and [Numerics](@ref).

## Landmark

In addition to the MMS studies discussed above, we also compare the results to the [Landmark test case](https://www.landmark-plasma.com/test-case-3)s for 1D fluid Hall Thruster discharges. Below, we compare the time-averaged output of HallThruster.jl for each of the three test cases to the expected results from Landmark. The cases differ only in the amount of electron energy lost to to radial sheaths inside the thruster.  For the purpose of verification, the boundary conditions, source terms, collision models and anomalous collision frequency has been set to match Landmark. The results shown are time-averaged, performed using 160 cells using the first-order Rusanov flux and without gradient reconstruction. 

Landmark energy loss term:
```math
    W = \nu_\epsilon \exp\left(\frac{-20}{\epsilon}\right)
```

where

```math
    \nu_{\epsilon}=
    \begin{cases}
        \alpha_1 \times 10^7 & z - z_0 \leq L_{ch} \\
        \alpha_2 \times 10^7 & z - z_0 > L_{ch}
    \end{cases}
```

and

```math
\epsilon = \frac{3}{2} T_{ev}
```

In the above, ``L_{ch}`` refers to thruster channel length and ``z_0`` is `domain[1]`, or the z-location of the anode.

Case 1
``\; \; \alpha_1 = 1.0, \alpha_2 = 1.0``
![Landmark1](https://raw.githubusercontent.com/UM-PEPL/HallThruster.jl/main/docs/src/assets/landmark_case1_rusanov_160cells.jpg)

Case 2
``\; \; \alpha_1 = 0.5, \alpha_2 = 1.0``
![Landmark2](https://raw.githubusercontent.com/UM-PEPL/HallThruster.jl/main/docs/src/assets/landmark_case2_rusanov_160cells.jpg)

Case 3
``\; \; \alpha_1 = 0.4, \alpha_2 = 1.0``
![Landmark3](https://raw.githubusercontent.com/UM-PEPL/HallThruster.jl/main/docs/src/assets/landmark_case3_rusanov_160cells.jpg)
