# HallThruster.jl 0.18.5-era translation validation

## Scope and reference

This document records M2 component validation of the historical Python
translation against:

```text
UM-PEPL/HallThruster.jl
commit 014a12fb193af6927cb10f77da5e7baf215b5bc0
```

The Python work is based on commit `bb00d8ce` on
`feature/julia-python-validation`, with the M2 changes left uncommitted for
review. HallThruster.jl and its authors remain the source of the physical model,
equations, numerical methods, and original architecture.

This is not a claim of scientific equivalence. The full historical regression,
postprocessing metrics, long-time behavior, and performance remain outside M2.

## Environment and executable-Julia limitation

The validation environment used Python 3.12.3 from the repository-local
`.venv`. No system Python or external repository environment was used.

No Julia executable was installed or present in common system/user locations.
Access to Julia's official download host was also unavailable: the configured
devbox proxy could not be reached, and direct DNS resolution was disabled.
Consequently:

- component expectations in the normal test suite are derived directly from
  the historical Julia source, its historical unit tests, and deterministic
  formulas/data files;
- these results are explicitly called **source-level parity**;
- setup, nonzero single-step, and short-simulation Julia comparisons are
  **blocked**, not silently treated as passing;
- the optional Julia exporter is present but has not been executed in this
  environment.

## Methodology

The validation cases use deterministic inputs shared by the Python and Julia
exporters in `validation/`. Both produce JSON with the following top-level
components:

- constants, gases/species, geometry, and magnetic field;
- even and uneven grids;
- interpolation, finite differences, integration, and tridiagonal solves;
- slope limiters and Rusanov/global Lax-Friedrichs/HLLE fluxes;
- elastic, excitation, and ionization tables;
- Config defaults and physical fluid/index mappings;
- deterministic initialization and setup arrays;
- selected fields from a fixed-step 20-cell, `1e-7 s` simulation.

`validation/compare_outputs.py` recursively compares matching fields and
reports maximum absolute error, maximum relative error, and relative L2 error
for each component. Its default tolerance is `rtol=1e-12`, `atol=1e-30`.
The source-derived tests generally use exact equality or relative tolerances of
`1e-15` to `2e-15`; no tolerance larger than `1e-8` was introduced.

Generated JSON under `validation/outputs/` is ignored. See
`validation/README.md` for exact regeneration commands using an extracted
checkout of `014a12f`, rather than modern upstream code.

## Component validation matrix

`PASS (source)` means the Python result was compared with a historical literal,
formula, algorithm, or unit-test expectation at `014a12f`; it does not mean a
Julia executable was run. `FIXED (source)` means such a comparison exposed and
fixed a translation discrepancy.

| Component | Status | Maximum observed error | Notes |
|---|---:|---:|---|
| Physical constants | PASS (source) | 0 | Historical literals and `R0 = kB * NA`. |
| Gas/species | PASS (source) | 0 | Xenon derived properties and charge-state strings. |
| Geometry | PASS (source) | 0 | Generic formulas and SPT-100 dimensions. |
| EvenGrid | PASS (source) | `<= 1e-17` absolute | Counts 4, 10, and 20; edges, centers, ghost centers, and spacings. |
| UnevenGrid | FIXED (source) | `<= 2e-15` relative | Restored the historical N-point inverse-CDF interpolation. |
| Interpolation | PASS (source) | 0 | Interior, endpoints, clamping, and nonuniform locations. |
| Finite differences | PASS (source) | `<= 1e-15` relative | All six coefficient helpers and direct quadratic derivatives. |
| Integration | PASS (source) | `<= 1e-15` absolute | Cumulative trapezoid with nonzero initial value. |
| Linear algebra | FIXED (source) | `<= 1e-15` | Non-mutating tridiagonal solve and input preservation. |
| Limiters | PASS (source) | `<= 1e-15` relative | Negative, zero, 0.25, 1, 10, Inf, and NaN inputs. |
| Flux functions | FIXED (source) | `<= 2e-15` relative | Physical flux, Rusanov, global LF, HLLE, and zero-density IEEE behavior. |
| Reaction tables | FIXED (source) | `<= 2e-15` relative | Elastic/excitation/ionization at 0, 1, 5.5, 10, 20, 50, 100, and 255 eV. |
| Magnetic field | PASS (source) | `<= 2e-15` relative | Six locations from anode through plume, including the exit plane. |
| Config defaults | PASS (source) | 0 | Values and model/source semantics checked; core M1 fixes retained. |
| Fluid/index mapping | PASS (source) | 0 | One, two, and three charge states; physical rows and velocity mask. |
| Initialization | PASS (source) | `<= 2e-15` relative | State, electron density, energy density, and Te on a four-cell case. |
| Setup | PARTIAL | Julia error metrics unavailable | Python arrays exported; source and invariants audited; executable Julia blocked. |
| Single-step solver | BLOCKED | unavailable | Python timestep helpers run; no matched nonzero Julia step. |
| Short simulation | BLOCKED | unavailable | Python reaches `1e-7 s` with 3 finite frames; Julia run unavailable. |

## Confirmed translation discrepancies fixed in M2

### Uneven-grid inverse CDF

- Python: `src/hallthruster/grid/gridspec.py`
- Julia: `src/grid/gridspec.jl`
- Observed difference: Python sampled 2,000 auxiliary points and selected the
  nearest cumulative-density index.
- Root cause: the historical N-point sampling and linear inverse-CDF
  interpolation had been replaced during translation.
- Fix: sample exactly N positions and call the translated linear interpolation
  on N evenly spaced CDF values.
- Regression: `test_uneven_grid_matches_historical_inverse_cdf_algorithm`.

### Tridiagonal non-mutating solve

- Python: `src/hallthruster/utilities/linearalgebra.py`
- Julia: `src/utilities/linearalgebra.jl`
- Observed difference: `tridiagonal_solve` always raised because it called an
  absent `Tridiagonal.copy()` method.
- Root cause: Julia's `copy(A)` behavior was not translated into the wrapper.
- Fix: add a deep copy of all three diagonals.
- Regression: `test_nonmutating_tridiagonal_solve_matches_reference_and_preserves_inputs`.

### Rusanov historical sound-speed behavior

- Python: `src/hallthruster/numerics/flux_functions.py`
- Julia: `src/numerics/flux_functions.jl`
- Observed difference: Python evaluated the right sound speed from `UR`; the
  historical Julia implementation evaluates both `aL` and `aR` from `UL`.
- Root cause: the translation silently corrected a historical implementation
  quirk instead of preserving it.
- Fix: use `UL` for both, with an explicit provenance comment.
- Regression: `test_flux_functions_match_historical_reference_case`.

### Julia IEEE division semantics in fluid fluxes

- Python: `src/hallthruster/physics/thermodynamics.py` and
  `src/hallthruster/numerics/flux_functions.py`
- Julia: `src/physics/thermodynamics.jl` and `src/numerics/flux_functions.jl`
- Observed difference: zero-density states either raised Python
  `ZeroDivisionError` or were forced to zero; Julia Float64 operations yield
  IEEE Inf/NaN values.
- Root cause: language division semantics and defensive translation branches.
- Fix: use a small NumPy-backed Julia-style Float64 division helper in the
  affected thermodynamic/flux expressions.
- Regression: `test_zero_density_flux_uses_julia_ieee_float_semantics`.

### OVS excitation coefficient at zero energy

- Python: `src/hallthruster/collisions/excitation.py`
- Julia: `src/collisions/excitation.jl`
- Observed difference: constructing the OVS table raised at 0 eV in Python;
  Julia evaluates `exp(-8.32 / 0.0)` as zero.
- Root cause: Python raises on scalar float division by zero.
- Fix: return zero explicitly at zero energy.
- Regression: `test_ovs_excitation_zero_energy_matches_julia_float_behavior`.

### Flat-slope reconstruction and wave-speed semantics

- Python: `src/hallthruster/numerics/edge_fluxes.py`
- Julia: `src/numerics/edge_fluxes.jl`
- Observed difference: Python forced a zero reconstruction ratio when the
  upwind slope denominator vanished, causing `no_limiter` to reconstruct a
  spurious slope. It also used the right state for `aR` and a finite `1e9`
  zero-wave-speed timestep sentinel.
- Root cause: Python guard branches changed Julia Float64 Inf/NaN behavior and
  the historical left-state convention.
- Fix: reproduce Inf/NaN ratios, use the left state for `aR`, and return Inf
  for a zero wave speed.
- Regression: `test_reconstruction_with_flat_upwind_slope_matches_julia_nan_limiter_path`.

### Config overload for heavy-species integration

- Python: `src/hallthruster/simulation/update_heavy_species.py`
- Julia: `src/simulation/update_heavy_species.jl`
- Observed difference: the overload accepting a Config indexed it as
  `config["scheme"]`, which always fails for the translated Config class.
- Root cause: dictionary/object access mismatch.
- Fix: use `config.scheme`.
- Regression: `test_config_overload_for_heavy_species_integration_uses_config_scheme`.

## Numerical comparison status

For the deterministic four-cell, three-charge initialization case, comparison
against compact source-derived reference arrays gives:

| Array | Max absolute error | Max relative error | Relative L2 error |
|---|---:|---:|---:|
| Heavy-species state | 0 | 0 | 0 |
| Electron density | 0 | 0 | 0 |
| Electron energy density | 0 | 0 | 0 |
| Electron temperature | 0 | 0 | 0 |

These zero values indicate equality with the committed decimal fixtures after
Float64 round-tripping; they are not measurements from an executed Julia run.

The Python short case currently reports:

```text
retcode=success
requested_end_time=1e-07
actual_end_time=1e-07
saved_frames=3
finite_state_arrays=True
```

Selected Python-only diagnostics (recorded for later comparison, not treated as
reference values) are:

| Time (s) | Mean neutral density | Max ion density | Max Te (eV) | Max abs(E) (V/m) | Id (A) |
|---:|---:|---:|---:|---:|---:|
| 0 | `1.5788463126e19` | `9.7449540553e17` | `26.3856537069` | `1.7976957081e4` | `5.9809148174` |
| `5e-8` | `1.6045944203e19` | `9.4502852333e17` | `24.0938022559` | `1.8888302088e4` | `5.5867031675` |
| `1e-7` | `1.6045192199e19` | `9.2367767433e17` | `23.9326019374` | `1.8198739557e4` | `5.6462703177` |

Julia/Python max-absolute, max-relative, and relative-L2 metrics for setup,
single-step evolution, and the short simulation are unavailable until the Julia
exporter can run. No placeholder zeros or historical full-regression values are
used in their place.

## Automated coverage and remaining work

The M2 tests are:

- `tests/test_julia_source_parity.py`
- `tests/test_initialization_parity.py`
- `tests/test_solver_component_parity.py`

They augment, rather than replace, the M1 suite. Compact expected values are
documented in the tests with the exact reference commit; no large generated
outputs are committed.

Remaining discrepancies and blocked validation are grouped as follows:

- **Indexing:** restart mapping for multiple charge states and less-used
  one-based ranges still need executable/restart cases.
- **Setup:** the matching exporter exists, but no Julia setup JSON is available.
- **Solver:** nonzero one-step and matched short-run comparisons remain blocked.
- **Numerical:** defensive zero-frequency branches outside the validated flux
  path need executable Julia edge-case checks before alteration.
- **Postprocessing:** thrust/current/efficiency parity remains deferred.
- **JSON:** the historical Python JSON entry path remains the existing xfail.
- **Performance:** no optimization or runtime parity work was performed.
- **Scientific validation:** the 200-cell, `1e-3 s` historical regression has
  not been run or used for tuning.

Once Julia 1.10 is available, run the commands in `validation/README.md`, keep
the strict comparator tolerance initially, diagnose the earliest failing
component, and only then proceed to nonzero-step and full scientific validation.
