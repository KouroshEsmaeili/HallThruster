# HallThruster.jl 0.18.5-era translation validation

## Scope and reference

This document records M2 component validation of the historical Python
translation against:

```text
UM-PEPL/HallThruster.jl
commit 014a12fb193af6927cb10f77da5e7baf215b5bc0
```

The Python foundation is commit `bb00d8ce`; the M2 source-parity checkpoint is
`d10696e8` on `feature/julia-python-validation`. The final executable-parity
correction described here is left uncommitted for review. HallThruster.jl and
its authors remain the source of the physical model, equations, numerical
methods, and original architecture.

This is not a claim of scientific equivalence. The full historical regression,
postprocessing metrics, long-time behavior, and performance remain outside M2.

## Environment and validation terminology

The validation environment used Python 3.12.3 from the repository-local
`.venv`. No system Python or external repository environment was used. Julia
1.10.11 was installed user-locally at `/home/dev/.local/bin/julia`. An exact
archive of HallThruster.jl 0.18.5 at `014a12f` was instantiated as its own
temporary Julia project and executed directly.

This report uses the following terms:

- **Executable parity:** results directly produced by Julia 1.10.11 executing
  HallThruster.jl at `014a12f` and Python executing this translation.
- **Source-level parity:** normal Python tests derived from historical source,
  formulas, or compact Julia fixtures; Julia is not required to run them.
- **Deferred scientific validation:** the full historical regression,
  postprocessing metrics, long-duration behavior, and performance comparison.

## Methodology

The validation cases use deterministic inputs shared by the Python and Julia
exporters in `validation/`. Both exporters were run afresh and produced JSON
with the following top-level components:

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
checkout of `014a12f`, rather than modern upstream code. The final executable
comparison exited successfully with code 0.

## Component validation matrix

`PASS (exec)` means both exporters executed and the strict comparator accepted
the component. `FIXED; PASS (exec)` means a confirmed translation discrepancy
was corrected before the successful executable comparison.

| Component | Status | Max abs | Max rel | Relative L2 | Notes |
|---|---:|---:|---:|---:|---|
| Physical constants | PASS (exec) | `0` | `0` | `0` | Historical literals and `R0 = kB * NA`. |
| Gas/species | PASS (exec) | `0` | `0` | `0` | Xenon properties and charge-state representation. |
| Geometry | PASS (exec) | `0` | `0` | `0` | Generic formulas and SPT-100 dimensions. |
| EvenGrid | PASS (exec) | `1.387779e-17` | `3.469447e-15` | `1.392545e-18` | Counts 4, 10, and 20, including ghost cells and spacings. |
| UnevenGrid | FIXED; PASS (exec) | `1.387779e-17` | `4.385825e-15` | `2.449302e-18` | Historical N-point inverse-CDF interpolation. |
| Interpolation | PASS (exec) | `0` | `0` | `0` | Interior, endpoints, clamping, and nonuniform locations. |
| Finite differences | PASS (exec) | `0` | `0` | `0` | Six coefficient helpers and direct quadratic derivatives. |
| Integration | PASS (exec) | `0` | `0` | `0` | Cumulative trapezoid with nonzero initial value. |
| Linear algebra | FIXED; PASS (exec) | `0` | `0` | `0` | Tridiagonal right-hand side and solution. |
| Limiters | PASS (exec) | `0` | `0` | `0` | Negative, zero, 0.25, 1, 10, Inf, and NaN inputs. |
| Flux functions | FIXED; PASS (exec) | `0` | `0` | `0` | Physical flux, Rusanov, global LF, and HLLE. |
| Reaction tables | FIXED; PASS (exec) | `7.888609e-31` | `2.324503e-16` | `1.629429e-33` | Elastic, excitation, and ionization tables at eight energies. |
| Magnetic field | PASS (exec) | `8.673617e-19` | `1.516927e-16` | `6.132858e-18` | Six positions from anode through plume. |
| Config defaults | PASS (exec) | `0` | `0` | `0` | Values and model/source semantics. |
| Fluid/index mapping | PASS (exec) | `0` | `0` | `0` | One, two, and three charge states, compared by physical meaning. |
| Initialization | PASS (exec) | `5.421011e-20` | `1.788993e-16` | `6.016104e-39` | Four-cell, three-charge deterministic initialization. |
| Setup | PASS (exec) | `1.048576e+06` | `5.913643e-14` | `7.498280e-17` | Matched 20-cell state, grid, geometry, and cache arrays. |
| Single-step solver | DEFERRED | unavailable | unavailable | unavailable | No separately isolated one-step export; do not infer it from the short run. |
| Short simulation | PASS (exec) | `4.718592e+06` | `7.885877e-15` | `2.987034e-16` | Matched 20-cell, fixed-step run to `1e-7 s`, with three frames. |

The large setup and short-run absolute errors occur in density or flux-like
arrays whose values are around `1e19` or similarly large scales. They are
reported rather than hidden; the relative and relative-L2 errors show that the
differences are at floating-point scale.

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

### HLLE historical sound-speed behavior

- Classification: `PYTHON_TRANSLATION_BUG`.
- Python: `src/hallthruster/numerics/flux_functions.py`
- Julia: `src/numerics/flux_functions.jl`
- Observed difference: with the matched unequal states, historical Julia
  evaluates `aR = 177.94316410919706` from `UL`; Python evaluated
  `aR = 251.6496360148078` from `UR`. This changed `sR_min` from
  `-77.94316410919706` to `-151.6496360148078` and changed all three HLLE
  entries.
- Root cause: as with Rusanov, historical Julia deliberately or accidentally
  calls `sound_speed(UL, fluid)` for both sides. The Python translation had
  replaced the second argument with the conventional right state.
- Fix: reproduce the historical left-state call exactly; no equation,
  tolerance, or harness input was changed.
- Regression:
  `test_hlle_matches_executable_historical_left_sound_speed_fixture`, using
  values generated by HallThruster.jl at `014a12f` under Julia 1.10.11.

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

For the deterministic four-cell, three-charge initialization case, executable
comparison gives:

| Array | Max absolute error | Max relative error | Relative L2 error |
|---|---:|---:|---:|
| Heavy-species state | `5.421011e-20` | `1.788993e-16` | `3.111064e-17` |
| Electron density | 0 | 0 | 0 |
| Electron energy density | 0 | 0 | 0 |
| Electron temperature | 0 | 0 | 0 |

For all three saved frames of the matched short simulation, executable
comparison gives:

| Field | Max absolute error | Max relative error | Relative L2 error |
|---|---:|---:|---:|
| Saved times | `0` | `0` | `0` |
| Neutral density (`nn`) | `0` | `0` | `0` |
| Ion density (`ni`) | `2.560000e+02` | `6.517716e-16` | `1.076509e-16` |
| Ion flux (`niui`) | `4.718592e+06` | `1.338462e-15` | `2.987202e-16` |
| Ion velocity (`ui`) | `2.182787e-11` | `1.221258e-15` | `3.581579e-16` |
| Electron density (`ne`) | `2.560000e+02` | `6.517716e-16` | `1.076509e-16` |
| Electron temperature (`Tev`) | `1.065814e-14` | `7.042426e-16` | `2.359758e-16` |
| Potential gradient (`∇ϕ`) | `1.091394e-11` | `7.885877e-15` | `4.854247e-16` |
| Discharge current (`Id`) | `2.664535e-15` | `4.719107e-16` | `2.962646e-16` |
| Ionization frequency (`νiz`) | `1.746230e-10` | `1.400053e-15` | `2.876816e-16` |
| Collision frequency (`νc`) | `3.725290e-09` | `9.814356e-16` | `5.092375e-17` |

The standalone M1 smoke script still reports:

```text
retcode=success
requested_end_time=1e-07
actual_end_time=1e-07
saved_frames=3
finite_state_arrays=True
```

For this matched 20-cell, fixed-step short simulation, the Python translation
reproduces the exported historical Julia quantities to floating-point
precision. This result is specific to this small case and is not a claim of
full scientific equivalence.

## Automated coverage and remaining work

The M2 tests are:

- `tests/test_julia_source_parity.py`
- `tests/test_initialization_parity.py`
- `tests/test_solver_component_parity.py`

They augment, rather than replace, the M1 suite. Compact expected values are
documented in the tests with the exact reference commit; no large generated
outputs are committed. Some edge cases, including zero-density IEEE behavior,
flat-slope reconstruction, and the Config overload, remain source-level tests
because those specific cases are not part of the executable JSON export.

Remaining or deliberately deferred validation is grouped as follows:

- **Indexing:** restart mapping for multiple charge states and less-used
  one-based ranges still need executable/restart cases.
- **Setup:** the deterministic setup export passes; restart setup is not covered.
- **Solver:** the matched short run passes, but an independently isolated
  nonzero single-step export remains deferred.
- **Numerical:** defensive zero-frequency branches outside the validated flux
  path need executable Julia edge-case checks before alteration.
- **Postprocessing:** thrust/current/efficiency parity remains deferred.
- **JSON:** the historical Python JSON entry path remains the existing xfail.
- **Performance:** no optimization or runtime parity work was performed.
- **Scientific validation:** the 200-cell, `1e-3 s` historical regression has
  not been run or used for tuning.
- **Compatibility:** modern `upstream/main` behavior is outside this historical
  translation target.

The translated Python implementation therefore has executable parity with
HallThruster.jl commit `014a12f` for the exported components, deterministic
initialization/setup state, and the matched short fixed-step simulation. Full
historical regression, postprocessing metrics, long-duration stability, and
performance comparison remain separate validation milestones.
