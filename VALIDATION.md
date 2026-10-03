# HallThruster.jl 0.18.5-era translation validation

## Scope and reference

This document records M2 component validation and M3 historical-regression
validation of the Python translation against:

```text
UM-PEPL/HallThruster.jl
commit 014a12fb193af6927cb10f77da5e7baf215b5bc0
```

The Python foundation is commit `bb00d8ce`; the M2 source-parity checkpoint is
`d10696e8`; the M2 executable-parity checkpoint is `a9c4740d`; and the M3
historical-regression validation was committed as `7b9e1b34` and subsequently
merged into `main`. HallThruster.jl and its authors remain the source of the
physical model, equations, numerical methods, SPT-100 regression case, and
original architecture.

This is not a claim of general scientific equivalence. M3 covers one exact
historical SPT-100 regression configuration; other configurations, longer
runs, modern upstream versions, and performance optimization remain outside
the validated scope.

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
- **Historical-regression validation:** the executable 200-cell, `1e-3 s`
  SPT-100 case and its upstream scalar acceptance criteria.
- **Deferred scientific validation:** other operating points, model choices,
  longer-duration stability, current upstream compatibility, and performance
  optimization.

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
- **Postprocessing:** M2 deferred thrust/current/efficiency parity; the M3
  section below validates those metrics for the historical SPT-100 case.
- **JSON:** the historical Python JSON entry path remains the existing xfail.
- **Performance:** M2 performed no optimization; M3 records a baseline runtime
  without changing the implementation for speed.
- **Scientific validation:** M2 stopped before the 200-cell case. The M3
  section below records that regression without tuning physical inputs.
- **Compatibility:** modern `upstream/main` behavior is outside this historical
  translation target.

At the M2 checkpoint, the translated Python implementation therefore had
executable parity with HallThruster.jl commit `014a12f` for the exported
components, deterministic initialization/setup state, and the matched short
fixed-step simulation. M3 extends, but does not replace, that evidence below.

## Historical SPT-100 full regression

### Reference case and execution method

M3 executes `test/regression/baseline.json` and the acceptance logic in
`test/regression/spt100.jl` and `test/regression/regression_utils.jl` from the
exact `014a12f` archive. The fresh Julia reference used Julia 1.10.11 and
HallThruster.jl 0.18.5. Python 3.12.3 ran from the repository-local `.venv`.
Neither current `upstream/main` nor the unfinished Python JSON runner was used.

The matched configuration is:

| Setting | Historical value |
|---|---:|
| Thruster geometry | SPT-100 name; `0.035 / 0.05 / 0.025 m` inner radius, outer radius, channel length |
| Grid and domain | `EvenGrid(200)`, `0.0` to `0.08 m` |
| Charge states | 3 |
| Mass flow / discharge voltage | `5e-6 kg/s` / `300 V` |
| Cathode coupling voltage | `30 V` |
| Neutral velocity / temperature | `278.031 m/s` / `300 K` |
| Ion temperature | `1000 K` |
| Anode / cathode electron temperature | `3.26173 eV` / `3.26173 eV` |
| Background | `1e-5 Torr`, `300 K` |
| Anomalous transport | `LogisticPressureShift(GaussianBohm(...))` with the values in `baseline.json` |
| Wall model | `WallSheath(BNSiO2, loss_scale=0.86904)` |
| Plume / ion wall loss / electron-ion collisions | enabled / enabled / enabled |
| Neutral ingestion multiplier | `6.23341` |
| Initial `dt` / adaptive / CFL | `5e-9 s` / true / `0.799` |
| Timestep bounds | `1e-10` to `1e-7 s`; 100 small steps |
| Duration / saves | `1e-3 s` / 1000 frames |

The direct Python harness constructs `Config` and `SimParams` from those data.
It has an M3-only atomic checkpoint path because the untranslated performance
requires several hours in this environment. The checkpoint contains `U`, all
cache/parameter arrays, time, adaptive-control flags, iteration, save index,
and saved frames. A normal test forces a process-boundary pickle/reload after
two accepted steps and proves the resumed result is bit-for-bit identical to
the ordinary direct solver for a 20-cell case. Checkpointing does not alter the
production solver or any physical input.

The historical statistics are reproduced exactly, including two quirks:

- tail quantities and averaged profiles use Julia frames 333 through 1000,
  which is 668 frames;
- their standard errors divide by `sqrt(667)`, while efficiency means use all
  1000 frames, as in the historical test.

Thrust and current targets use the fresh-run standard error as absolute
tolerance. Efficiencies and peak averaged profiles use historical `rtol=1e-2`.

### Completion and repeatability

Both languages completed successfully:

| Result | Julia | Python |
|---|---:|---:|
| Retcode | success | success |
| Reported final time | `1e-3 s` | `1e-3 s` |
| Accepted steps | 172,613 | 172,613 |
| Saved frames | 1000 | 1000 |
| Initial requested `dt` | `5e-9 s` | `5e-9 s` |
| Initial internal / minimum saved `dt` | `2.220446049250313e-14 s` | same |
| Maximum saved `dt` | `1.1538191002533778e-8 s` | `1.1538186046488877e-8 s` |

Two Julia runs produced identical accepted-step counts, adaptive extrema, and
all exported scalar metrics (zero measured difference). Their timed solver
runs were 31.6763 s and 28.5281 s. No warnings were recorded by either final
run. The saved-`dt` histories have relative-L2 difference `2.064865e-5`; save
times differ by at most `2.168404e-19 s`.

Only `dt` values stored at the 1000 save frames are available in the export.
The minimum above is the intentionally tiny initial cached value, not a claim
that every accepted timestep was recorded.

### Historical scalar regression

All fresh Julia and Python values pass the original historical test criteria.
The last column is fresh Python versus fresh Julia, not error versus the
rounded stored target.

| Quantity | Stored target | Fresh Julia | Fresh Python | Julia/Python relative error |
|---|---:|---:|---:|---:|
| Thrust (mN) | 87.304 | 87.30356894 | 87.30352788 | `4.702240e-7` |
| Discharge current (A) | 4.614 | 4.613871731 | 4.613868525 | `6.948353e-7` |
| Ion current (A) | 3.922 | 3.921837549 | 3.921834971 | `6.573864e-7` |
| Maximum electron temperature (eV) | 24.832 | 24.83210321 | 24.83213205 | `1.161662e-6` |
| Maximum electric field (V/m) | 66,650 | 66,646.07945 | 66,646.12744 | `7.200894e-7` |
| Maximum neutral density (m^-3) | `2.088e19` | `2.088229731e19` | `2.088229726e19` | `2.402370e-9` |
| Maximum ion density (m^-3) | `9.651e17` | `9.650560007e17` | `9.650557878e17` | `2.205553e-7` |
| Mass efficiency | 0.954 | 0.954283531 | 0.954283099 | `4.524926e-7` |
| Current efficiency | 0.873 | 0.872832872 | 0.872833243 | `4.252783e-7` |
| Divergence efficiency | 0.949 | 0.949137454 | 0.949137493 | `4.121172e-8` |
| Voltage efficiency | 0.661 | 0.660914999 | 0.660915187 | `2.841885e-7` |
| Anode efficiency | 0.5656 | 0.564569785 | 0.564569880 | `1.686929e-7` |

No coefficient or operating parameter was adjusted toward these targets.

### Trajectory and profile comparison

Early and midpoint arrays remain close to floating-point scale, followed by a
smoothly amplified late phase difference in this oscillatory solution:

| Saved physical time | Overall max abs | Overall max rel* | Relative L2 |
|---:|---:|---:|---:|
| 0 | `2.621440e6` | `9.947990e-15` | `1.274726e-16` |
| `1.001001e-6` | `8.451523e8` | `3.676981e-13` | `4.554096e-14` |
| `1.001001e-5` | `1.533542e7` | `7.311147e-12` | `2.283763e-14` |
| `1.001001e-4` | `5.145887e8` | `1.051778e-11` | `1.495378e-13` |
| `5.005005e-4` | `1.318689e10` | `4.177008e-10` | `3.105864e-12` |
| `1e-3` | `8.326781e18` | `1.299731e-2` | `4.411136e-4` |

`max rel*` ignores reference magnitudes below `1e-12` of each field maximum;
the absolute errors are shown because density and flux arrays reach very large
scales. At the final instantaneous frame, the largest individual-field
relative-L2 error is `1.302619e-3` for electric field. This is not hidden by
the much smaller time-averaged errors.

Saved histories show gradual rather than abrupt separation. For example, the
saved `dt` first exceeds `1e-12` relative difference near `6.3e-5 s`, `1e-8`
near `6.3e-4 s`, and `1e-6` near `6.7e-4 s`. Discharge-current relative error
first exceeds `1e-8` near `6.27e-4 s` and `1e-6` near `8.85e-4 s`. Together
with the M2 component/setup agreement and identical accepted-step counts, this
supports classification as cross-language numerical/phase sensitivity rather
than a newly located translation branch or parameter mismatch.

Time-averaged axial-profile errors are:

| Field | Max absolute error | Max relative error* | Relative L2 error |
|---|---:|---:|---:|
| Electric field | `3.317664e-1` | `3.814943e-4` | `5.138842e-6` |
| Ion flux | `1.939820e16` | `3.229953e-4` | `1.602786e-6` |
| Ion velocity | `1.022977e-1` | `3.562522e-5` | `6.659293e-7` |
| Electron temperature | `7.173632e-5` | `1.799119e-5` | `2.304299e-6` |
| Electron density | `2.136789e12` | `9.865448e-6` | `1.092194e-6` |
| Ion density | `2.138204e12` | `6.413677e-4` | `1.090252e-6` |
| Magnetic field | `1.040834e-17` | `1.022940e-15` | `1.418270e-16` |
| Neutral density | `1.281957e12` | `9.654076e-7` | `4.775431e-8` |
| Electric potential | `2.629015e-4` | `7.407341e-6` | `5.186442e-7` |

The strict M2-style `1e-8` comparator is retained and exits 1 for the full
long run. A second reported M3 envelope exits 0 with limits of `2e-6` for
scalar relative error, `5e-5` for saved-`dt` relative L2, `1e-5` for averaged
profiles, and `2e-3` for instantaneous checkpoint fields. These are post-hoc
diagnostic envelopes, set just above the measured errors after locating their
smooth growth; they are not independent pre-registered tolerances and do not
replace the original historical acceptance criteria. Every envelope remains
below the historical 1% profile/efficiency tolerance, but only the stored
historical criteria provide an independent regression pass.

### Confirmed M3 translation repairs

Two defects blocked or invalidated this exact case before comparison:

- `src/hallthruster/simulation/plume.py` compared the translated list-valued
  `grid.cell_centers` directly with a float. Historical Julia performs a
  vectorized comparison. Converting the centers to a NumPy array restores that
  behavior; `test_plume_update_accepts_translated_list_cell_centers` covers it.
- `src/hallthruster/simulation/postprocess.py` mixed Julia one-based and Python
  zero-based frame indices, treated `Config` and `Grid1D` as dictionaries, and
  referenced an undefined `config`. The smallest consistent repair uses
  zero-based public Python frame indices and attribute access while retaining
  the historical formulas. `tests/test_postprocess.py` covers thrust, charge-
  weighted ion current, efficiencies, frame conversion, and averaging.

The long-run phase difference was not treated as a translation defect because
no first discontinuous calculation was found: machine-scale differences grow
continuously while configuration, control flow, accepted-step count, M2
components, and early trajectory remain matched.

### Runtime and generated inspection data

Hardware was an x86-64 Intel Core i7-7700HQ (4 cores / 8 threads) in the Linux
devbox. Julia's two in-process timed simulations took 31.6763 s and 28.5281 s;
the latter is the warmed figure. The pure-Python checkpointed validation took
21,540.7 s (about 5.98 h). That Python value includes checkpoint serialization
overhead and excludes work lost in earlier infrastructure-interrupted
attempts. These are context measurements, not a controlled performance
benchmark, and no optimization was attempted.

Ignored generated results include both JSON exports, strict and documented-
envelope summaries, the 57 MB Python restart checkpoint, and nine PNG plots:
neutral/electron/ion densities, electron temperature, potential, electric and
magnetic fields, discharge current history, and saved adaptive timestep.

### M3 claim boundary

The Python translation reproduces the historical HallThruster.jl `014a12f`
SPT-100 regression within the original stored regression criteria for every
validated scalar and averaged-profile peak. Fresh Julia/Python time-averaged
scientific outputs agree at roughly `1e-6` relative scale, while the final
instantaneous oscillatory state exhibits quantified phase drift up to
`1.3e-3` relative L2 in an individual field. This one case does not establish
equivalence for all configurations.

Still deferred are other thrusters and operating points, longer-duration
stability, complete per-accepted-step adaptive histories, JSON-runner repair,
broader postprocessing cases, performance optimization, and compatibility
with modern HallThruster.jl.
