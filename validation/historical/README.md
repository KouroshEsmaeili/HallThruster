# Historical SPT-100 regression harness

This harness runs the `test/regression/baseline.json` case from
HallThruster.jl commit `014a12fb193af6927cb10f77da5e7baf215b5bc0` through
the historical Julia implementation and the direct Python API.

From the repository root, with `.venv` active and `REFERENCE_DIR` pointing to
an instantiated archive of the exact historical commit:

```bash
julia --project="$REFERENCE_DIR" \
  validation/historical/julia/export_regression.jl \
  validation/outputs/historical_julia.json \
  --baseline "$REFERENCE_DIR/test/regression/baseline.json" \
  --repeat 2

python validation/historical/python/export_regression.py \
  validation/outputs/historical_python.json \
  --checkpoint validation/outputs/historical_python_checkpoint.pkl \
  --checkpoint-steps 5000

# If that process is interrupted, use the same case arguments and resume it:
python validation/historical/python/export_regression.py \
  validation/outputs/historical_python.json \
  --checkpoint validation/outputs/historical_python_checkpoint.pkl \
  --checkpoint-steps 5000 \
  --resume

# Strict cross-language comparison (retained as a diagnostic):
python validation/historical/compare_regression.py \
  validation/outputs/historical_julia.json \
  validation/outputs/historical_python.json \
  --summary validation/outputs/historical_comparison_strict.json

# Documented long-run oscillatory envelope; see VALIDATION.md for rationale:
python validation/historical/compare_regression.py \
  validation/outputs/historical_julia.json \
  validation/outputs/historical_python.json \
  --translation-rtol 2e-6 \
  --adaptive-l2-tolerance 5e-5 \
  --profile-l2-tolerance 1e-5 \
  --trajectory-l2-tolerance 2e-3 \
  --summary validation/outputs/historical_comparison.json

python validation/historical/plot_regression.py \
  validation/outputs/historical_julia.json \
  validation/outputs/historical_python.json \
  validation/outputs/historical_plots
```

Generated JSON and plots under `validation/outputs/` are ignored. The normal
Python test suite does not require Julia or the generated full-regression data.
The checkpoint is also ignored. A normal Python test verifies that a forced
checkpoint/pickle/reload continuation is bit-for-bit identical to the ordinary
direct solver for a small matched case.

The strict `1e-8` comparison intentionally remains available and does not pass
the late instantaneous oscillatory state. The second command records the
post-diagnostic M3 envelope; it does not replace the independent tolerances in
the historical Julia regression.
