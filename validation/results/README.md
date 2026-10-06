# Committed validation summary figures

The figures in this directory use only the numerical M3 summary metrics
committed in [VALIDATION.md](../../VALIDATION.md). The underlying Julia and
Python raw M3 outputs were generated during the completed validation, but were
intentionally gitignored and are no longer retained in the repository.

The committed PNG files are reproducibly regenerated from the immutable,
documented M3 summary metrics encoded in `generate_summary_plots.py`:

```bash
python validation/results/generate_summary_plots.py
```

The script produces exactly these presentation artifacts:

- `scalar_relative_errors.png` — selected scalar Julia/Python relative errors;
- `profile_relative_l2.png` — time-averaged profile relative-L2 errors; and
- `trajectory_relative_l2.png` — total trajectory relative-L2 at documented
  physical-time checkpoints.

These are plots of committed summary values, not regenerated raw-simulation
profiles or fabricated Julia/Python overlays. They do not replace the raw-data
validation methodology, comparator results, or claim boundaries documented in
[VALIDATION.md](../../VALIDATION.md).
