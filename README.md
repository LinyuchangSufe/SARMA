# SARMA Paper Reproduction Package

Generated: 2026-06-03T15:25:58

This directory is a clean shareable subset for the current `sinica/SARMA_main.tex`
simulation and real-data settings. It is additive: no original files were
deleted or moved.

## Layout

- `code/src`: canonical SARMA implementation used by the experiments.
- `code/tests11`: only the paper simulation runners and minimum table aggregators.
- `code/Application`: six-variable FRED-MD real-data runner.
- `code/data/FRED-MD.csv`: real-data source used by the application.
- `results/simulations`: final paper-facing simulation tables/configs.
- `results/realdata`: final paper-facing empirical SARMA/VAR run.
- `paper_tables/main`: manuscript-facing CSV/TEX tables.
- `paper_tables/from_manuscript`: table environments extracted from the current
  manuscript source for direct comparison.
- `paper_figures`: real-data figures used by the manuscript.

## Paper Setting Map

- Paper DGP1 result source: `code/tests11/results/Exp1/diag_balanced_trueinit_lamm0p8_gam0p8_T500_750_1000_n500_dgpseed1012112_20260602`
- Paper DGP2 / BIC result source: `code/tests11/results/Exp3/combined_strong_symmetric_T500_750_1000_lam06_09_n500_dgpseed1012112_20260601`
- Paper DGP3 / ARE result source: `code/tests11/results/Exp2/diag_balanced_true_sample_T20000_dgpseed1012112_20260602`
- Real data result source: `code/Application/results/empirical6_notebook/empirical6_bic10_sarma_var_only_20260530`

Historical script/result names still contain older labels such as DGP2 or DGP4.
The paper names above are the authoritative labels for this package.

## Reproduction Entry Points

Run from this package root:

```bash
export PYTHONPATH="$(pwd)/code"
export SARMA_COMMIT=b69beca
python code/tests11/test_sarma_core.py
```

Main experiment runners:

```bash
python code/tests11/run_dgp1.py --help
python code/tests11/run_dgp3_bic.py --help
python code/tests11/run_dgp2_are.py --help
python code/Application/run_empirical6_notebook.py --help
```

Remote `.sh` launch scripts, monitors, diagnostics, smoke tests, historical
scale-sensitivity runs, and experiments not shown in the manuscript were
intentionally excluded.

## Important Real-Data Note

The available scale-1 empirical result in this package reproduces the current
selected SARMA order `(1,1,0)` and contains SARMA/VAR metrics. The current
manuscript forecast table also reports VARMA1/VARMA2, but an exact scale-1
VARMA metrics source was not found in the current local result tree. Therefore:

- `paper_tables/main/table_forecast_from_manuscript.csv` preserves the values
  currently pasted in the manuscript.
- `paper_tables/main/table_forecast_recomputed_from_bic10_metrics.csv`
  recomputes SARMA/VAR relative improvements from the available scale-1 metrics.

Also note that the manuscript formula uses RI-RMSFE, while the available
metrics reproduce the displayed SARMA percentages only when using MSE-relative
improvement. This should be reviewed before public release.

## Manifest

`MANIFEST.csv` records source path, destination path, file size, and SHA256 for
each copied or generated file.
