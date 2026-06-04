# Quasi-Maximum Likelihood Estimation for Scalable ARMA Models

This repository contains the code, final tables, selected result files, and
paper figures for the manuscript:

**Yuchang Lin, Wenyu Li, and Qianqian Zhu. _Quasi-maximum likelihood
estimation for scalable ARMA models_. Manuscript, 2026.**

The accompanying paper draft is included at
[`paper/SARMA_arXiv.pdf`](paper/SARMA_arXiv.pdf).

## Paper Summary

The paper studies scalable autoregressive moving average (SARMA) models for
multivariate time series. Existing scalable ARMA work mainly relies on
regularized least squares estimation, which is statistically less efficient and
typically requires sub-Gaussian assumptions. This paper develops a
quasi-maximum likelihood estimation (QMLE) framework for SARMA models.

The repository reproduces the paper-facing code and outputs for:

- QMLE estimation and asymptotic-variance calculations.
- A block coordinate descent algorithm for SARMA fitting.
- BIC-based order selection for the SARMA order `(p, r, s)`.
- Simulation tables for finite-sample estimation, BIC selection, and ARE.
- The six-variable FRED-MD empirical application and forecast comparison.

## Citation

If you use this code or the accompanying results, please cite the manuscript.
Until an arXiv identifier or journal DOI is available, use the following
provisional BibTeX entry:

```bibtex
@misc{lin2026sarmaqmle,
  title  = {Quasi-maximum likelihood estimation for scalable {ARMA} models},
  author = {Lin, Yuchang and Li, Wenyu and Zhu, Qianqian},
  year   = {2026},
  note   = {Manuscript}
}
```

Once an arXiv identifier or final publication record is available, replace the
`note` field with the corresponding arXiv or journal information.

## Package Scope

This branch is a clean reproduction package for the current paper version. It
is intentionally smaller than the full working directory: remote launch scripts,
monitor scripts, diagnostics, smoke tests, historical scale-sensitivity runs,
and experiments not reported in the manuscript are excluded.

The source commit recorded for this package is:

```text
b69beca
```

The package was generated on 2026-06-03 and the README was updated on
2026-06-04.

## Repository Layout

- `paper/SARMA_arXiv.pdf`: current arXiv-style manuscript draft.
- `code/src`: canonical SARMA implementation used by the experiments.
- `code/tests11`: paper simulation runners and minimum table aggregators.
- `code/Application`: six-variable FRED-MD empirical runner.
- `code/data/FRED-MD.csv`: real-data source used by the application.
- `results/simulations`: final paper-facing simulation tables and configs.
- `results/realdata`: final paper-facing empirical SARMA/VAR run.
- `paper_tables/main`: manuscript-facing CSV/TEX tables.
- `paper_tables/from_manuscript`: table environments extracted from the current
  manuscript source for direct comparison.
- `paper_figures`: real-data figures used by the manuscript.
- `MANIFEST.csv`: file sizes and SHA256 checksums for the release package.

## Paper Setting Map

- DGP1 estimation table:
  `results/simulations/exp1_qmle/`
- DGP2 BIC table:
  `results/simulations/exp2_bic/`
- DGP3 ARE table:
  `results/simulations/exp3_are/`
- Real-data empirical result:
  `results/realdata/empirical6_bic10/`

Historical local script and result names may contain older labels such as
`DGP2`, `DGP3`, or `DGP4`. The paper labels above are the authoritative labels
for this public release package.

## Installation

Create a Python environment and install the listed dependencies:

```bash
pip install -r requirements.txt
```

Run commands from the repository root with:

```bash
export PYTHONPATH="$(pwd)/code"
export SARMA_COMMIT=b69beca
```

## Reproduction Entry Points

The included result tables can be inspected directly without rerunning the
expensive simulations. To rerun or inspect the configurable runners:

```bash
python code/tests11/run_dgp1.py --help
python code/tests11/run_dgp3_bic.py --help
python code/tests11/run_dgp2_are.py --help
python code/Application/run_empirical6_notebook.py --help
```

Main outputs already included in the release:

- `paper_tables/main/tab_DGP1_qmle_x10.csv`
- `paper_tables/main/tab_BIC_paper_lam07_09.csv`
- `paper_tables/main/tab_ARE_paper_a0_06_09.csv`
- `paper_tables/main/table_forecast_from_manuscript.csv`
- `paper_tables/main/table_forecast_recomputed_from_bic10_metrics.csv`

## Real-Data Note

The available scale-1 empirical result in this package reproduces the selected
SARMA order `(1,1,0)` and contains SARMA/VAR metrics. The manuscript forecast
table also reports VARMA1/VARMA2, but an exact scale-1 VARMA metrics source was
not found in the current local result tree. Therefore:

- `paper_tables/main/table_forecast_from_manuscript.csv` preserves the values
  currently pasted in the manuscript.
- `paper_tables/main/table_forecast_recomputed_from_bic10_metrics.csv`
  recomputes SARMA/VAR relative improvements from the available scale-1
  metrics.

The manuscript formula uses RI-RMSFE, while the available metrics reproduce the
displayed SARMA percentages only when using MSE-relative improvement. This
point should be reviewed before final journal release.

## Contact

For questions about the code or manuscript, please contact the corresponding
author:

Qianqian Zhu, School of Statistics and Data Science, Shanghai University of
Finance and Economics, Shanghai, China.

Email: `zhu.qianqian@mail.shufe.edu.cn`
