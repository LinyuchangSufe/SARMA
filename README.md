# SARMA: Scalable AutoRegressive Moving Average Model

A comprehensive implementation of the Seasonal AutoRegressive Moving Average (SARMA) model with advanced estimation methods, order selection, and empirical applications to macroeconomic data.

## Overview

This repository provides a modern, modular implementation of SARMA models with the following key components:

- **SARMA Estimator**: High-level Python interface for model fitting and forecasting
- **Block-Coordinate Descent (BCD)**: Both LS and MLE estimation algorithms
- **Automatic Order Selection**: BIC-based model selection with parallel computation
- **Empirical Applications**: Real-world examples using FRED macroeconomic data
- **Model Comparison**: Benchmarking against VAR, VARMA, and AR models

## Features

### Core Modeling Capabilities

- **Flexible SARMA Specification**: Supports arbitrary (p, r, s) orders where:
  - `p`: AR order
  - `r`: Number of seasonal AR magnitudes
  - `s`: Number of seasonal pairs
  
- **Dual Estimation Methods**:
  - Least Squares (LS) estimation with identity covariance
  - Maximum Likelihood Estimation (MLE) with estimated covariance
  
- **Multi-start Optimization**: Robust parameter initialization across multiple starting points

- **Automatic Order Selection**: BIC-based criteria with parallel joblib support

- **Forecasting**: One-step and multi-step ahead predictions using tensor representations

## Installation

### Prerequisites

- Python 3.8+
- NumPy, SciPy, Pandas
- Joblib (for parallel processing)

### Setup

```bash
git clone https://github.com/LinyuchangSufe/SARMA.git
cd SARMA
pip install -r requirements.txt
```

### Dependencies

Key packages (see `requirements.txt`):
- numpy==1.24.3
- scipy==1.10.1
- pandas==2.0.1
- scikit-learn (implied by joblib)
- matplotlib==3.7.1
- statsmodels==0.14.0
- torch==2.0.1 (optional, for future extensions)

## Project Structure

```
SARMA/
├── src/
│   ├── sarma/
│   │   ├── estimator.py          # High-level SARMAEstimator class
│   │   ├── optim.py              # BCD_SARMA optimization engine
│   │   ├── selection.py          # BIC order selection with parallel support
│   │   ├── param_utils.py        # Parameter utilities
│   │   └── __init__.py
│   └── utils/
│       ├── help_function.py      # Core utility functions (get_A, gen_X_AR, etc.)
│       ├── tensorOp.py           # Tensor operations (unfold, fold, matricization)
│       └── __init__.py
├── Application/
│   ├── Empirical example.ipynb   # Real-world FREDMD example
│   ├── Compare_method.py         # Benchmark against VAR, VARMA, AR
│   ├── current_analysis.py       # Data preparation and analysis
│   ├── MACM_realdata.pdf         # Matrix Autocorrelation Function results
│   └── coef_G.pdf
├── data/
│   ├── FRED-MD.csv              # Monthly macroeconomic indicators
│   ├── FRED-QD.csv              # Quarterly macroeconomic indicators
│   ├── FRED_process.py          # Data preprocessing scripts
│   └── macro20.csv
├── OtherModel/
│   ├── IOLS_VARMA1.py           # Iterative OLS for VARMA comparison
│   └── __pycache__/

```

## Quick Start

### Basic Usage

```python
import numpy as np
from src.sarma.estimator import SARMAEstimator

# Load your time series data
y = np.load('your_data.npy')  # shape: (T, N)

# Create and fit estimator
estimator = SARMAEstimator(
    P=200,          # truncation parameter
    n_iter=100,     # BCD iterations
    stop_thres=1e-6,
    verbose=True
)

# Automatic order selection + fitting
estimator.fit(y, n_jobs_BIC=4)  # Uses BIC to select (p,r,s)

# Get fitted parameters
params = estimator.get_params()
print(f"Selected orders: p={params['p']}, r={params['r']}, s={params['s']}")

# Forecast
y_forecast = estimator.predict(y, steps=10)
print(y_forecast.shape)  # (10, N)

# Model summary
print(estimator.summary())
```

### Specifying Orders Manually

```python
# If you know the orders, fit directly
estimator.fit(y, p=1, r=1, s=0, n_jobs_BIC=1)

# Or use a different estimation method
estimator.SARMA_fitBCD_SARMA(
    y, p=1, r=1, s=0,
    lmbd=np.array([0.5]),
    eta=np.array([[0.5, np.pi/2]]),
    esti_method='mle',  # 'ls' or 'mle'
    P=150,
    n_iter=100,
    Cal_AsyVar=True
)
```

### Empirical Application: FRED Data

See [Application/Empirical example.ipynb](Application/Empirical%20example.ipynb) for a complete walkthrough:

```python
import pandas as pd
from src.sarma.estimator import SARMAEstimator

# Load FRED macroeconomic data
df = pd.read_csv('data/FRED-MD.csv', index_col=0, parse_dates=True)

# Select variables and apply transformations
vars_ = ['RPI', 'INDPRO', 'UNRATE', 'M2SL', 'CPIAUCSL', 'DPCERA3M086SBEA']
df = df[vars_].iloc[1:]  # skip header row

# Apply FRED standard transformations
# (see Application/current_analysis.py for full preprocessing)

# Standardize
df = (df - df.mean()) / df.std()
y = df.values

# Fit SARMA
est = SARMAEstimator(P=200, n_iter=100, stop_thres=1e-5, verbose=True)
est.fit(y)

# Compare with benchmarks
from Application.Compare_method import var_one_step_forecast, varma_one_step_forecast
```

## Key Components

### SARMAEstimator (estimator.py)

Main user-facing class providing:
- `fit()`: Automatic order selection and model fitting
- `predict()`: Multi-step forecasting
- `get_params()`: Retrieve fitted parameters
- `summary()`: Human-readable model summary

**Constructor Parameters:**
- `P`: Truncation for seasonal design matrices (default: 200)
- `n_iter`: Max BCD iterations (default: 100)
- `stop_thres`: Convergence tolerance (default: 1e-6)
- `grid_mode`: 'full', 'random', or 'auto' for multi-start initialization
- `n_random`: Number of random initializations (default: 2000)
- `seed`: Random seed for reproducibility
- `verbose`: Enable detailed logging

**Output: SARMAFitResult dataclass**
- `loss`: Final loss value
- `p, r, s`: Selected orders
- `lmbd`: Seasonal AR magnitudes (shape: r)
- `eta`: Seasonal parameters (shape: s × 2)
- `G`: VAR coefficient tensor (shape: N × N × d)
- `Sigma`: Estimated covariance matrix (shape: N × N)
- `AsyVar`: Asymptotic variance
- `A`: Lagged operator tensor

### BCD Solver (optim.py)

Block-Coordinate Descent algorithm with:
- Per-parameter optimization for λ (seasonal AR magnitudes)
- Per-pair optimization for η = (γ, φ) (seasonal phase/amplitude)
- Convex LS updates for G given current seasonal parameters
- Optional Sigma re-estimation (MLE mode)

**Key Function:**
```python
BCD_SARMA(y, p, r, s, lmbd=None, eta=None, Sigma=None, 
          esti_method='ls', P=150, n_iter=500, stop_thres=1e-5, 
          Cal_AsyVar=True)
```

### Order Selection (selection.py)

BIC-based model selection with parallel computation:
```python
BIC_parallel_joblib(y, P=200, seed=None, verbose=False, n_jobs_BIC=1)
```

Returns dictionary with:
- `ML_min_index`: Selected (p, r, s)
- `ML_lmbd_value`: Initial λ estimate
- `ML_eta_value`: Initial η estimate
- Full BIC surface and results


## Model Comparison

The [Application/Compare_method.py](Application/Compare%20method.py) module provides benchmark methods:

- `var_one_step_forecast()`: VAR (statsmodels)
- `varma_one_step_forecast()`: VARMA/ARIMA (statsmodels)
- `ar_one_step_forecast()`: Individual AR models (statsmodels)

Example comparison:
```python
from Application.Compare_method import var_one_step_forecast

# SARMA forecast
sarma_pred = estimator.predict(y, steps=1)

# VAR benchmark (p=1)
var_pred = var_one_step_forecast(y, p=1)

# Compare MSE
mse_sarma = np.mean((y_true - sarma_pred)**2)
mse_var = np.mean((y_true - var_pred)**2)
```

## Empirical Data

### FRED-MD (Monthly)
Monthly macroeconomic indicators from the Federal Reserve Economic Data (FRED) database:
- 128+ variables
- Monthly frequency
- Source: [FRED-MD: A Monthly Database for Macroeconomic Research](https://research.stlouisfed.org/wp/more/2015-012)

### FRED-QD (Quarterly)
Quarterly equivalent of FRED-MD dataset

### Example Variables
Common macroeconomic variables with FRED transform codes:
- **RPI**: Retail Price Index (log-difference: code 5)
- **INDPRO**: Industrial Production (log-difference: code 5)
- **UNRATE**: Unemployment Rate (log-difference: code 5)
- **M2SL**: M2 Money Stock (2nd log-difference: code 6)
- **CPIAUCSL**: CPI All Items (2nd log-difference: code 6)

## Performance Characteristics

### Computational Complexity
- **BCD Iterations**: Typically 10-50 iterations for convergence
- **Single Iteration**: O((T-p)Nd²) where d = p+r+2s
- **Order Selection (BIC)**: Parallelizable across (p,r,s) grid
- **Multi-start**: Linear in number of initializations


## License

[Specify your license - e.g., MIT, Apache 2.0]

## Contact & Support

For questions, issues, or suggestions:
- GitHub Issues: [SARMA/issues](https://github.com/LinyuchangSufe/SARMA/issues)
- Email: [lin_yuchang@163.com]

---

**Note**: This implementation is designed for research and educational purposes. For production forecasting applications, consider ensemble methods and robustness checks as presented in the empirical examples.
