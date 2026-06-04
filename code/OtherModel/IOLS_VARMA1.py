import numpy as np

def varma11_iols_fit_no_c(
    y: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
    burn_in: int = 0,
    ridge: float = 0.0,
):
    """
    Fit VARMA(1,1) by IOLS, no intercept:
        y_t = A y_{t-1} + B u_{t-1} + u_t
    y: (T, N), already demeaned
    Returns A,B,resid u (T,N), info
    """
    y = np.asarray(y, dtype=float)
    T, N = y.shape
    if T < 3:
        raise ValueError("Need T>=3.")

    y_lag = y[:-1, :]   # (T-1, N)
    y_cur = y[1:, :]    # (T-1, N)

    # init: VAR(1) without intercept, y_cur ≈ y_lag @ A0_T
    A0_T, *_ = np.linalg.lstsq(y_lag, y_cur, rcond=None)  # (N, N) in beta orientation
    u = np.zeros((T, N))
    u[1:, :] = y_cur - y_lag @ A0_T

    # regression rows: drop early usable rows if burn_in>0
    start_t = 1 + max(0, burn_in)          # time index t >= start_t
    rows = np.arange(T-1)                  # row r corresponds to time t=r+1
    rows = rows[rows >= start_t - 1]

    converged = False
    last_diff = None
    it_used = 0

    for k in range(max_iter):
        it_used = k + 1
        u_lag = u[:-1, :]                  # (T-1, N)

        X = np.hstack([y_lag, u_lag])      # (T-1, 2N)
        Xr = X[rows, :]
        Yr = y_cur[rows, :]

        if ridge > 0:
            XtX = Xr.T @ Xr
            XtY = Xr.T @ Yr
            beta = np.linalg.solve(XtX + ridge * np.eye(2 * N), XtY)  # (2N, N)
        else:
            beta, *_ = np.linalg.lstsq(Xr, Yr, rcond=None)

        u_new = np.zeros_like(u)
        u_new[1:, :] = y_cur - X @ beta

        diff = np.linalg.norm(u_new - u, ord="fro")
        last_diff = diff
        u = u_new

        if diff < tol:
            converged = True
            break

    # beta = [A^T; B^T]
    A = beta[0:N, :].T
    B = beta[N:2*N, :].T

    info = {"converged": converged, "n_iter": it_used, "final_resid_diff": last_diff}
    return A, B, u, info

def varma11_iols_forecast1(y: np.ndarray, A: np.ndarray, B: np.ndarray, u: np.ndarray):
    """One-step ahead forecast y_{T} given last y_{T-1}, u_{T-1}."""
    y_last = y[-1, :]
    u_last = u[-1, :]
    return A @ y_last + B @ u_last


import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX

def build_varmax_start_params_from_AB(mod: VARMAX, A: np.ndarray, B: np.ndarray):
    """
    Fill VARMAX start_params with given A,B for VARMA(1,1), trend='n'.
    Works with pandas columns (recommended).
    """
    sp = np.array(mod.start_params, dtype=float)
    names = mod.param_names
    cols = list(mod.endog_names)  # variable names

    name_to_idx = {n: i for i, n in enumerate(names)}
    N = len(cols)
    assert A.shape == (N, N) and B.shape == (N, N)

    # AR: 'L1.<reg>.<dep>'
    for dep_i, dep in enumerate(cols):
        for reg_j, reg in enumerate(cols):
            key = f"L1.{reg}.{dep}"
            if key in name_to_idx:
                sp[name_to_idx[key]] = A[dep_i, reg_j]

    # MA: 'L1.e(<reg>).<dep>'
    for dep_i, dep in enumerate(cols):
        for reg_j, reg in enumerate(cols):
            key = f"L1.e({reg}).{dep}"
            if key in name_to_idx:
                sp[name_to_idx[key]] = B[dep_i, reg_j]

    return sp

def varmax_fit_forecast1_with_iols_start(y: np.ndarray, A: np.ndarray, B: np.ndarray,
                                        maxiter: int = 200):
    """
    Fit VARMAX(1,1) trend='n' using IOLS A,B as start_params, then forecast 1-step.
    y: (T,N)
    """
    y = np.asarray(y, dtype=float)
    T, N = y.shape
    # use DataFrame to get stable param_names with variable labels
    df = pd.DataFrame(y, columns=[f"y{i+1}" for i in range(N)])

    mod = VARMAX(df, order=(1, 1), trend="n")  # no intercept
    sp = build_varmax_start_params_from_AB(mod, A, B)

    res = mod.fit(start_params=sp, method="lbfgs", maxiter=maxiter, disp=False)
    y_next = np.asarray(res.forecast(steps=1)).reshape(1, N)[0]
    return y_next, res
