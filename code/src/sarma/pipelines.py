from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

# Prefer package-local import first; fallback to repo-relative path
try:
    from .estimator import SARMAEstimator  # type: ignore
except Exception:  # pragma: no cover
    from src.sarma.estimator import SARMAEstimator  # type: ignore

try:
    from sklearn.decomposition import PCA
except Exception as e:  # pragma: no cover
    raise ImportError("pipelines.py requires scikit-learn. Please `pip install scikit-learn`.\n" + str(e))

try:
    from .param_utils import make_estimator_kwargs, make_bic_kwargs
except Exception:
    from src.sarma.param_utils import make_estimator_kwargs, make_bic_kwargs

__all__ = [
    "dynamic_factor_sarma_pipeline",
    "factor_sarma_forecast",
    "eigen_ratio_select",
    "DynamicFactorSARMA",
    "make_estimator_kwargs",
    "make_bic_kwargs",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# def _as_TN(y: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#     """Ensure array shape is (T, N): time along axis 0, variables along axis 1.
#     Accepts numpy array or DataFrame. Copies to avoid surprising inplace edits.
#     """
#     if isinstance(y, pd.DataFrame):
#         y = y.values
#     y = np.asarray(y)
#     if y.ndim != 2:
#         raise ValueError("y must be 2D (T, N) or (N, T)")
#     # Heuristic: if first dim < second dim, likely (N, T) → transpose
#     return y

# ---------------------------------------------------------------------------
# Eigenvalue Ratio (ER) selector for factor number (Ahn & Horenstein, 2013)
# ---------------------------------------------------------------------------

def eigen_ratio_select(y: np.ndarray, k_max: Optional[int] = None) -> Tuple[int, np.ndarray, np.ndarray]:
    """Select the number of factors using the Eigenvalue Ratio method.

    Parameters
    ----------
    y : array (T, N)
        Data matrix (recommend standardized). Time along axis 0.
    k_max : int or None
        Max number of factors to consider. If None, use min(10, N-1).

    Returns
    -------
    k_hat : int
        Selected factor number (>=1).
    eigvals : (N,) ndarray
        Eigenvalues of the column covariance, sorted descending.
    ratios : (N-1,) ndarray
        Adjacent eigenvalue ratios, i.e., eigvals[i] / eigvals[i+1].
    """
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError("y must be 2D (T, N)")
    T, N = y.shape
    if N < 2:
        return 1, np.array([1.0]), np.array([np.inf])

    if k_max is None:
        k_max = min(10, N - 1)
    else:
        k_max = int(max(1, min(k_max, N - 1)))

    # Column covariance; ddof=T-1 by np.cov default when rowvar=False
    cov = y.T @ y / (T - 1)
    # Use symmetric eigensolver, descending
    eigvals = np.linalg.eigvalsh(cov)[::-1]

    # Guard against non-positive small numerical values
    eigvals = np.maximum(eigvals, 1e-12)
    ratios = eigvals[:-1] / eigvals[1:]
    ratios = ratios[:k_max]

    # argmax over 1..k_max → index 0..k_max-1, then +1 to convert to k
    k_hat = int(np.argmax(ratios)) + 1 if ratios.size else 1
    return k_hat, eigvals, ratios





# ---------------------------------------------------------------------------
# Dynamic Factor Model (DFM) → SARMA
#   - choose factor number k (optional via search)
#   - fit SARMA on factors
#   - preserve identifiability by using orthogonal PCA factors
# ---------------------------------------------------------------------------
def dynamic_factor_sarma_pipeline(
    y: Union[np.ndarray, pd.DataFrame],
    X: Union[np.ndarray, pd.DataFrame]= None,
    *,
    k: Optional[int] = None,
    k_max: int = 8,
    k_select: str = "eigen_ratio",
    p: Optional[int] = None,                 
    r: Optional[int] = None,                 
    s: Optional[int] = None,                
    est_kwargs: Optional[Dict[str, Any]] = None,
    bic_kwargs: Optional[Dict[str, Any]] = None,
    init_params: Optional[Dict[str, np.ndarray]] = None,   
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end pipeline for Dynamic Factor Model with SARMA factors.

    Assumes `y` is already standardized if desired.
    If `k` is None, we search k=1..min(k_max, N) using the eigenvalue ratio method
    (Ahn–Horenstein). Identifiability is handled by PCA orthogonal factors (rotation fixed).
    
    Parameters
    ----------
    y : array-like (T, N) or (N, T)
        Input time-series/panel data, assumed standardized if desired.
    X : array-like (T, M) or None
        Exogenous features (optional).
    k : int or None
        Number of factors. If None, the pipeline chooses k automatically.
    k_max : int
        Upper bound for factor search when k is None.
    k_select : {"eigen_ratio"}
        How to choose the factor number when k is None. Currently only
        the Ahn–Horenstein eigenvalue ratio ("eigen_ratio") is supported.
    p, r, s : int or None
        SARMA orders. If None, will use BIC to select (if run_bic=True).
    est_kwargs : dict or None
        Estimator configuration (e.g., method='mle', use_multistart, P, n_iter, ...).
        Use `make_estimator_kwargs(...)` to generate with validation.
    bic_kwargs : dict or None
        BIC-related settings (run_bic, bic_branch, refine_top_k, bic_caps, ...).
        Use `make_bic_kwargs(...)` to generate with validation.
    init_params : dict or None
        Initial parameters for SARMA fitting.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    dict with keys:
        - 'k': chosen factor dimension
        - 'pca': SVD results {'U', 'S', 'Vt'}
        - 'y_lowdim': factor series F (T, k)
        - 'Lambda': loadings (N, k)
        - 'y_proc': input Y (assumed already standardized if desired)
        - 'est': fitted SARMAEstimator on factors
        - 'auto_search': records from k-search (if k was None)
    """
    # Use the object-oriented wrapper as the authoritative implementation
    # so behavior is centralized in `DynamicFactorSARMA`.
    if est_kwargs is None:
        est_kwargs = {}
    if bic_kwargs is None:
        bic_kwargs = {}

    # Create wrapper instance
    dfm = DynamicFactorSARMA(y, X=X, verbose=verbose)

    # Select/confirm factor dimension (method handles both k specified and k=None)
    dfm.select_factors(k=k, k_max=k_max, k_select=k_select)

    # Decompose to obtain factors & loadings
    F_hat, Lambda_hat = dfm.pca_decompose()

    # Fit SARMA on the combined data (X stacked with factors inside the wrapper)
    dfm.fit_sarma(p=p, r=r, s=s, est_kwargs=est_kwargs, bic_kwargs=bic_kwargs, init_params=init_params)

    # Build return pack compatible with previous pipeline output
    pca_svd = dfm.pca_svd if dfm.pca_svd is not None else {"U": None, "S": None, "Vt": None}
    pack = {
        "k": int(dfm.k),
        "pca": pca_svd,
        "y_lowdim": dfm.get_factors(),
        "Lambda": dfm.get_loadings(),
        "y_proc": dfm.y_raw,
        "est": dfm.get_estimator(),
        "auto_search": dfm.auto_search,
    }
    return pack


# ---------------------------------------------------------------------------
# Forecasting on DFM-SARMA pipeline output
#   - Reconstruct next-step forecast for Y by forecasting factors, then
#     inverse-transform through PCA and de-standardization
#   - If SARMAEstimator exposes .predict, use it. Otherwise, do AR-only fallback.
# ---------------------------------------------------------------------------
def factor_sarma_forecast(X,
    pipeline_pack: Dict[str, Any],
    *,
    steps: int = 1,
) -> np.ndarray:
    """
    Forecast next `steps` observations of Y using a fitted DFM-SARMA pack.

    Parameters
    ----------
    pipeline_pack : dict
        The return dict from `dynamic_factor_sarma_pipeline` or from
        `pca_sarma_pipeline(..., return_full=True)`, containing:
          - 'pca'      : PCA object
          - 'y_proc'   : standardized Y (T, N)
          - 'y_lowdim' : factors F (T, k)
          - 'mean','std': arrays for inverse transform
          - 'est'      : fitted SARMAEstimator on factors
    steps : int
        Forecast horizon.
    ar_only_fallback : bool
        If SARMAEstimator lacks `.predict` (NotImplemented), use an AR-only
        one-step scheme based on the selected order p.

    Returns
    -------
    np.ndarray
        Forecasts with shape (steps, N).
    """
    # pca = pipeline_pack["pca"]
    Lambda = pipeline_pack["Lambda"]
    F = pipeline_pack["y_lowdim"]
    est = pipeline_pack["est"]

    # Determine dimensions
    k = int(F.shape[1]) if (F is not None and F.ndim == 2) else 0
    M = 0 if (X is None) else (0 if getattr(X, 'size', 0) == 0 else np.asarray(X).shape[1])

    # Infer how the estimator was trained by checking the fitted Sigma size
    trained_n = None
    try:
        trained_params = est.get_params()
        Sigma = trained_params.get("Sigma", None)
        if Sigma is not None:
            trained_n = int(np.asarray(Sigma).shape[0])
    except Exception:
        trained_n = None

    # Case A: estimator trained on factors only (trained_n == k)
    if trained_n is not None and k > 0 and trained_n == k:
        # Prepare recent factor history and predict factors
        F_recent = F
        F_fore = est.predict(F_recent, steps=steps)
        # Inverse PCA transform: factors -> original variable space
        Y_fore = np.asarray(F_fore) @ np.asarray(Lambda).T

    # Case B: estimator trained on combined [X, F]
    elif trained_n is not None and k >= 0 and trained_n == (M + k):
        # Prepare recent combined history and let estimator predict full combined Y
        Y_combined = np.hstack([X, F]) if (X is not None and getattr(X, 'size', 0) > 0) else F
        Y_fore = est.predict(Y_combined, steps=steps)

    else:
        # Fallback: try to call estimator on combined data; if result has k columns,
        # treat it as factor forecasts and inverse-transform; otherwise return as-is.
        try:
            Y_combined = np.hstack([X, F]) if (X is not None and getattr(X, 'size', 0) > 0) else F
            Yc_fore = est.predict(Y_combined, steps=steps)
            Yc_fore = np.asarray(Yc_fore)
            if Yc_fore.ndim == 2 and Yc_fore.shape[1] == k:
                # got factor forecasts
                Y_fore = Yc_fore @ np.asarray(Lambda).T
            else:
                Y_fore = Yc_fore
        except Exception:
            # Last-resort: predict factors via estimator if it accepts factor history,
            # otherwise raise the original error
            F_recent = F
            F_fore = est.predict(F_recent, steps=steps)
            Y_fore = np.asarray(F_fore) @ np.asarray(Lambda).T
    # try:
    #     F_fore = est.predict(F[-est.get_params()["p"]:, :], steps=steps)
    # except Exception:
    #     if not ar_only_fallback:
    #         raise
    #     # AR-only fallback: use last p factor lags with coefficients implicitly
    #     # learned inside estimator. We approximate with a simple VAR(p) in
    #     # companion form using least squares on F (lightweight).
    #     F_fore = _varp_forecast_fallback(F, p=est.get_params()["p"], steps=steps)

    return np.asarray(Y_fore)




# # ---------------------------------------------------------------------------
# # Minimal CLI smoke test
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":  # pragma: no cover
#     rng = np.random.default_rng(0)
#     T, N = 500, 6
#     y = rng.standard_normal((T, N))

#     # quick run: PCA→SARMA for fixed k (data assumed standardized if desired)
#     pack_fixed = dynamic_factor_sarma_pipeline(
#         y,
#         k=3,
#         estimator_kwargs=dict(
#             method="ls",
#             run_bic=True,
#             bic_branch="ls",
#             refine_top_k=3,
#             refine_parallel=True,
#             refine_n_jobs=2,
#             use_multistart=True,
#             multistart_parallel=False,
#             P=150,
#             n_iter=200,
#             stop_thres=1e-4,
#             verbose=False,
#         ),
#         verbose=True,
#     )
#     print("[dynamic_factor_sarma_pipeline:fixed] ", pack_fixed["est"].summary())

#     # dynamic factor → SARMA using Eigenvalue Ratio for k selection
#     pack_er = dynamic_factor_sarma_pipeline(
#         y,
#         k=None,
#         k_max=4,
#         k_select="eigen_ratio",
#         estimator_kwargs=dict(
#             method="ls",
#             run_bic=True,
#             bic_branch="ls",
#             refine_top_k=3,
#             refine_parallel=True,
#             refine_n_jobs=2,
#             use_multistart=True,
#             multistart_parallel=False,
#             P=150,
#             n_iter=200,
#             stop_thres=1e-4,
#             verbose=False,
#         ),
#         verbose=True,
#     )
#     print(f"[dynamic_factor_sarma_pipeline:ER] chosen k={pack_er['k']}, est={pack_er['est'].summary()}")


# ---------------------------------------------------------------------------
# DynamicFactorSARMA: Wrapper class for clean step-by-step workflow
# ---------------------------------------------------------------------------
class DynamicFactorSARMA:
    """
    Wrapper class for Dynamic Factor Model + SARMA forecasting.

    Provides a clean step-by-step interface:
      1. Create instance with data
      2. (Optional) Select eigenvalue ratio or specify k
      3. Fit SARMA model on factors
      4. Forecast

    Example:
        dfm = DynamicFactorSARMA(y, X=exog_data)
        dfm.select_factors(k=None, k_max=8)  # auto-select or specify k
        dfm.fit_sarma(estimator_kwargs=dict(method='mle'))
        forecast = dfm.forecast(steps=10)
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        X: Union[np.ndarray, pd.DataFrame, None] = None,
        verbose: bool = False,
    ):
        """
        Initialize DFM-SARMA wrapper.

        Parameters
        ----------
        y : array-like (T, N)
            Time-series data (T time steps, N variables). Should be standardized if desired.
        X : array-like (T, M) or None
            Exogenous features (optional). If provided, will be stacked with factors.
        verbose : bool
            Whether to print progress information.
        """
        self.y_raw = np.asarray(y)
        if self.y_raw.ndim != 2:
            raise ValueError(f"y must be 2D (T, N), got shape {self.y_raw.shape}")

        self.X = None if X is None else np.asarray(X)
        if self.X is not None and self.X.shape[0] != self.y_raw.shape[0]:
            raise ValueError(f"X and y must have same number of time steps (T), got {self.X.shape[0]} vs {self.y_raw.shape[0]}")

        self.verbose = verbose
        self.T, self.N = self.y_raw.shape

        # State variables
        self.k: Optional[int] = None
        self.pca_svd: Optional[Dict[str, np.ndarray]] = None  # U, S, Vt
        self.F_hat: Optional[np.ndarray] = None  # factors (T, k)
        self.Lambda: Optional[np.ndarray] = None  # loadings (N, k)
        self.est: Optional[SARMAEstimator] = None
        self.auto_search: Optional[Dict[str, Any]] = None

    def select_factors(
        self,
        k: Optional[int] = None,
        k_max: int = 8,
        k_select: str = "eigen_ratio",
    ) -> int:
        """
        Choose the number of factors (either automatically or by specifying k).

        Parameters
        ----------
        k : int or None
            If provided, use this as the factor dimension.
            If None, auto-select using k_select method.
        k_max : int
            Upper bound for k-search when k is None.
        k_select : str
            Method for auto-selecting k: only "eigen_ratio" supported.

        Returns
        -------
        int
            The chosen factor dimension k.
        """
        if k is not None:
            if not isinstance(k, int) or k < 1 or k > self.N:
                raise ValueError(f"k must be int in [1, {self.N}], got {k}")
            self.k = k
            if self.verbose:
                print(f"[select_factors] Using provided k={k}")
        else:
            if k_select.lower() != "eigen_ratio":
                raise ValueError(f"Only k_select='eigen_ratio' supported, got {k_select}")
            k_hat, eigvals, ratios = eigen_ratio_select(self.y_raw, k_max=min(k_max, self.N - 1))
            self.k = int(k_hat)
            self.auto_search = {"method": k_select, "eigvals": eigvals, "ratios": ratios}
            if self.verbose:
                peak = ratios[k_hat - 1] if ratios.size >= k_hat else np.nan
                print(f"[select_factors] Auto-selected k={k_hat} (peak ratio={peak:.3g})")

        return self.k

    def pca_decompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform PCA/SVD decomposition to extract factors and loadings.

        Must call `select_factors()` first. Returns:
          - F_hat: factor series (T, k)
          - Lambda: loadings (N, k)

        Returns
        -------
        F_hat : (T, k) ndarray
            Factor time series
        Lambda : (N, k) ndarray
            Loading matrix
        """
        if self.k is None:
            raise RuntimeError("Must call select_factors() before pca_decompose()")

        U, S, Vt = np.linalg.svd(self.y_raw, full_matrices=False)
        self.F_hat = np.sqrt(self.T) * U[:, : self.k]
        self.Lambda = (1 / np.sqrt(self.T)) * (Vt[: self.k, :].T * S[: self.k])
        self.pca_svd = {"U": U, "S": S, "Vt": Vt}

        if self.verbose:
            print(f"[pca_decompose] Extracted factors F_hat shape={self.F_hat.shape}, Lambda shape={self.Lambda.shape}")

        return self.F_hat, self.Lambda

    def fit_sarma(
        self,
        p: Optional[int] = None,
        r: Optional[int] = None,
        s: Optional[int] = None,
        est_kwargs: Optional[Dict[str, Any]] = None,
        bic_kwargs: Optional[Dict[str, Any]] = None,
        init_params: Optional[Dict[str, np.ndarray]] = None,
    ) -> "DynamicFactorSARMA":
        """
        Fit SARMA model on the factors (optionally stacked with exogenous X).

        Must call `pca_decompose()` first.

        Parameters
        ----------
        p, r, s : int or None
            SARMA orders. If any is None and run_bic=True in est_kwargs or bic_kwargs,
            BIC will be used to select missing orders.
        est_kwargs : dict or None
            Passed to SARMAEstimator constructor (e.g., method='mle', use_multistart, P, n_iter, etc.).
            Use `make_estimator_kwargs(...)` to generate with validation.
        bic_kwargs : dict or None
            BIC-related settings (e.g., run_bic, bic_branch, refine_top_k, bic_caps, etc.).
            Use `make_bic_kwargs(...)` to generate with validation.
            The 'bic_caps' key (if present) will be extracted and passed to fit().
        init_params : dict or None
            Initial parameters for SARMA fitting.

        Returns
        -------
        self
        """
        if self.F_hat is None:
            raise RuntimeError("Must call pca_decompose() before fit_sarma()")

        if est_kwargs is None:
            est_kwargs = {}
        if bic_kwargs is None:
            bic_kwargs = {}

        # Extract bic_caps from bic_kwargs (default to (3,3,2) if not provided)
        bic_caps = bic_kwargs.pop("bic_caps", (3, 3, 2))

        # Merge BIC-related kwargs into estimator kwargs for SARMAEstimator constructor
        # (run_bic, bic_branch, refine_top_k, refine_n_random, etc.)
        merged_est_kwargs = {**est_kwargs}
        bic_keys = {"run_bic", "bic_branch", "refine_top_k", "refine_n_random", 
                   "refine_n_iter", "refine_grid_mode", "refine_stop_thres", 
                   "refine_parallel", "refine_n_jobs"}
        for key in bic_keys:
            if key in bic_kwargs:
                merged_est_kwargs[key] = bic_kwargs[key]

        # Stack X and F if X is provided
        if self.X is not None:
            Y_fit = np.hstack([self.X, self.F_hat])
        else:
            Y_fit = self.F_hat

        self.est = SARMAEstimator(**merged_est_kwargs)
        self.est.fit(Y_fit, p=p, r=r, s=s, bic_caps=bic_caps, init_params=init_params)

        if self.verbose:
            print(f"[fit_sarma] Fitted model: {self.est.summary()}")

        return self

    def forecast(self, steps: int = 1, y_recent: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forecast next `steps` observations.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        y_recent : (T_recent, N_combined) ndarray or None
            Recent history for forecasting. If None, uses last 50 steps of combined data.
            Should include both X (if applicable) and factors.

        Returns
        -------
        np.ndarray
            Forecasts with shape (steps, N_combined).
        """
        if self.est is None:
            raise RuntimeError("Must call fit_sarma() before forecast()")

        if y_recent is None:
            if self.X is not None:
                Y_combined = np.hstack([self.X, self.F_hat])
            else:
                Y_combined = self.F_hat
            y_recent = Y_combined

        y_fore= self.est.predict(y_recent, steps=steps)

        if self.verbose:
            print(f"[forecast] Generated {steps}-step forecast, shape={y_fore.shape}")
        return np.asarray(y_fore)
    def get_factors(self) -> Optional[np.ndarray]:
        """Return extracted factors F_hat (T, k) or None if not yet decomposed."""
        return self.F_hat

    def get_loadings(self) -> Optional[np.ndarray]:
        """Return factor loadings Lambda (N, k) or None if not yet decomposed."""
        return self.Lambda

    def get_estimator(self) -> Optional[SARMAEstimator]:
        """Return fitted SARMAEstimator or None if not yet fitted."""
        return self.est

    def summary(self) -> str:
        """Return a summary of the DFM-SARMA workflow status."""
        lines = ["DynamicFactorSARMA Summary:"]
        lines.append(f"  Data shape: (T={self.T}, N={self.N})")
        lines.append(f"  Selected k: {self.k}")
        if self.F_hat is not None:
            lines.append(f"  Factors shape: {self.F_hat.shape}")
            lines.append(f"  Loadings shape: {self.Lambda.shape}")
        if self.est is not None:
            lines.append(f"  Fitted model: {self.est.summary()}")
        else:
            lines.append(f"  Fitted model: None (not fitted yet)")
        return "\n".join(lines)