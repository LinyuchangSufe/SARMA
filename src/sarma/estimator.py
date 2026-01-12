from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
# ---------------------------------------------------------------------------
# Imports with robust fallbacks
# ---------------------------------------------------------------------------
try:
    # preferred package-local imports
    from .optim import BCD_SARMA, multi_start_BCD
except Exception:  # pragma: no cover
    from src.sarma.optim import BCD_SARMA, multi_start_BCD  # type: ignore


try:
    from .selection import BIC_parallel_joblib
except Exception:  # pragma: no cover
    from src.sarma.selection import BIC_parallel_joblib  # type: ignore

# Try to import get_A from utils (builds the stacked lag operator A)
try:
    from .utils.help_function import get_A  # type: ignore
    from .utils.tensorOp import tensor_op
except Exception:  # pragma: no cover

    from src.utils.help_function import get_A, get_y_pre  # type: ignore
    from src.utils.tensorOp import tensor_op



# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SARMAFitResult:
    """Container for the fitted SARMA model parameters and diagnostics."""
    loss: float
    p: int
    r: int
    s: int
    lmbd: np.ndarray
    eta: np.ndarray
    G: np.ndarray
    Sigma: np.ndarray
    AsyVar: np.ndarray
    A: np.ndarray 

# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
class SARMAEstimator:
    """
    High-level SARMA interface:
      - (optional) Order selection via BIC when (p,r,s) not supplied
      - (optional) Top-k refinement via multi-start (BIC_refine)
      - (optional) Multi-start to choose robust initials
      - Final fit via block-coordinate descent (BCD_SARMA)

    Notes
    -----
    * PCA / other preprocessing should live in a pipeline module,
      not inside this estimator.
    """

    def __init__(
        self,
        *,
        # Core solver controls
        # Loss / likelihood branch
        P: int = 200,
        n_iter: int = 100,
        stop_thres: float = 1e-6,

        # Multi-start controls (for initials, after orders are fixed)
        grid_mode: str = "auto",       # 'full'|'random'|'auto'
        n_random: int = 2000,
        seed: Optional[int] = None,

        verbose: bool = False,
    ) -> None:
        # Solver configs
        self.P = P
        self.n_iter = n_iter
        self.stop_thres = stop_thres

        # Multi-start
        self.grid_mode = grid_mode
        self.n_random = n_random
        self.seed = seed


        self.verbose = verbose

        self._fitted: Optional[SARMAFitResult] = None
        self._history: Dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _choose_orders_via_bic(
        self,
        y: np.ndarray,
        n_jobs_BIC: int = 1
    ):
        """
        Run BIC and return selected (p,r,s), initial params (lmbd, eta), and full bic_result.
        
        Returns
        -------
        (p, r, s, lmbd_init, eta_init, bic_result)
        """

        bic_res = BIC_parallel_joblib(
            y,
            P=self.P, seed=self.seed,
            verbose=self.verbose, n_jobs_BIC=n_jobs_BIC
        )
        p, r, s = bic_res['ML_min_index']
        lmbd_init = bic_res.get('ML_lmbd_value', None)
        eta_init = bic_res.get('ML_eta_value', None)

        return p, r, s, lmbd_init, eta_init, bic_res

    # ---------------------------------------------------------------------
    # Core training SARMA with initialization
    # ---------------------------------------------------------------------
    def SARMA_fitBCD_SARMA(self,
                    y: np.ndarray,
                    p: int,
                    r: int,
                    s: int,
                    *,
                    lmbd: Optional[np.ndarray] = None,          # shape (r,)
                    eta: Optional[np.ndarray] = None,           # shape (s, 2) with columns (gamma, phi)
                    Sigma: Optional[np.ndarray] = None,         # N x N covariance; if None, LS uses I
                    esti_method: str = "ls",                    # 'ls' or 'mle' (GLS via MLE Sigma)
                    P: int = 150,                               # truncation for seasonal design matrices
                    n_iter: int = 100,
                    stop_thres: float = 1e-6,
                    verbose: bool = False,
                    Cal_AsyVar: bool = True,
                ):
        result = BCD_SARMA(y,p,r,s,lmbd = lmbd, eta = eta, Sigma = Sigma, 
                           esti_method = esti_method, 
                           P = P, n_iter = n_iter, stop_thres = stop_thres, 
                           verbose = verbose, Cal_AsyVar = Cal_AsyVar)
        self._history['BCD_SAMRA'] = result
        return result
    # ---------------------------------------------------------------------
    # Core training routine
    # ---------------------------------------------------------------------
    def fit(
        self,
        y: np.ndarray,
        p: Optional[int] = None,
        r: Optional[int] = None,
        s: Optional[int] = None,
        n_jobs_BIC: int = 1,
        n_jobs_profiling: int = 1
    ) -> "SARMAEstimator":
        """
        Fit SARMA on a panel/time-series matrix.

        Parameters
        ----------
        y : array (T, N)
            Time along axis 0, variables along axis 1.
        p, r, s : int or None
            Model orders. If any is None and run_bic=True, BIC will be used to select (p, r, s).
        bic_caps : tuple (p_m, r_m, s_m)
            Upper bounds for the BIC search when orders are missing.

        Returns
        -------
        self
        """
        assert y.ndim == 2, "y must have shape (T, N)"
        T, N = y.shape

        # 1) If orders not supplied (or run_bic requested), select via BIC on the original data
        if (p is None) or (r is None) or (s is None):
            p, r, s, lmbd_init, eta_init, bic_res = self._choose_orders_via_bic(y, n_jobs_BIC=n_jobs_BIC)
            self._history["bic"] = bic_res
            if self.verbose:
                print(f"[Estimator] Selected orders (p,r,s)=({p},{r},{s}) via BIC.")
       
        assert p is not None and r is not None and s is not None
        self.p = p
        self.r = r
        self.s = s

        # 2) Multi-start BCD to find robust initials and refine
        #    Returns (res_ls, res_mle) where each is a dict
        if self.verbose:
            print(f"[Estimator] Running multi-start BCD for (p={p}, r={r}, s={s})...")
        
        res_ls, res_mle = multi_start_BCD(
            y=y,
            p=p, r=r, s=s,
            step=0.1,
            Cal_AsyVar=True,
            P=self.P,
            grid_mode=self.grid_mode,
            n_random=self.n_random,
            n_iter=self.n_iter,
            stop_thres=self.stop_thres,
            seed=self.seed,
            n_jobs_profiling=n_jobs_profiling,
            verbose=self.verbose,
        )
        self._history['LS_result'] = res_ls
        self._history['MLE_result'] = res_mle

        # 3) Store the final MLE result
        if self.verbose:
            print(f"[Estimator] Final MLE loss={res_mle['Loss']:.6e}, "
                  f"converged={res_mle.get('converged', False)}, "
                  f"iter_no={res_mle.get('iter_no', '?')}")

        self._history["multistart"] = {
            "loss_ls": float(res_ls["Loss"]),
            "loss_mle": float(res_mle["Loss"]),
            "iter_no_ls": int(res_ls.get("iter_no", -1)),
            "iter_no_mle": int(res_mle.get("iter_no", -1)),
            "converged_ls": bool(res_ls.get("converged", False)),  
            "converged_mle": bool(res_mle.get("converged", False)),
        }
    
        # 4) Create fitted result from MLE branch (final result)
        res = res_mle
        self._fitted = SARMAFitResult(
            loss=float(res["Loss"]),
            p=int(p), r=int(r), s=int(s),
            lmbd=np.asarray(res["lmbd"]),
            eta=np.asarray(res["eta"]),
            G=np.asarray(res["G"]),
            Sigma=np.asarray(res["Sigma"]),
            AsyVar=np.asarray(res["AsyVar"]),
            A=np.asarray(res["A"]),
        )
        return self

    # ---------------------------------------------------------------------
    # Accessors / Utilities
    def get_params(self) -> Dict[str, Any]:
        """Return fitted parameters and training config."""
        if self._fitted is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        d = asdict(self._fitted)
        d.update({
            "P": self.P,
            "n_iter": self.n_iter,
            "stop_thres": self.stop_thres,
            "grid_mode": self.grid_mode,
            "n_random": self.n_random,
            "seed": self.seed,
        })
        return d

    def summary(self) -> str:
        """A compact, human-readable summary of the fitted model."""
        if self._fitted is None:
            return "SARMAEstimator(not fitted)"
        f = self._fitted
        return (f"SARMA(p={f.p}, r={f.r}, s={f.s}), "
                f"loss={f.loss:.6g}, P={self.P}, n_iter={self.n_iter}, "
                f"stop_thres={self.stop_thres}")
    
        
    def predict(self, y_last: np.ndarray, steps: int = 1, **kwargs) -> np.ndarray:
        """Forecast the next `steps` observations using fitted SARMA parameters.

        This implements a NumPy-only version using the A tensor (VAR representation),
        and returns forecasts with shape (steps, N).

        Parameters
        ----------
        y_last : array-like
            Recent history of Y with shape (T, N) or (N, T). Time must be along
            one axis; the method will infer and convert as needed. The last row
            (if (T,N)) or last column (if (N,T)) should be the most recent time.
        steps : int, default 1
            Forecast horizon.
        **kwargs :
            Optional controls (reserved for future use).

        Returns
        -------
        np.ndarray
            Array of shape (steps, N) with the forecasts.
        """
        if self._fitted is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        # Pull fitted params
        f = self._fitted
        p, r, s = int(f.p), int(f.r), int(f.s)
        N = f.Sigma.shape[0]
        G = np.asarray(f.G)
        lmbd = np.asarray(f.lmbd) if f.lmbd is not None else np.array([])
        eta = np.asarray(f.eta) if f.eta is not None else np.zeros((0, 2))

        if get_A is None:
            raise ImportError(
                "get_A could not be imported from utils; forecasting requires it."
            )

        if tensor_op is None:
            raise ImportError(
                "tensor_op could not be imported from utils; forecasting requires it."
            )

        y_last = np.asarray(y_last)
      
        # Build A tensor (N x N x L) and unfold to (N, N*L)
        A_tensor = get_A(lmbd, eta, G, p, r, s, self.P)  # expected shape (N, N, L_total)
        A_mat = np.asarray(tensor_op.unfold(A_tensor, 0)) # shape (N, N*L)
    
        # Rollout forecasts
        y_pre = np.zeros((int(steps), N), dtype=float)
        # for k in range(int(steps)):
            # Build lagged observation vector
            # Most recent future predictions first, then past observations
            # if k > 0:
            #     left = np.flip(y_pre[:k], axis=0).ravel(order="C")
            # else:
            #     left = np.empty((0,), dtype=float)
            
        rem = A_tensor.shape[2]
            # if rem > 0 and y_last.shape[0] >= rem:
        x = np.flip(y_last[-rem:], axis=0).ravel(order="C")
        
        
        # x = np.concatenate([left, right], axis=0)
        y_pre[0] = A_mat @ x
        
        return y_pre
# import pandas as pd
# df = pd.read_csv("data/FRED-MD.csv", header=0, index_col=0)

# vars_ = ['RPI','INDPRO','UNRATE','M2SL','CPIAUCSL','DPCERA3M086SBEA']
# df = df[vars_]
# df = df.drop(df.index[0])
# # ------------------------------------------------
# # 2. FRED-MD 官方 transform code
# # ------------------------------------------------
# df.head()

# transform_codes = {
#     'RPI': 5,
#     'INDPRO': 5,
#     'UNRATE': 2,
#     'M2SL': 6,
#     'CPIAUCSL': 6,
#     'DPCERA3M086SBEA': 5
# }

# def fred_md_transform(x: pd.Series, code: int) -> pd.Series:
#     if code == 1:          # level
#         return x
#     elif code == 2:        # first difference
#         return x.diff()
#     elif code == 3:        # second difference
#         return x.diff().diff()
#     elif code == 4:        # log level
#         return np.log(x)
#     elif code == 5:        # Δ log
#         return np.log(x).diff()
#     elif code == 6:        # Δ² log
#         return np.log(x).diff().diff()
#     else:
#         raise ValueError(f"Unknown transform code: {code}")

# # ------------------------------------------------
# # 3. 平稳化变换
# # ------------------------------------------------
# df_trans = pd.DataFrame(index=df.index)

# for v in vars_:
#     df_trans[v] = fred_md_transform(df[v], transform_codes[v])

# # ------------------------------------------------
# # 4. 去除 NaN（由差分产生）
# # ------------------------------------------------
# df_trans = df_trans.dropna(how="any")

# # ------------------------------------------------
# # 5. 标准化（去均值 / 除以标准差）
# # ------------------------------------------------
# df_std = (df_trans - df_trans.mean()) / df_trans.std(ddof=0)

# # df_std 即为最终可用于建模的数据
# y = df_std.values
# T,N = df_std.shape
# est = SARMAEstimator(
#     P=200,
#     n_iter=100,
#     stop_thres=1e-5,
#     verbose=True,
# )
# est.SARMA_fitBCD_SARMA(y,0,1,0)