from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
# Progress bar
from tqdm import tqdm

# Also import the multi-start initializer for optional refinement passes
try:
    from .optim import multi_start_BCD
except Exception:  # pragma: no cover
    from src.sarma.optim import multi_start_BCD  # type: ignore

def _bic_one_p(
    p: int,
    y: np.ndarray,
    r_m: int,
    s_m: int,
    P: int,
    seed: Optional[int],
    verbose: bool,
) -> pd.DataFrame:
    """Compute BIC rows for a fixed p over all feasible (r, s)."""
    T, N = y.shape
    T_eff = T - p

    idx = [(p, r, s) for r in range(r_m + 1) for s in range(s_m + 1) if r + 2 * s <= N]
    index = pd.MultiIndex.from_tuples(idx, names=["p", "r", "s"])
    df = pd.DataFrame(index=index, columns=["BIC", "Loss", "lmbd", "eta"])

    for r in range(r_m + 1):
        for s in range(s_m + 1):
            if r + 2 * s > N:
                continue

            _, result_MLE = multi_start_BCD(
                y, p, r, s,
                step=0.05, P=P,
                grid_mode="full",
                n_random=2000,
                n_iter=100,
                stop_thres=1e-6,
                seed=seed,
                verbose=verbose,
                Cal_AsyVar=False,
            )

            loss = float(result_MLE["Loss"])
            df.loc[(p, r, s), "Loss"] = loss

            if (r == 0) and (s == 0):
                df.loc[(p, r, s), ["lmbd", "eta"]] = [np.nan, np.nan]
            else:
                df.loc[(p, r, s), ["lmbd", "eta"]] = [
                    result_MLE["lmbd"], result_MLE["eta"]
                ]

            k = r + 2 * s + (N ** 2) * (p + r + 2 * s) + N * (N + 1) // 2
            df.loc[(p, r, s), "BIC"] = T_eff * loss + k * np.log(T_eff)

    return df
# BIC_parallel_joblib(y,n_jobs_BIC = 3)
def BIC_parallel_joblib(
    y: np.ndarray,
    p_m: int = 2,
    r_m: Optional[int] = 2,
    s_m: Optional[int] = 2,
    *,
    P: int = 200,
    seed: Optional[int] = 42,
    verbose: bool = False,
    n_jobs_BIC: int = 3,   # -1 = 用所有 CPU
) -> Dict[str, Any]:
    """
    Parallel BIC for SARMA over (p,r,s), parallelized by p using joblib.
    """
    y = np.asarray(y)
    T, N = y.shape
    P = min(P, T)

    r_m = 2 if r_m is None else r_m
    s_m = 2 if s_m is None else s_m

    # 并行：每个 p 一个任务。若 n_jobs_BIC==1，则顺序运行并显示进度条；
    # 否则使用 joblib 并行化（按 p 划分任务）。
    if n_jobs_BIC == 1:
        dfs = []
        for p in tqdm(range(p_m + 1), desc="BIC scan (sequential, by p)", ncols=90):
            dfs.append(_bic_one_p(p, y, r_m, s_m, P, seed, verbose))
    else:
        dfs = Parallel(n_jobs=n_jobs_BIC)(
            delayed(_bic_one_p)(p, y, r_m, s_m, P, seed, verbose)
            for p in tqdm(range(p_m + 1), desc="BIC scan (joblib, by p)", ncols=90)
        )

    # 合并所有 p 的结果
    ML_BIC_table = pd.concat(dfs).sort_index()

    # 选最小 BIC（不要转 int）
    ML_min_index = ML_BIC_table["BIC"].astype(float).idxmin()
    ML_min_index = tuple(int(x) for x in ML_min_index)

    ML_lmbd_value, ML_eta_value = ML_BIC_table.loc[ML_min_index, ["lmbd", "eta"]]

    return {
        "ML_BIC_table": ML_BIC_table,
        "ML_min_index": ML_min_index,
        "ML_lmbd_value": ML_lmbd_value,
        "ML_eta_value": ML_eta_value,
    }


# def BIC(
#     y: np.ndarray,
#     p_m: int = 3,
#     r_m: Optional[int] = 3,
#     s_m: Optional[int] = 2,
#     *,
#     P = 150,
#     branch: str = "mle",
#     seed : Optional[int] = 42,
#     verbose: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Compute BIC tables for SARMA over a grid of (p, r, s).
#     You can choose which branch to compute via `branch`:
#       - 'ls'   : only LS branch
#       - 'mle'  : only MLE branch
#       - 'both' : compute both (default; higher compute)
#     """
#     y = np.asarray(y)
#     # y_TN, transposed = _to_TN(np.asarray(y))
#     T, N = y.shape
#     P = min(P, T)  # safe lower bound inside log

#     branch = branch.lower()
#     if branch not in {"ls", "mle", "both"}:
#         raise ValueError("branch must be one of {'ls','mle','both'}")

#     if r_m is None:
#         r_m = 2
#     if s_m is None:
#         s_m = 2

#     # Build index only for feasible combinations to avoid NaNs
#     idx = [(pp, rr, ss)
#            for pp in range(p_m + 1)
#            for rr in range(r_m + 1)
#            for ss in range(s_m + 1)
#            if rr + 2 * ss <= N]
#     index = pd.MultiIndex.from_tuples(idx, names=["p", "r", "s"])
#     LS_BIC_table = pd.DataFrame(columns=["BIC", "initial"], index=index) if branch in {"ls", "both"} else None
#     ML_BIC_table = pd.DataFrame(columns=["BIC", "initial"], index=index) if branch in {"mle", "both"} else None

#     # Predefine small candidate grids for initials (only used when r or s > 0)

#     for p in tqdm(range(p_m + 1), desc="BIC: scanning p values", ncols=90):
#         for r in range(r_m + 1):
#             for s in range(s_m + 1):
#                 if r + 2 * s > N:
#                     continue
#                 result_LS, result_MLE = multi_start_BCD(y,
#                                             p, r, s,
#                                             step = 0.1, P = P,
#                                             grid_mode = "full",
#                                             n_random = 2000,
#                                             n_iter = 500,
#                                             stop_thres = 1e-6,
#                                             seed = seed,
#                                             verbose = verbose,
#                                         )
#                 # Unified candidate enumeration for all (p,r,s):
#                 # Build a small grid of candidate initials; when r=s=0 this becomes a single empty tuple.
#                 # if (r == 0) and (s == 0):
#                 #     alpha_iterable = [tuple()]
#                 # else:
#                 #     # Build candidates similar to multi_start_BCD but with smaller pools here.
#                 #     # We deduplicate by sorting lambda/gamma in descending order; phi can repeat.
#                 #     seen = set()
#                 #     alpha_iterable = []
#                 #     if (r <= 2) and (s <= 2):
#                 #         lam_iter = permutations(lmbd_grid, r) if r > 0 else [()]
#                 #         gam_iter = permutations(gamma_grid, s) if s > 0 else [()]
#                 #         th_iter  = product(phi_grid, repeat=s) if s > 0 else [()]
#                 #         for lam in lam_iter:
#                 #             lam_s = tuple(sorted(lam, reverse=True))
#                 #             for gam in gam_iter:
#                 #                 gam_s = tuple(sorted(gam, reverse=True))
#                 #                 for th in th_iter:
#                 #                     key = (lam_s, gam_s, th)
#                 #                     if key in seen:
#                 #                         continue
#                 #                     seen.add(key)
#                 #                     # concatenate to a single tuple: (lmbd..., gamma..., phi...)
#                 #                     alpha_iterable.append(tuple(lam_s + gam_s + th))
#                 #     else:
#                 #         lam_iter = product(lmbd_grid, repeat=r) if r > 0 else [()]
#                 #         gam_iter = product(gamma_grid, repeat=s) if s > 0 else [()]
#                 #         th_iter  = product(phi_grid, repeat=s) if s > 0 else [()]
#                 #         for lam in lam_iter:
#                 #             lam_s = tuple(sorted(lam, reverse=True))
#                 #             for gam in gam_iter:
#                 #                 gam_s = tuple(sorted(gam, reverse=True))
#                 #                 for th in th_iter:
#                 #                     key = (lam_s, gam_s, th)
#                 #                     if key in seen:
#                 #                         continue
#                 #                     seen.add(key)
#                 #                     alpha_iterable.append(tuple(lam_s + gam_s + th))

#                 # # Track best per-branch without building large Series
#                 # LS_log_loss = None
#                 # ML_log_loss = None
#                 # LS_best_alpha = None
#                 # ML_best_alpha = None

#                 # for alpha in alpha_iterable:
#                 #     # Decode initials
#                 #     if (r == 0) and (s == 0):
#                 #         lmbd = []
#                 #         gamma = []
#                 #         phi = []
#                 #     else:
#                 #         lmbd = list(alpha[:r])
#                 #         gamma = list(alpha[r:r + s])
#                 #         phi = list(alpha[r + s:r + 2 * s])

#                 #         # Optional uniqueness filter to reduce symmetric duplicates
#                 #         if (len(lmbd) != len(set(lmbd))) or (len(gamma) != len(set(gamma))) or (len(phi) != len(set(phi))):
#                 #             continue

#                 #     eta = np.column_stack([np.asarray(gamma, float), np.asarray(phi, float)]) if s > 0 else None

#                 #     # Short LS run
#                 #     if branch in {"ls", "both"}:
#                 #         res_ls = BCD_SARMA(
#                 #             y=y_TN, p=p, r=r, s=s,
#                 #             lmbd=np.asarray(lmbd, float) if r > 0 else None,
#                 #             eta=eta, Sigma=None, esti_method="ls",
#                 #             P=P, n_iter=20, stop_thres=1e-3, result_show=False,
#                 #         )
#                 #         Sigma_ls = np.asarray(res_ls["Sigma"])
#                 #         sign_ls, logdet_ls = np.linalg.slogdet(Sigma_ls + 1e-12 * np.eye(N))
#                 #         if sign_ls <= 0:
#                 #             logdet_ls = np.log(np.linalg.det(Sigma_ls + 1e-12 * np.eye(N)) + 1e-5)
#                 #         if (LS_log_loss is None) or (logdet_ls < LS_log_loss):
#                 #             LS_log_loss = float(logdet_ls)
#                 #             LS_best_alpha = alpha

#                     # Short MLE run
#                     if branch in {"mle", "both"}:
#                         res_ml = BCD_SARMA(
#                             y=y_TN, p=p, r=r, s=s,
#                             lmbd=np.asarray(lmbd, float) if r > 0 else None,
#                             eta=eta, Sigma=None, esti_method="mle",
#                             P=P, n_iter=20, stop_thres=1e-3, result_show=False,
#                         )
#                         Sigma_ml = np.asarray(res_ml["Sigma"])
#                         sign_ml, logdet_ml = np.linalg.slogdet(Sigma_ml + 1e-12 * np.eye(N))
#                         if sign_ml <= 0:
#                             logdet_ml = np.log(np.linalg.det(Sigma_ml + 1e-12 * np.eye(N)) + 1e-30)
#                         if (ML_log_loss is None) or (logdet_ml < ML_log_loss):
#                             ML_log_loss = float(logdet_ml)
#                             ML_best_alpha = alpha

#                 # Record best initials in tables (flattened as (lmbd..., gamma..., phi...))
#                 # if branch in {"ls", "both"} and (LS_log_loss is not None):
#                 #     if (r == 0) and (s == 0):
#                 #         LS_BIC_table.loc[(p, r, s), "initial"] = tuple()  # type: ignore[index]
#                 #     else:
#                 #         a = LS_best_alpha
#                 #         LS_BIC_table.loc[(p, r, s), "initial"] = tuple(list(a[:r]) +
#                 #                                                        list(a[r:r + s]) +
#                 #                                                        list(a[r + s:r + 2 * s]))  # type: ignore[index]

#                 if branch in {"mle", "both"} and (ML_log_loss is not None):
#                     if (r == 0) and (s == 0):
#                         ML_BIC_table.loc[(p, r, s), "initial"] = tuple()  # type: ignore[index]
#                     else:
#                         a = ML_best_alpha
#                         ML_BIC_table.loc[(p, r, s), "initial"] = tuple(list(a[:r]) +
#                                                                        list(a[r:r + s]) +
#                                                                        list(a[r + s:r + 2 * s]))  # type: ignore[index]

#                 if verbose:
#                     print(f"[BIC] Completed (p={p}, r={r}, s={s})")
#                 d_m = r + 2 * s + (N ** 2) * (p + r + 2 * s) + N*(N + 1) // 2  # number of free parameters
#                 ML_Sigma = result_MLE['Sigma']
#                 sign_ml, logdet_ml = np.linalg.slogdet(ML_Sigma + 1e-12 * np.eye(N))
#                 if sign_ml <= 0:
#                     logdet_ml = np.log(np.linalg.det(Sigma_ml + 1e-12 * np.eye(N)) + 1e-30)
#                 # if branch in {"ls","both"} and (LS_log_loss is not None):
#                     # LS_BIC_table.loc[(p, r, s), "BIC"] = LS_log_loss + d_m * np.log(T) / T  # type: ignore[index]
#                 # if branch in {"mle","both"} and (ML_log_loss is not None):
#                 ML_BIC_table.loc[(p, r, s), "BIC"] = logdet_ml + d_m * np.log(T) / T  # type: ignore[index]

#     # LS_min_index = LS_initial_value = None
#     ML_min_index = ML_initial_value = None
#     # if branch in {"ls","both"} and LS_BIC_table is not None:
#     #     if LS_BIC_table["BIC"].notna().any():
#     #         LS_min_index = LS_BIC_table["BIC"].dropna().astype(float).idxmin()
#     #         LS_initial_value = LS_BIC_table.loc[LS_min_index]
#     if branch in {"mle","both"} and ML_BIC_table is not None:
#         if ML_BIC_table["BIC"].notna().any():
#             ML_min_index = ML_BIC_table["BIC"].dropna().astype(float).idxmin()
#             ML_initial_value = ML_BIC_table.loc[ML_min_index]
#     result = {
#         # "LS_BIC_table": LS_BIC_table,
#         # "LS_min_index": LS_min_index,
#         # "LS_initial_value": LS_initial_value,
#         "ML_BIC_table": ML_BIC_table,
#         "ML_min_index": ML_min_index,
#         "ML_initial_value": ML_initial_value,
#     }
#     return result


# def BIC_refine(
#     y: np.ndarray,
#     bic_result: Dict[str, Any],
#     *,
#     top_k: int = 5,
#     branch: str = "both",
#     n_random: int = 20,
#     n_iter: int = 50,
#     grid_mode: str = "auto",
#     stop_thres: float = 1e-3,
#     seed: Optional[int] = 42,
#     verbose: bool = False,
#     parallel: bool = False,
#     n_jobs: int = 1,
# ) -> Dict[str, Any]:
#     """
#     Refine BIC by re-evaluating the best `top_k` (p,r,s) combinations with
#     a multi-start BCD run and replacing/adding refined BIC values.
    
#     Parameters
#     ----------
#     y : array-like (T, N) or (N, T)
#     bic_result : output dict of `BIC(...)`
#     branch : {'ls','mle','both'}
#         Which branch tables to refine.
#     parallel : bool
#         Whether to parallelize the internal multi_start_BCD calls.
#     n_jobs : int
#         Number of workers to use when parallel=True.
    
#     Returns
#     -------
#     dict : a shallow copy of `bic_result` with additional columns
#            'BIC_refined' in the selected tables and a 'refined' summary.
#     """
#     y_TN = np.asarray(y)
#     # y_TN, _ = _to_TN(np.asarray(y))
#     T, N = y_TN.shape
#     P = int(np.floor(np.log(max(T, 3))))

#     out = dict(bic_result)
#     refined_summary: Dict[str, Any] = {}

#     def _refine_table(tbl_key: str) -> None:
#         df = out.get(tbl_key, None)
#         if df is None:
#             return
#         df = df.copy()
#         # pick top-k by original BIC, explicitly ignore NaNs
#         series = df["BIC"]
#         mask = series.notna()
#         top_idx = series[mask].astype(float).nsmallest(min(top_k, int(mask.sum()))).index.tolist()
#         refined = {}
#         for (p, r, s) in tqdm(top_idx, desc=f"Refining {tbl_key}", ncols=90):
#             ms = multi_start_BCD(
#                 y=y_TN, p=int(p), r=int(r), s=int(s),
#                 P=P, grid_mode=grid_mode, n_random=n_random,
#                 n_iter=n_iter, stop_thres=stop_thres,
#                 seed=seed, verbose=verbose,
#                 parallel=parallel, n_jobs=n_jobs
#             )
#             Sigma = np.asarray(ms["Sigma"])
#             bic_val = _bic_from_sigma(Sigma, T, int(p), int(r), int(s))
#             refined[(int(p), int(r), int(s))] = {
#                 "BIC_refined": bic_val,
#                 "init_refined": (tuple(np.asarray(ms["lmbd"]).reshape(-1).tolist()) +
#                                  tuple(np.asarray(ms["eta"]).reshape(-1, 2)[:, 0].tolist()) +
#                                  tuple(np.asarray(ms["eta"]).reshape(-1, 2)[:, 1].tolist()) if s > 0 else ())
#             }
#             # write back to table
#             df.loc[(p, r, s), "BIC_refined"] = bic_val
#         out[tbl_key] = df
#         refined_summary[tbl_key] = refined

#     if branch in ("ls", "both"):
#         _refine_table("LS_BIC_table")
#     if branch in ("mle", "both"):
#         _refine_table("ML_BIC_table")

#     out["refined"] = refined_summary
#     return out
