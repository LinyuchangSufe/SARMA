import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import permutations, product

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

# -----------------------------------------------------------------------------
# Imports from utils: prefer package-style; fall back to legacy "src.utils"
# -----------------------------------------------------------------------------
try:
    # preferred when "sarma" is a proper package
    from sarma.utils.help_function import *   
    from sarma.utils.tensorOp import *        
except Exception:  # pragma: no cover
    # legacy import path in current repo layout
    from src.utils.help_function import *     
    from src.utils.tensorOp import *         


# -----------------------------------------------------------------------------
#                           Main BCD‑LS / GLS routine
# -----------------------------------------------------------------------------
def _safe_inv_cov(S: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    try:
        return np.linalg.inv(S)
    except np.linalg.LinAlgError:
        n = S.shape[0]
        return np.linalg.inv(S + jitter * np.eye(n))


def _G_to_flat(G: np.ndarray) -> np.ndarray:
    N, _, d = G.shape
    return G.transpose(0, 2, 1).reshape(N, N * d)


def _flat_to_G(G_flat: np.ndarray, N: int, d: int) -> np.ndarray:
    return G_flat.reshape(N, d, N).transpose(0, 2, 1)


def get_G_gradient_descent(
    Y: np.ndarray,
    Z: np.ndarray,
    Sigma_inv: np.ndarray,
    *,
    G_init: Optional[np.ndarray] = None,
    n_steps: int = 100,
    tol: float = 1e-7,
) -> np.ndarray:
    T_, N = Y.shape
    _, _, d = Z.shape
    Z_vec = Z.transpose(0, 2, 1).reshape(T_, -1)
    if G_init is None:
        G_flat = np.zeros((N, N * d), dtype=float)
    else:
        G_flat = _G_to_flat(np.asarray(G_init, dtype=float)).copy()

    xtx = Z_vec.T @ Z_vec
    try:
        lip_x = float(np.linalg.eigvalsh(xtx).max())
        lip_s = float(np.linalg.eigvalsh((Sigma_inv + Sigma_inv.T) / 2).max())
        step = 0.9 / max((2.0 / T_) * lip_x * lip_s, 1e-8)
    except Exception:
        step = 1e-3

    prev_loss = np.inf
    for _ in range(max(1, int(n_steps))):
        residual = Y - Z_vec @ G_flat.T
        grad_flat = -2.0 * (residual @ Sigma_inv).T @ Z_vec / T_
        G_flat -= step * grad_flat
        if not np.all(np.isfinite(G_flat)):
            return _flat_to_G(G_flat=np.zeros((N, N * d), dtype=float), N=N, d=d)
        loss_val = float(np.trace(residual.T @ residual @ Sigma_inv) / T_)
        if np.isfinite(prev_loss) and abs(prev_loss - loss_val) / (abs(prev_loss) + 1e-8) < tol:
            break
        prev_loss = loss_val

    return _flat_to_G(G_flat, N, d)


def update_G_block(
    Y: np.ndarray,
    Z: np.ndarray,
    Sigma_inv: np.ndarray,
    *,
    g_update: str = "ols",
    G_init: Optional[np.ndarray] = None,
    g_gd_steps: int = 100,
    g_gd_tol: float = 1e-7,
) -> np.ndarray:
    if g_update == "ols":
        return get_G(Y, Z)
    if g_update == "gd":
        return get_G_gradient_descent(
            Y,
            Z,
            Sigma_inv,
            G_init=G_init,
            n_steps=g_gd_steps,
            tol=g_gd_tol,
        )
    raise ValueError("g_update must be one of {'ols', 'gd'}")


def BCD_SARMA(
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
    n_iter: int = 500,
    stop_thres: float = 1e-5,
    verbose: bool = False,
    Cal_AsyVar: bool = True,
    g_update: str = "ols",
    g_gd_steps: int = 100,
    g_gd_tol: float = 1e-7,
) -> pd.Series:
    """
    Block-Coordinate Descent for SARMA:
      - Updates λ (seasonal AR magnitudes) one-by-one
      - Updates η_k = (γ_k, φ_k) pairs one-by-one
      - Updates G via convex LS given current design matrices
      - Optionally refreshes Sigma (for 'mle' branch)

    Parameters
    ----------
    y : array (T, N)
        Time along axis 0, variables along axis 1.
    p, r, s : int
        AR order p, number of seasonal magnitudes r, number of seasonal pairs s.
    lmbd : array, optional
        Initial λ of shape (r,). If None, defaults to zeros (center of bounds).
    eta : array, optional
        Initial η of shape (s, 2): columns are (γ, φ).
        If None, defaults to γ=0.5 and φ=π/2 for each seasonal term.
    Sigma : array, optional
        Initial covariance (N x N). For 'ls' with Sigma=None, uses Identity.
        For 'mle', Sigma is re-estimated from residuals each iteration.
    esti_method : {'ls','mle'}
        LS uses fixed Sigma (I if None). MLE re-estimates Sigma from residuals.
    P : int
        Truncation parameter for constructing seasonal design matrices.
    n_iter : int
        Maximum number of BCD iterations.
    stop_thres : float
        Tolerance for both parameter change and relative loss change.
    verbose : bool
        If True, prints per-iteration diagnostics.

    Returns
    -------
    pd.Series with keys:
      - 'Loss_plot', 'A', 'lmbd', 'eta', 'G', 'Sigma',
        'Loss', 'iter_no', 'theta_diff', 'loss_diff'
    """
    # ---------- shapes & caches ----------
    T, N = y.shape
    P = min(T, P)

    # ---------- robust init for λ and η ----------
    # Defaults keep parameters in feasible interior, aiding stability.
    if lmbd is None:
        lmbd = 0.3 * np.ones(r, dtype=float)
    else:
        lmbd = np.asarray(lmbd, dtype=float).reshape(-1)
        assert lmbd.size == r, f"lmbd size {lmbd.size} != r={r}"

    if eta is None:
        # columns: (gamma in (0,1)), (phi in (0, pi))
        eta = np.column_stack([np.full(s, 0.5, dtype=float), np.full(s, np.pi/2, dtype=float)])
    else:
        eta = np.asarray(eta, dtype=float).reshape(-1, 2)
        assert eta.shape == (s, 2), f"eta shape {eta.shape} != (s,2)=({s},2)"

    # ---------- design matrices ----------
    Y = y[p:]
    X_AR = gen_X_AR(y, p)
    X_lmbd = gen_X_lmbd(lmbd, y, p, P)
    X_eta = gen_X_eta(eta, y, p, P)
    Z = np.concatenate([X_AR, X_lmbd, X_eta], axis=2)  # (T-p) x N x (p + r + 2*s)
    G = update_G_block(
        Y,
        Z,
        np.eye(N),
        g_update=g_update,
        G_init=None,
        g_gd_steps=g_gd_steps,
        g_gd_tol=g_gd_tol,
    )

    # ---------- initial Sigma_inv ----------
    if esti_method == "mle":
        epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s)
        Sigma_inv = _safe_inv_cov(epsilon.T @ epsilon / (T - p))
    elif (esti_method == "ls") and (Sigma is None):
        Sigma_inv = np.eye(N)
    else:
        Sigma_inv = _safe_inv_cov(Sigma)

    Loss_plot = []
    stop_counter = 0
    loss_diff = np.inf  # define for logging in the first iteration

    for it in range(n_iter):
        pre_lmbd, pre_eta = lmbd.copy(), eta.copy()

        # --- λ block: update each lambda_k within bounds ---
        for k in range(r):
            prev_val = float(lmbd[k])
            lmbd_opt = optimize_parameter(
                init_val=np.array([lmbd[k]]),
                update_design_matrix_fn=lambda lam, k=k: update_X_lmbd(X_lmbd, lam, k, y, p, P),
                vec_jac_fn=lambda lam, k=k: vec_jac_lmbd(
                    lam, k, G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s, P
                ),
                vec_jac_hess_fn=lambda lam, k=k: vec_jac_hess_lmbd(
                    lam, k, G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s, P
                ),
                loss_fn=lambda: loss_gls(G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s),
                bounds=[(-0.98, 0.98)],
            )
            cand = float(np.asarray(lmbd_opt).reshape(-1)[0])
            if not np.isfinite(cand):
                cand = prev_val
            lmbd[k] = float(np.clip(cand, -0.98, 0.98))

        # --- (γ, φ) block: update each pair within bounds ---
        for k in range(s):
            prev_eta = np.array(eta[k], dtype=float)
            eta_new = optimize_parameter(
                init_val=np.array(eta[k]),
                update_design_matrix_fn=lambda eta_k, k=k: update_X_eta(X_eta, eta_k, k, y, p, P),
                vec_jac_fn=lambda eta_k, k=k: vec_jac_eta(
                    eta_k, k, G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s, P
                ),
                vec_jac_hess_fn=lambda eta_k, k=k: vec_jac_hess_eta(
                    eta_k, k, G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s, P
                ),
                loss_fn=lambda: loss_gls(G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s),
                bounds=[(0.02, 0.98), (0.02 * np.pi, 0.98 * np.pi)],
            )
            eta_new = np.asarray(eta_new, dtype=float).reshape(2)
            if np.any(~np.isfinite(eta_new)):
                eta_new = prev_eta
            eta[k, 0] = float(np.clip(eta_new[0], 0.02, 0.98))
            eta[k, 1] = float(np.clip(eta_new[1], 0.02 * np.pi, 0.98 * np.pi))

        # --- G block ---
        Z = np.concatenate([X_AR, X_lmbd, X_eta], axis=2)
        G = update_G_block(
            Y,
            Z,
            Sigma_inv,
            g_update=g_update,
            G_init=G,
            g_gd_steps=g_gd_steps,
            g_gd_tol=g_gd_tol,
        )

        # --- recompute loss & (optionally) refresh Sigma for 'mle' ---
        if esti_method == "mle":
            epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s)
            Sigma = epsilon.T @ epsilon / (T - p)
            Sigma_inv = _safe_inv_cov(Sigma)
            loss_val = loss_mle(G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s)
        else:
            loss_val = loss_gls(G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s)

        if not np.isfinite(loss_val):
            if verbose:
                print(f"iter={it:4d} unstable loss encountered; stop early.")
            break

        Loss_plot.append(loss_val)

        # parameter change (∞-norm across blocks)
        theta_diff = max(
            np.max(np.abs(lmbd - pre_lmbd)) if r > 0 else 0.0,
            np.max(np.abs(eta - pre_eta)) if s > 0 else 0.0,
        )

        # relative loss change
        if it == 0:
            prev_loss = loss_val
            loss_diff = np.inf
        else:
            loss_diff = abs(loss_val - prev_loss) / (abs(prev_loss) + 1e-6)
            prev_loss = loss_val

        if verbose:
            print(f"iter={it:4d}  theta_diff={theta_diff:.3e}  loss_diff={loss_diff:.3e}  loss={loss_val:.6e}")

        # double criterion + small patience
        if (theta_diff < 100 * stop_thres) and (loss_diff < stop_thres):
            stop_counter += 1
        else:
            stop_counter = 0
        if stop_counter >= 5:
            break

    # final residuals and Sigma
    epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s)
    Sigma = (epsilon.T @ epsilon) / (T - p)  # N x N

    A = get_A(lmbd, eta, G, p, r, s, P)
    if Cal_AsyVar:
        AsyVar = asymptotic(lmbd,eta,G,Sigma,y,epsilon, X_AR,X_lmbd,X_eta,p,r,s,P,method = esti_method)
    # 将返回结果改为 Dict（比 Series 更灵活）或自定义 dataclass
    
    result = {
        "A": A,  # 返回完整的 A，或加参数控制截断长度
        "lmbd": lmbd,
        "eta": eta,
        "G": G,
        "Sigma": Sigma,
        "Sigma_inv": Sigma_inv,  # 返回逆矩阵避免重复计算
        "AsyVar": AsyVar if Cal_AsyVar else None,
        "Loss": loss_val,
        "Loss_plot": Loss_plot[:it + 1],
        "converged": stop_counter >= 5,  # 显式记录收敛状态
        "iter_no": it,
        "theta_diff": theta_diff,
        "loss_diff": loss_diff,
        "epsilon": epsilon,  # 残差，方便后续诊断
    }

    if verbose:
        print(
            "=======================================\n"
            "Stop due to SepEst criteria\n"
            f"Order: (p,r,s)=({p},{r},{s}); Iter: {it}\n"
            f"theta_diff: {theta_diff:.3e}; loss_diff: {loss_diff:.3e}\n"
            f"Final Loss: {loss_val:.6e}\n"
            f"Params:\n  lmbd={lmbd}\n  eta={eta}\n  G shape={None if G is None else G.shape}\n"
            f"  Sigma shape={Sigma.shape}\n"
            "============================================="
        )
    return result
    
    


def FGLS_SARMA(
    y: np.ndarray,
    p: int,
    r: int,
    s: int,
    *,
    lmbd: Optional[np.ndarray] = None,
    eta: Optional[np.ndarray] = None,
    Sigma: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Simple two-step Feasible GLS:
      1) Run BCD_SARMA under LS (Sigma=I or provided).
      2) Re-run BCD_SARMA under LS but with Sigma fixed to the residual covariance
         estimated from step (1). (If you want iterative FGLS, call this in a loop.)
    """
    # Step 1: initial LS pass (Sigma is ignored if None)
    res1 = BCD_SARMA(y, p, r, s, lmbd=lmbd, eta=eta, Sigma=Sigma, esti_method="ls")
    # Step 2: LS with fixed Sigma from residuals of step 1
    res2 = BCD_SARMA(y, p, r, s, lmbd=res1.lmbd, eta=res1.eta, Sigma=res1.Sigma, esti_method="gls")
    return res2

# multi_start_BCD(y,1,1,0)
# -----------------------------------------------------------------------------
#                       Multi‑start wrapper (grid / random)
# -----------------------------------------------------------------------------
def multi_start_BCD(
    y: np.ndarray,
    p: int,
    r: int,
    s: int,
    step = 0.1,
    *,
    Cal_AsyVar: bool = True,
    P: int = 150,
    grid_mode: str = "random",
    n_random: int = 2000,
    n_iter: int = 500,
    stop_thres: float = 1e-6,
    seed: Optional[int] = None,
    n_jobs_profiling: int = 1,
    verbose: bool = True,
    g_update: str = "ols",
    g_gd_steps: int = 100,
    g_gd_tol: float = 1e-7,
) -> tuple[pd.Series, pd.Series]:
    """
    Run BCD from multiple initial points and keep the best result.

    Parameters
    ----------
    grid_mode : {'auto','full','random'}
      - 'auto'  : if r,s ≤ 2 then 'full' else 'random'
      - 'full'  : enumerate all sparse-grid combos (λ,γ sorted desc, θ free)
      - 'random': draw n_random unique combinations from that grid
    """
    rng = np.random.default_rng(seed)

    # candidate pools (you can tune these centrally)
    lambda_pool = np.concatenate((np.arange(-0.9, -0.1, step=step+0.05), np.arange(0.1, 0.95, step=step+0.05)))
    gamma_pool = np.arange(0.2, 0.95, step=step)
    phi_pool = np.arange(0.2, 0.95, step=step) * np.pi
    seen = set()
    full_list = []

    # —— Build the "full" candidate list (sorted λ,γ to remove duplicates). θ can repeat. ——
    # if (r <= 2) and (s <= 2):
    for lam in permutations(lambda_pool, r):
        lam_s = tuple(sorted(lam, reverse=True))
        for gam in permutations(gamma_pool, s):
            gam_s = tuple(sorted(gam, reverse=True))
            for th in product(phi_pool, repeat=s):
                key = (lam_s, gam_s, th)
                if key not in seen:
                    seen.add(key)
                    full_list.append((np.array(lam_s), np.array(gam_s), np.array(th)))
    # else:  # allow repetition when r or s larger
    #     for lam in product(lambda_pool, repeat=r):
    #         lam_s = tuple(sorted(lam, reverse=True))
    #         for gam in product(gamma_pool, repeat=s):
    #             gam_s = tuple(sorted(gam, reverse=True))
    #             for th in product(phi_pool, repeat=s):
    #                 key = (lam_s, gam_s, th)
    #                 if key not in seen:
    #                     seen.add(key)
    #                     full_list.append((np.array(lam_s), np.array(gam_s), np.array(th)))

    # pick candidates
    if grid_mode == "full" or (grid_mode == "auto" and r <= 2 and s <= 2):
        candidate_list = full_list
    else:
        if n_random > len(full_list):
            if verbose:
                print(
                    f"Warning: n_random={n_random} exceeds available {len(full_list)} candidates; "
                    f"using full_list instead."
                )
            candidate_list = full_list
        else:
            idxs = rng.choice(len(full_list), size=n_random, replace=False)
            candidate_list = [full_list[i] for i in idxs]
    # ---------- Warm-start profiling (cheap OLS profile) ----------
    # Build AR design once
    T, N = y.shape
    X_AR = gen_X_AR(y, p)
    Y = y[p:]

    def profiled_loss(lam_vec, gam_vec, phi_vec,N,T,p):
        # assemble eta array shape (s,2)
        eta_arr = np.column_stack([np.array(gam_vec), np.array(phi_vec)]) if s > 0 else np.zeros((0, 2))
        X_lmbd = gen_X_lmbd(np.array(lam_vec), y, p, P) if r > 0 else np.zeros((T - p, N, 0))
        X_eta = gen_X_eta(eta_arr, y, p, P) if s > 0 else np.zeros((T - p, N, 0))
        Z = np.concatenate([X_AR, X_lmbd, X_eta], axis=2)
        G = get_G(Y, Z)
        epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s)
        Sigma = epsilon.T @ epsilon/(T-p)
        Sigma_inv = _safe_inv_cov(Sigma)
        val = loss_mle(G, Sigma_inv,y, X_AR, X_lmbd, X_eta, p, r, s)
        if not np.isfinite(val):
            return np.inf
        return val

    best_idx = None
    best_loss = np.inf
    best_candidate = None

    if verbose:
        print(f"Evaluating {len(candidate_list)} grid candidates (profiled LS)...")

    # Determine number of jobs based on unique lambda values
    unique_lams = set(tuple(lam0) for lam0, _, _ in candidate_list)
    n_jobs_profiling = min(min(len(unique_lams), len(candidate_list)), n_jobs_profiling) if len(unique_lams) > 1 else 1

    def evaluate_candidate(idx, lam0, gam0, ph0):
        try:
            loss_val = profiled_loss(lam0, gam0, ph0,N,T,p)
        except Exception:
            loss_val = np.inf
        if not np.isfinite(loss_val):
            loss_val = np.inf
        return idx, loss_val, lam0, gam0, ph0

    # Run profiling in parallel
    if n_jobs_profiling > 1 and len(candidate_list) > 1:
        results = Parallel(n_jobs=n_jobs_profiling, verbose=1 if verbose else 0)(
            delayed(evaluate_candidate)(idx, lam0, gam0, ph0)
            for idx, (lam0, gam0, ph0) in enumerate(candidate_list, 1)
        )
        for idx, loss_val, lam0, gam0, ph0 in results:
            if verbose:
                print(f"  candidate {idx}/{len(candidate_list)} loss={loss_val:.6e}")
            if loss_val < best_loss:
                best_loss = loss_val
                best_idx = idx
                best_candidate = (lam0.copy(), gam0.copy(), np.array(ph0).copy())
    else:
        # Sequential fallback
        for idx, (lam0, gam0, ph0) in enumerate(candidate_list, 1):
            idx, loss_val, lam0, gam0, ph0 = evaluate_candidate(idx, lam0, gam0, ph0)
            if verbose:
                print(f"  candidate {idx}/{len(candidate_list)} loss={loss_val:.6e}")
            if loss_val < best_loss:
                best_loss = loss_val
                best_idx = idx
                best_candidate = (lam0.copy(), gam0.copy(), np.array(ph0).copy())

    if best_candidate is None:
        raise RuntimeError("No valid candidate found during profiling grid search.")

    lam_init, gam_init, phi_init = best_candidate
    if verbose:
        print("Best profiled initializer found:")
        print(f"  λ_init={lam_init}\n  γ_init={gam_init}\n  φ_init={phi_init}")
    init_metadata = {
        "initial_lmbd": lam_init.copy(),
        "initial_eta": np.column_stack([gam_init, phi_init]),
        "profile_loss": float(best_loss),
        "profile_candidate_index": int(best_idx),
        "profile_candidate_count": int(len(candidate_list)),
    }

    # ---------- Local refinement under LS (small number of BCD updates) ----------
    eta_init = init_metadata["initial_eta"]
    res_refine = BCD_SARMA(
        y,
        p,
        r,
        s,
        lmbd=lam_init.copy(),
        eta=eta_init.copy(),
        Sigma=None,
        esti_method="ls",
        P=P,
        n_iter=n_iter,
        stop_thres=stop_thres,
        verbose=False,
        Cal_AsyVar=Cal_AsyVar,
        g_update=g_update,
        g_gd_steps=g_gd_steps,
        g_gd_tol=g_gd_tol,
    )
    res_refine.update(init_metadata)

    if verbose:
        print("Refinement complete. Starting full BCD with Sigma estimation...")

    # ---------- Full BCD (joint Sigma estimation) starting from refined init ----------
    final_res = BCD_SARMA(
        y,
        p,
        r,
        s,
        lmbd=np.asarray(res_refine["lmbd"], dtype=float).copy(),
        eta=np.asarray(res_refine["eta"], dtype=float).copy(),
        Sigma=None,
        esti_method="mle",
        P=P,
        n_iter=n_iter,
        stop_thres=stop_thres,
        verbose=verbose,
        Cal_AsyVar=Cal_AsyVar,
        g_update=g_update,
        g_gd_steps=g_gd_steps,
        g_gd_tol=g_gd_tol,
    )
    final_res.update(init_metadata)
    return res_refine, final_res
    # res_refine['AsyVar']
    # final_res['AsyVar']
