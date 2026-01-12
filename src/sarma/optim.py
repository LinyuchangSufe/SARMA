import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import permutations, product

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
    G = get_G(Y, Z)  # N x N x d tensor

    # ---------- initial Sigma_inv ----------
    if esti_method == "mle":
        epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s)
        Sigma_inv = np.linalg.inv(epsilon.T @ epsilon / (T - p))
    elif (esti_method == "ls") and (Sigma is None):
        Sigma_inv = np.eye(N)
    else:
        Sigma_inv = np.linalg.inv(Sigma)

    Loss_plot = []
    stop_counter = 0
    loss_diff = np.inf  # define for logging in the first iteration

    for it in range(n_iter):
        pre_lmbd, pre_eta = lmbd.copy(), eta.copy()

        # --- λ block: update each lambda_k within bounds ---
        for k in range(r):
            lmbd[k] = optimize_parameter(
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

        # --- (γ, φ) block: update each pair within bounds ---
        for k in range(s):
            eta[k] = optimize_parameter(
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

        # --- G block (closed-form LS) ---
        Z = np.concatenate([X_AR, X_lmbd, X_eta], axis=2)
        G = get_G(Y, Z)

        # --- recompute loss & (optionally) refresh Sigma for 'mle' ---
        if esti_method == "mle":
            epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s)
            Sigma = epsilon.T @ epsilon / (T - p)
            Sigma_inv = np.linalg.inv(Sigma)
            loss_val = loss_mle(G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s)
        else:
            loss_val = loss_gls(G, Sigma_inv, y, X_AR, X_lmbd, X_eta, p, r, s)

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
        # Use spherical working covariance (I) for profiling
        Sigma_inv = np.linalg.inv(Sigma)
        return loss_mle(G, Sigma_inv,y, X_AR, X_lmbd, X_eta, p, r, s)

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

    # ---------- Local refinement under LS (small number of BCD updates) ----------
    eta_init = np.column_stack([gam_init, phi_init]) 
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
        Cal_AsyVar = Cal_AsyVar
    )

    if verbose:
        print("Refinement complete. Starting full BCD with Sigma estimation...")

    # ---------- Full BCD (joint Sigma estimation) starting from refined init ----------
    final_res = BCD_SARMA(
        y,
        p,
        r,
        s,
        lmbd=lam_init.copy(),
        eta=eta_init.copy(),
        Sigma=None,
        esti_method="mle",
        P=P,
        n_iter=n_iter,
        stop_thres=stop_thres,
        verbose=verbose,
        Cal_AsyVar = Cal_AsyVar
    )
    return res_refine, final_res
    # res_refine['AsyVar']
    # final_res['AsyVar']