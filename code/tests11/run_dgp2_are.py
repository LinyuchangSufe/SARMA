"""
Runner for DGP2 ARE (Asymptotic Relative Efficiency) experiments.

Designs:
  - DGP2(a): N=6, (p,r,s)=(1,2,0), roots {lambda, -lambda}
  - DGP2(b): N=6, (p,r,s)=(1,0,1), one complex block (gamma, pi/4)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ortho_group

repo_root = Path(__file__).parent.parent.absolute()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from tests11.simulation_dgp1 import _phi_diag_values, simulate_with_params
except Exception:
    from simulation_dgp1 import _phi_diag_values, simulate_with_params

try:
    from src.utils.help_function import (
        asymptotic,
        gen_X_AR,
        gen_X_lmbd,
        gen_X_eta,
        get_epsilon,
    )
except Exception:
    from utils.help_function import (
        asymptotic,
        gen_X_AR,
        gen_X_lmbd,
        gen_X_eta,
        get_epsilon,
    )

try:
    from src.sarma.estimator import SARMAEstimator
except Exception:
    try:
        from estimator import SARMAEstimator
    except Exception:
        SARMAEstimator = None


def parse_csv_list(text: str, cast):
    vals = [token.strip() for token in str(text).split(",")]
    return [cast(v) for v in vals if v]


def git_commit_hash() -> str:
    env_commit = os.environ.get("SARMA_COMMIT", "").strip()
    if env_commit:
        return env_commit
    commit_file = Path(".sarma_commit")
    if commit_file.exists():
        return commit_file.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=repo_root.parent,
        ).strip()
    except Exception:
        return "unknown"


def _rot2(gamma: float, phi: float) -> np.ndarray:
    return np.array(
        [
            [gamma * np.cos(phi), gamma * np.sin(phi)],
            [-gamma * np.sin(phi), gamma * np.cos(phi)],
        ]
    )


def diagnose_imports():
    if SARMAEstimator is not None:
        print("[OK] SARMAEstimator imported successfully")
        return
    print("[ERROR] SARMAEstimator could not be imported. Checking dependencies...")
    try:
        import torch  # noqa: F401

        print("  - torch: OK")
    except Exception as exc:
        print(f"  - torch: FAIL ({exc})")


def fit_one_rep(y, p, r, s, est_kwargs=None):
    ek = est_kwargs or {}
    out = {"qml": None, "ls": None}
    if SARMAEstimator is None:
        out["qml"] = {"error": "SARMAEstimator_not_available"}
        out["ls"] = {"error": "SARMAEstimator_not_available"}
        return out
    try:
        t0 = time.time()
        est_q = SARMAEstimator(**ek)
        est_q.fit(y, p=p, r=r, s=s)
        t_q = time.time() - t0
        params_q = est_q.get_params()
        out["qml"] = {
            "lmbd": np.asarray(params_q.get("lmbd")) if params_q.get("lmbd") is not None else None,
            "eta": np.asarray(params_q.get("eta")) if params_q.get("eta") is not None else None,
            "G": np.asarray(params_q.get("G")) if params_q.get("G") is not None else None,
            "Sigma": np.asarray(params_q.get("Sigma")) if params_q.get("Sigma") is not None else None,
            "AsyVar": np.asarray(params_q.get("AsyVar")) if params_q.get("AsyVar") is not None else None,
            "time": t_q,
        }
        res_l = est_q._history["LS_result"]
        out["ls"] = {
            "lmbd": np.asarray(res_l.get("lmbd")) if res_l.get("lmbd") is not None else None,
            "eta": np.asarray(res_l.get("eta")) if res_l.get("eta") is not None else None,
            "G": np.asarray(res_l.get("G")) if res_l.get("G") is not None else None,
            "Sigma": np.asarray(res_l.get("Sigma")) if res_l.get("Sigma") is not None else None,
            "AsyVar": np.asarray(res_l.get("AsyVar")) if res_l.get("AsyVar") is not None else None,
            "time": t_q,
            "Loss": res_l.get("Loss"),
        }
    except Exception as exc:
        out["qml"] = {"error": str(exc)}
        out["ls"] = {"error": str(exc)}
    return out


def fit_one_from_info(rep_info, T, p, r, s, innovation_dist, est_kwargs=None):
    seed = rep_info.get("seed")
    params = rep_info.get("params")
    phi = params.get("Phi")
    theta = params.get("Theta")
    sigma = params.get("Sigma")
    if simulate_with_params is None:
        return {
            "qml": {"error": "simulate_with_params_not_available"},
            "ls": {"error": "simulate_with_params_not_available"},
        }
    y = simulate_with_params(
        Phi=phi,
        Theta=theta,
        Sigma=sigma,
        T=T,
        innovation_dist=innovation_dist,
        seed=seed,
        burn=1000,
    )
    return fit_one_rep(y, p=p, r=r, s=s, est_kwargs=est_kwargs)


def true_sarma_g(phi: np.ndarray, theta: np.ndarray, b: np.ndarray, p: int, r: int, s: int) -> np.ndarray:
    """Build true SARMA G blocks from true VARMA matrices and dynamic basis."""
    n = phi.shape[0]
    d = p + r + 2 * s
    blocks = [phi - theta]
    abar = b.T @ blocks[0]
    for h in range(r):
        blocks.append(np.outer(b[:, h], abar[h]))
    for h in range(s):
        i0 = r + 2 * h
        i1 = i0 + 1
        blocks.append(np.outer(b[:, i0], abar[i0]) + np.outer(b[:, i1], abar[i1]))
        blocks.append(np.outer(b[:, i0], abar[i1]) - np.outer(b[:, i1], abar[i0]))
    if len(blocks) != d:
        raise ValueError(f"Expected {d} G blocks, got {len(blocks)}.")
    return np.stack(blocks, axis=2).reshape(n, n, d)


def true_parameter_are(
    y: np.ndarray,
    lmbd: np.ndarray,
    eta: np.ndarray,
    g: np.ndarray,
    sigma: np.ndarray,
    p: int,
    r: int,
    s: int,
    P: int,
) -> dict[str, float]:
    """Compute Section 5.2 ARE at true parameters using sample averages."""
    x_ar = gen_X_AR(y, p)
    x_lmbd = gen_X_lmbd(lmbd, y, p, P) if r > 0 else np.zeros((y.shape[0] - p, y.shape[1], 0))
    x_eta = gen_X_eta(eta, y, p, P) if s > 0 else np.zeros((y.shape[0] - p, y.shape[1], 0))
    epsilon = get_epsilon(g, y, x_ar, x_lmbd, x_eta, p, r, s)

    asy_ls = asymptotic(lmbd, eta, g, sigma, y, epsilon, x_ar, x_lmbd, x_eta, p, r, s, P, method="ls")
    asy_qml = asymptotic(lmbd, eta, g, sigma, y, epsilon, x_ar, x_lmbd, x_eta, p, r, s, P, method="mle")

    alpha_dim = r + 2 * s + (p + r + 2 * s) * y.shape[1] * y.shape[1]
    v_ls = asy_ls[:alpha_dim, :alpha_dim]
    v_qml = asy_qml[:alpha_dim, :alpha_dim]
    sign_ls, logdet_ls = np.linalg.slogdet(v_ls)
    sign_qml, logdet_qml = np.linalg.slogdet(v_qml)
    if sign_ls <= 0 or sign_qml <= 0:
        return {
            "ARE": np.nan,
            "ARE_QML_over_LS": np.nan,
            "ARE_LS_over_QML": np.nan,
            "logdet_ls": logdet_ls,
            "logdet_qml": logdet_qml,
            "alpha_dim": alpha_dim,
            "status": "non_positive_det",
        }
    log_ls_over_qml = (logdet_ls - logdet_qml) / float(alpha_dim)
    log_qml_over_ls = -log_ls_over_qml
    return {
        # The manuscript table reports the efficiency of LS relative to QML,
        # i.e. values below one when QML is more efficient.
        "ARE": float(np.exp(log_qml_over_ls)),
        "ARE_QML_over_LS": float(np.exp(log_qml_over_ls)),
        "ARE_LS_over_QML": float(np.exp(log_ls_over_qml)),
        "logdet_ls": float(logdet_ls),
        "logdet_qml": float(logdet_qml),
        "alpha_dim": int(alpha_dim),
        "status": "ok",
    }


def run_dgp2_config(
    design: str,
    param_val: float,
    b: np.ndarray,
    phi: np.ndarray,
    n_reps: int = 100,
    T: int = 5000,
    innovation_dist: str = "normal",
    a: float = 0.0,
    out_dir: str = "code/tests11/results/Exp2",
    n_jobs: int = 100,
    seed_base: int = 12345,
    dgp_seed: int | None = None,
    phi_design: str = "fullrank_symmetric",
    phi_diag_vals: list[float] | None = None,
    commit_hash: str = "unknown",
    are_mode: str = "true-sample",
    P: int = 200,
):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if design == "2a":
        N, p, r, s = 6, 1, 2, 0
        lam_0 = param_val
        J = np.zeros((N, N))
        J[0, 0] = lam_0
        J[1, 1] = -lam_0
        gam_0, phi_0 = 0.0, 0.0
    elif design == "2b":
        N, p, r, s = 6, 1, 0, 1
        lam_0 = 0.0
        gam_0 = param_val
        phi_0 = np.pi / 4
        J = np.zeros((N, N))
        J[:2, :2] = _rot2(gam_0, phi_0)
    else:
        raise ValueError(f"Unknown design: {design}")

    if b.shape != (N, N):
        raise ValueError(f"Expected rotation matrix B shape {(N, N)}, got {b.shape}")
    if phi.shape != (N, N):
        raise ValueError(f"Expected Phi shape {(N, N)}, got {phi.shape}")
    dgp_seed = int(seed_base if dgp_seed is None else dgp_seed)
    phi_diag_vals = [] if phi_diag_vals is None else [float(x) for x in phi_diag_vals]

    ones = np.ones((N, 1))
    sigma_true = a * (ones @ ones.T) + (1 - a) * np.eye(N)
    config_name = f"DGP2{design}_N{N}_T{T}_{design[1]}{param_val:.1f}_a{a:.1f}_{innovation_dist}"
    if are_mode == "true-sample":
        print(f"\n[{config_name}] Computing one true-parameter ARE from a T={T} sample...")
    else:
        print(f"\n[{config_name}] Generating {n_reps} replications...")

    theta = b @ J @ b.T
    base_params = {"Phi": phi, "Theta": theta, "Sigma": sigma_true}
    reps = [{"seed": int(seed_base + rep), "params": base_params.copy()} for rep in range(n_reps)]

    if are_mode == "true-sample":
        if design == "2a":
            lmbd = np.array([lam_0, -lam_0], dtype=float)
            eta = np.empty((0, 2), dtype=float)
        else:
            lmbd = np.empty(0, dtype=float)
            eta = np.array([[gam_0, phi_0]], dtype=float)
        y = simulate_with_params(
            Phi=phi,
            Theta=theta,
            Sigma=sigma_true,
            T=T,
            innovation_dist=innovation_dist,
            seed=seed_base,
            burn=1000,
        )
        g_true = true_sarma_g(phi=phi, theta=theta, b=b, p=p, r=r, s=s)
        are_info = true_parameter_are(y, lmbd, eta, g_true, sigma_true, p, r, s, P=P)

        meta = {
            "config_name": config_name,
            "design": design,
            "param_val": float(param_val),
            "T": int(T),
            "N": int(N),
            "p": int(p),
            "r": int(r),
            "s": int(s),
            "a": float(a),
            "innovation_dist": innovation_dist,
            "are_mode": are_mode,
            "P": int(P),
            "n_reps": 1,
            "n_jobs": int(n_jobs),
            "dgp_seed": int(dgp_seed),
            "seed_base": int(seed_base),
            "seed_first": int(seed_base),
            "seed_last": int(seed_base),
            "phi_design": phi_design,
            "Phi_diag_vals": phi_diag_vals,
            "commit_hash": commit_hash,
            **are_info,
        }
        meta_path = out_root / f"{config_name}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        csv_are = out_root / f"{config_name}_ARE_per_rep.csv"
        pd.DataFrame([{"rep": "true_sample", **are_info}]).to_csv(csv_are, index=False)
        print(f"  Saved true-parameter ARE to: {csv_are}")
        return

    est_kwargs = {"P": 200, "n_iter": 100, "stop_thres": 1e-6}
    if n_jobs == 1:
        results = []
        for i, rep_info in enumerate(reps):
            if (i + 1) % max(10, n_reps // 10) == 0:
                print(f"  Fitting {i + 1}/{n_reps}...")
            results.append(
                fit_one_from_info(
                    rep_info,
                    T=T,
                    p=p,
                    r=r,
                    s=s,
                    innovation_dist=innovation_dist,
                    est_kwargs=est_kwargs,
                )
            )
    else:
        print(f"  Fitting in parallel with n_jobs={n_jobs}...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_one_from_info)(rep_info, T, p, r, s, innovation_dist, est_kwargs) for rep_info in reps
        )

    pkl_path = out_root / f"{config_name}_reps.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(
            {
                "reps": reps,
                "results": results,
                "config": {
                    "design": design,
                    "param": param_val,
                    "T": T,
                    "N": N,
                    "p": p,
                    "r": r,
                    "s": s,
                    "lam_0": lam_0,
                    "gam_0": gam_0,
                    "phi_0": phi_0,
                    "innovation_dist": innovation_dist,
                    "a": a,
                    "n_reps": n_reps,
                    "n_jobs": n_jobs,
                    "dgp_seed": int(dgp_seed),
                    "seed_base": seed_base,
                    "phi_design": phi_design,
                    "Phi_diag_vals": phi_diag_vals,
                    "commit_hash": commit_hash,
                    "are_mode": are_mode,
                },
            },
            fh,
        )

    meta_path = out_root / f"{config_name}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "config_name": config_name,
                "design": design,
                "param_val": float(param_val),
                "T": int(T),
                "N": int(N),
                "p": int(p),
                "r": int(r),
                "s": int(s),
                "a": float(a),
                "innovation_dist": innovation_dist,
                "n_reps": int(n_reps),
                "n_jobs": int(n_jobs),
                "dgp_seed": int(dgp_seed),
                "seed_base": int(seed_base),
                "seed_first": int(seed_base),
                "seed_last": int(seed_base + n_reps - 1),
                "phi_design": phi_design,
                "Phi_diag_vals": phi_diag_vals,
                "commit_hash": commit_hash,
                "are_mode": are_mode,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    L = p + r + 2 * s
    n = r + 2 * s + L * N * N
    are_list = []
    are_qml_over_ls_list = []
    are_ls_over_qml_list = []
    for res in results:
        q_res = res.get("qml", {}) if isinstance(res, dict) else {}
        l_res = res.get("ls", {}) if isinstance(res, dict) else {}
        if q_res.get("error") is not None or l_res.get("error") is not None:
            are_list.append(np.nan)
            are_qml_over_ls_list.append(np.nan)
            are_ls_over_qml_list.append(np.nan)
            continue
        asy_q = q_res.get("AsyVar")
        asy_l = l_res.get("AsyVar")
        try:
            asy_q = np.asarray(asy_q, dtype=float)
            asy_l = np.asarray(asy_l, dtype=float)
        except Exception:
            are_list.append(np.nan)
            are_qml_over_ls_list.append(np.nan)
            are_ls_over_qml_list.append(np.nan)
            continue
        if asy_q.ndim != 2 or asy_l.ndim != 2 or asy_q.shape[0] < n or asy_l.shape[0] < n:
            are_list.append(np.nan)
            are_qml_over_ls_list.append(np.nan)
            are_ls_over_qml_list.append(np.nan)
            continue
        A_q = asy_q[:n, :n]
        A_l = asy_l[:n, :n]
        sign_q, logdet_q = np.linalg.slogdet(A_q)
        sign_l, logdet_l = np.linalg.slogdet(A_l)
        if sign_q <= 0 or sign_l <= 0:
            are_list.append(np.nan)
            are_qml_over_ls_list.append(np.nan)
            are_ls_over_qml_list.append(np.nan)
            continue
        log_ls_over_qml = (logdet_l - logdet_q) / float(n)
        are_ls_over_qml = float(np.exp(log_ls_over_qml))
        are_qml_over_ls = float(np.exp(-log_ls_over_qml))
        are_list.append(are_qml_over_ls)
        are_qml_over_ls_list.append(are_qml_over_ls)
        are_ls_over_qml_list.append(are_ls_over_qml)

    df_are = pd.DataFrame(
        {
            "rep": list(range(len(are_list))),
            "ARE": are_list,
            "ARE_QML_over_LS": are_qml_over_ls_list,
            "ARE_LS_over_QML": are_ls_over_qml_list,
        }
    )
    mean_are = float(np.nanmean(are_list)) if len(are_list) > 0 else np.nan
    mean_qml_over_ls = float(np.nanmean(are_qml_over_ls_list)) if len(are_qml_over_ls_list) > 0 else np.nan
    mean_ls_over_qml = float(np.nanmean(are_ls_over_qml_list)) if len(are_ls_over_qml_list) > 0 else np.nan
    df_mean = pd.DataFrame(
        {
            "rep": ["mean"],
            "ARE": [mean_are],
            "ARE_QML_over_LS": [mean_qml_over_ls],
            "ARE_LS_over_QML": [mean_ls_over_qml],
        }
    )
    df_out = pd.concat([df_are, df_mean], ignore_index=True)
    csv_are = out_root / f"{config_name}_ARE_per_rep.csv"
    df_out.to_csv(csv_are, index=False)
    print(f"  Saved ARE per-rep and mean to: {csv_are}")


def main():
    parser = argparse.ArgumentParser(description="Run DGP2 ARE simulations with configurable parallelism.")
    parser.add_argument("--n-jobs", type=int, default=100, help="Parallel workers for replication fitting.")
    parser.add_argument("--n-reps", type=int, default=500, help="Replications per config.")
    parser.add_argument("--seed-base", type=int, default=12345, help="Base seed for replication-level innovations.")
    parser.add_argument("--dgp-seed", type=int, default=None, help="Seed used once to construct fixed Phi and dynamic basis B. Defaults to --seed-base for backward compatibility.")
    parser.add_argument("--N", type=int, default=6, help="Dimension (fixed to 6 for this experiment).")
    parser.add_argument("--T", type=int, default=5000, help="Sample size per replication.")
    parser.add_argument(
        "--T-values",
        type=str,
        default="",
        help="Optional comma-separated T list. If set, overrides --T and runs all listed T values.",
    )
    parser.add_argument("--P", type=int, default=200, help="Truncation length for SARMA filters/asymptotic matrices.")
    parser.add_argument(
        "--are-mode",
        choices=["true-sample", "fit-rep"],
        default="true-sample",
        help="true-sample matches Section 5.2; fit-rep keeps the old repeated-fitting diagnostic.",
    )
    parser.add_argument("--designs", type=str, default="2a,2b", help="Comma-separated design list.")
    parser.add_argument("--lambda-values", type=str, default="0.2,0.4,0.6,0.8", help="Lambda values for DGP2(a).")
    parser.add_argument("--gamma-values", type=str, default="0.2,0.4,0.6,0.8", help="Gamma values for DGP2(b).")
    parser.add_argument("--a-values", type=str, default="0.0,0.3,0.6,0.9", help="Compound covariance a values.")
    parser.add_argument("--dists", type=str, default="normal,t5", help="Innovation distributions.")
    parser.add_argument("--phi-design", type=str, default="fullrank_symmetric", choices=["rank2", "fullrank_stable", "fullrank_symmetric", "fullrank_symmetric_strong", "diag_balanced_pm05"], help="AR Phi spectrum design.")
    parser.add_argument("--out-dir", type=str, default="code/tests11/results/Exp2", help="Output directory.")
    args = parser.parse_args()

    if args.N != 6:
        raise ValueError("DGP2 is fixed at N=6 in this reproduction.")

    diagnose_imports()
    print()
    commit_hash = git_commit_hash()
    designs = parse_csv_list(args.designs, str)
    lambda_values = parse_csv_list(args.lambda_values, float)
    gamma_values = parse_csv_list(args.gamma_values, float)
    a_values = parse_csv_list(args.a_values, float)
    dists = parse_csv_list(args.dists, str)
    t_values = parse_csv_list(args.T_values, int) if str(args.T_values).strip() else [int(args.T)]
    dgp_seed = args.seed_base if args.dgp_seed is None else args.dgp_seed

    print("=" * 70)
    print("Running DGP2 ARE Experiments")
    print("=" * 70)
    print(
        f"commit={commit_hash}, n_jobs={args.n_jobs}, n_reps={args.n_reps}, "
        f"dgp_seed={dgp_seed}, seed_base={args.seed_base}, T_values={t_values}, "
        f"N={args.N}, are_mode={args.are_mode}, phi_design={args.phi_design}"
    )

    b_phi = np.eye(args.N) if args.phi_design == "diag_balanced_pm05" else ortho_group.rvs(args.N, random_state=dgp_seed)
    phi_diag_vals = _phi_diag_values(args.N, args.phi_design)
    phi_diag = np.diag(phi_diag_vals)
    Phi = b_phi @ phi_diag @ b_phi.T
    B = ortho_group.rvs(args.N, random_state=dgp_seed + 1)

    for T_cur in t_values:
        print("\n" + "=" * 70)
        print(f"Running T={T_cur}")
        print("=" * 70)

        if "2a" in designs:
            print("\n" + "=" * 70)
            print("Running DGP2(a): Real roots")
            print("=" * 70)
            for lam in lambda_values:
                for a in a_values:
                    for dist in dists:
                        run_dgp2_config(
                            design="2a",
                            param_val=lam,
                            b=B,
                            phi=Phi,
                            n_reps=args.n_reps,
                            T=T_cur,
                            innovation_dist=dist,
                            a=a,
                            out_dir=args.out_dir,
                            n_jobs=args.n_jobs,
                            seed_base=args.seed_base,
                            dgp_seed=dgp_seed,
                            phi_design=args.phi_design,
                            phi_diag_vals=phi_diag_vals.tolist(),
                            commit_hash=commit_hash,
                            are_mode=args.are_mode,
                            P=args.P,
                        )

        if "2b" in designs:
            print("\n" + "=" * 70)
            print("Running DGP2(b): Complex roots")
            print("=" * 70)
            for gamma in gamma_values:
                for a in a_values:
                    for dist in dists:
                        run_dgp2_config(
                            design="2b",
                            param_val=gamma,
                            b=B,
                            phi=Phi,
                            n_reps=args.n_reps,
                            T=T_cur,
                            innovation_dist=dist,
                            a=a,
                            out_dir=args.out_dir,
                            n_jobs=args.n_jobs,
                            seed_base=args.seed_base,
                            dgp_seed=dgp_seed,
                            phi_design=args.phi_design,
                            phi_diag_vals=phi_diag_vals.tolist(),
                            commit_hash=commit_hash,
                            are_mode=args.are_mode,
                            P=args.P,
                        )

    print("\n" + "=" * 70)
    print("DGP2 ARE experiments complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
