"""
Runner for DGP1 experiments (finite-sample performance of QMLE vs LS).

This script:
- Generates DGP1 replications with fixed DGP matrices.
- Fits QMLE/LS for each replication.
- Saves per-rep parameter estimates + ASD files.
- Saves run metadata per configuration (including commit hash, seed base, n_jobs).
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

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# Add parent directory to path so we can import src
repo_root = Path(__file__).parent.parent.absolute()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import simulation helper
try:
    from tests11.simulation_dgp1 import DGP1Simulation, simulate_with_params
except Exception:
    from simulation_dgp1 import DGP1Simulation, simulate_with_params

# Try to import estimator (may fail if runtime deps are missing)
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
    env_commit = Path(".sarma_commit")
    env_override = env_commit.read_text(encoding="utf-8").strip() if env_commit.exists() else ""
    if "SARMA_COMMIT" in os.environ and os.environ["SARMA_COMMIT"].strip():
        return os.environ["SARMA_COMMIT"].strip()
    if env_override:
        return env_override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=repo_root.parent,
        ).strip()
    except Exception:
        return "unknown"


def diagnose_imports():
    """Check what's missing for the estimator to work."""
    if SARMAEstimator is not None:
        print("[OK] SARMAEstimator imported successfully")
        return

    print("[ERROR] SARMAEstimator could not be imported. Checking dependencies...")
    print(f"  - Current sys.path[0]: {sys.path[0]}")
    print(f"  - Repo root: {repo_root}")
    try:
        import src.sarma.optim  # noqa: F401
        print("  - src.sarma.optim: OK")
    except Exception as exc:
        print(f"  - src.sarma.optim: FAIL ({exc})")

    try:
        import src.utils.tensorOp  # noqa: F401
        print("  - src.utils.tensorOp: OK")
    except Exception as exc:
        print(f"  - src.utils.tensorOp: FAIL ({exc})")

    try:
        import torch  # noqa: F401
        print("  - torch: OK")
    except Exception as exc:
        print(f"  - torch: FAIL ({exc})")


class _TqdmJoblib:
    """Context manager that lets tqdm track joblib batch completions."""

    def __init__(self, tqdm_object):
        self.tqdm_object = tqdm_object
        self._old_callback = None

    def __enter__(self):
        if self.tqdm_object is None:
            return None
        from joblib import parallel

        self._old_callback = parallel.BatchCompletionCallBack
        tqdm_object = self.tqdm_object
        old_callback = self._old_callback

        class TqdmBatchCompletionCallback(old_callback):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        return tqdm_object

    def __exit__(self, exc_type, exc, tb):
        if self.tqdm_object is not None:
            from joblib import parallel

            parallel.BatchCompletionCallBack = self._old_callback
            self.tqdm_object.close()
        return False


def fit_one_rep(y, p, r, s, est_kwargs=None, fit_kwargs=None):
    """Fit QMLE and LS to a single replication y."""
    ek = est_kwargs or {}
    fk = fit_kwargs or {}
    out = {"qml": None, "ls": None}

    if SARMAEstimator is None:
        out["qml"] = {"error": "SARMAEstimator_not_available"}
        out["ls"] = {"error": "SARMAEstimator_not_available"}
        return out

    try:
        t0 = time.time()
        est_q = SARMAEstimator(**ek)
        est_q.fit(y, p=p, r=r, s=s, **fk)
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


def fit_one_from_info(rep_info, T, innovation_dist, est_kwargs=None, init_mode: str = "search"):
    """Simulate y from rep_info (seed + params) then fit."""
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
    fit_kwargs = {}
    if init_mode == "true":
        fit_kwargs = {
            "init_lmbd": np.asarray([params["lam_0"]], dtype=float),
            "init_eta": np.asarray([[params["gam_0"], params["phi_0"]]], dtype=float),
            "init_label": "true",
        }
    elif init_mode != "search":
        raise ValueError(f"Unknown init_mode: {init_mode}")
    return fit_one_rep(y, p=1, r=1, s=1, est_kwargs=est_kwargs, fit_kwargs=fit_kwargs)


def run_config(
    n_reps=50,
    T=500,
    N=6,
    innovation_dist="normal",
    sigma_type="identity",
    out_dir="code/tests11/results/Exp1",
    n_jobs: int = 50,
    seed_base: int = 12345,
    dgp_seed: int | None = None,
    phi_design: str = "fullrank_symmetric",
    lam_0: float = -0.8,
    gam_0: float = 0.8,
    phi_0: float = np.pi / 4,
    init_mode: str = "search",
    g_update: str = "ols",
    g_gd_steps: int = 100,
    g_gd_tol: float = 1e-7,
    commit_hash: str = "unknown",
):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    dgp_seed = int(seed_base if dgp_seed is None else dgp_seed)

    sim = DGP1Simulation(n_reps=n_reps, seed_base=seed_base, verbose=True)
    config_name = f"DGP1_N{N}_T{T}_{innovation_dist}_{sigma_type}"
    print(f"Generating {n_reps} replications for {config_name} ...")
    reps = sim.generate_replications(
        config_name=config_name,
        T=T,
        N=N,
        p=1,
        r=1,
        s=1,
        lam_0=lam_0,
        gam_0=gam_0,
        phi_0=phi_0,
        innovation_dist=innovation_dist,
        sigma_type=sigma_type,
        phi_design=phi_design,
        dgp_seed=dgp_seed,
        fixed_dgp=True,
        fit_on_generate=False,
    )
    params0 = reps[0]["params"]
    phi_diag_vals = [
        float(x)
        for x in np.asarray(params0.get("Phi_diag_vals", []), dtype=float).ravel().tolist()
    ]

    # Save raw replications and metadata
    pkl_path = out_root / f"{config_name}_reps.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(
            {
                "results": sim.results,
                "run_meta": {
                    "config_name": config_name,
                    "N": int(N),
                    "T": int(T),
                    "n_reps": int(n_reps),
                    "dgp_seed": int(dgp_seed),
                    "seed_base": int(seed_base),
                    "seed_first": int(seed_base),
                    "seed_last": int(seed_base + n_reps - 1),
                    "n_jobs": int(n_jobs),
                    "phi_design": phi_design,
                    "Phi_diag_vals": phi_diag_vals,
                    "lam_0": float(lam_0),
                    "gam_0": float(gam_0),
                    "phi_0": float(phi_0),
                    "init_mode": init_mode,
                    "g_update": g_update,
                    "g_gd_steps": int(g_gd_steps),
                    "g_gd_tol": float(g_gd_tol),
                    "commit_hash": commit_hash,
                    "innovation_dist": innovation_dist,
                    "sigma_type": sigma_type,
                },
            },
            fh,
        )
    print(f"Saved replications to {pkl_path}")

    meta_path = out_root / f"{config_name}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "config_name": config_name,
                "N": int(N),
                "T": int(T),
                "n_reps": int(n_reps),
                "dgp_seed": int(dgp_seed),
                "seed_base": int(seed_base),
                "seed_first": int(seed_base),
                "seed_last": int(seed_base + n_reps - 1),
                "n_jobs": int(n_jobs),
                "phi_design": phi_design,
                "Phi_diag_vals": phi_diag_vals,
                "lam_0": float(lam_0),
                "gam_0": float(gam_0),
                "phi_0": float(phi_0),
                "init_mode": init_mode,
                "g_update": g_update,
                "g_gd_steps": int(g_gd_steps),
                "g_gd_tol": float(g_gd_tol),
                "commit_hash": commit_hash,
                "innovation_dist": innovation_dist,
                "sigma_type": sigma_type,
                "order": {"p": 1, "r": 1, "s": 1},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    est_kwargs = {
        "P": 200,
        "n_iter": 100,
        "stop_thres": 1e-6,
        "g_update": g_update,
        "g_gd_steps": g_gd_steps,
        "g_gd_tol": g_gd_tol,
    }
    if n_jobs == 1:
        results = []
        for i, rep_info in enumerate(reps):
            print(f"Fitting replication {i + 1}/{len(reps)}...")
            results.append(
                fit_one_from_info(
                    rep_info,
                    T=T,
                    innovation_dist=innovation_dist,
                    est_kwargs=est_kwargs,
                    init_mode=init_mode,
                )
            )
    else:
        print(f"Fitting replications in parallel with n_jobs={n_jobs}...")
        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=len(reps),
                desc=f"Fitting {config_name}",
                unit="rep",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        with _TqdmJoblib(progress):
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_one_from_info)(rep_info, T, innovation_dist, est_kwargs, init_mode) for rep_info in reps
            )

    first_info = reps[0]
    params0 = first_info["params"]
    N = params0["Phi"].shape[0]
    r = 1
    s = 1
    p = 1
    L = p + r + 2 * s

    cols = []
    for ii in range(r):
        cols.append(f"lmbd_{ii + 1}")
    for j in range(s):
        cols.append(f"eta_{j + 1}_gamma")
        cols.append(f"eta_{j + 1}_phi")
    for h in range(1, L + 1):
        for col in range(N):
            for row in range(N):
                cols.append(f"G{h}_r{row + 1}_c{col + 1}")
    sigma_cols = []
    for col in range(N):
        for row in range(col, N):
            sigma_cols.append(f"Sigma_r{row + 1}_c{col + 1}")
    cols.extend(sigma_cols)

    phi_true = params0["Phi"]
    theta_true = params0["Theta"]
    sigma_true = params0["Sigma"]
    b_mat = params0["B"]

    lmbd_true = [params0.get("lam_0")] if "lam_0" in params0 else [None]
    eta_true = [params0.get("gam_0"), params0.get("phi_0")] if "gam_0" in params0 else [None, None]

    g_true_blocks = []
    g1 = phi_true - theta_true
    g_true_blocks.append(g1)
    abar = np.linalg.inv(b_mat) @ g1
    for h in range(r):
        g_true_blocks.append(np.outer(b_mat[:, h], abar[h]))
    for h in range(s):
        g_true_blocks.append(
            np.outer(b_mat[:, r + 2 * h], abar[r + 2 * h]) + np.outer(b_mat[:, r + 2 * h + 1], abar[r + 2 * h + 1])
        )
        g_true_blocks.append(
            np.outer(b_mat[:, r + 2 * h], abar[r + 2 * h + 1]) - np.outer(b_mat[:, r + 2 * h + 1], abar[r + 2 * h])
        )

    true_vec = []
    true_vec.extend(lmbd_true)
    true_vec.extend(eta_true)
    for g_blk in g_true_blocks:
        true_vec += g_blk.ravel("F").tolist()
    for col in range(N):
        for row in range(col, N):
            true_vec.append(float(sigma_true[row, col]))

    if len(true_vec) != len(cols):
        raise ValueError(f"Length mismatch: true_vec={len(true_vec)} vs cols={len(cols)}")

    def extract_est_asd(method_res: dict):
        est_vec = [None] * len(cols)
        asd_vec = [None] * len(cols)
        if not isinstance(method_res, dict):
            return est_vec, asd_vec
        if method_res.get("error") is not None:
            return est_vec, asd_vec

        idx = 0
        lmbd_est = method_res.get("lmbd")
        eta_est = method_res.get("eta")
        g_est = method_res.get("G")
        sigma_est = method_res.get("Sigma")
        asy_var = method_res.get("AsyVar")

        if lmbd_est is not None:
            for ii in range(r):
                est_vec[idx] = float(lmbd_est[ii])
                idx += 1
        else:
            idx += r

        if eta_est is not None:
            for j in range(s):
                est_vec[idx] = float(eta_est[j, 0])
                est_vec[idx + 1] = float(eta_est[j, 1])
                idx += 2
        else:
            idx += 2 * s

        if g_est is not None:
            for h in range(L):
                gm = g_est[:, :, h]
                for col in range(N):
                    for row in range(N):
                        est_vec[idx] = float(gm[row, col])
                        idx += 1
        else:
            idx += N * N * L

        if sigma_est is not None:
            for col in range(N):
                for row in range(col, N):
                    est_vec[idx] = float(sigma_est[row, col])
                    idx += 1
        else:
            idx += len(sigma_cols)

        if asy_var is not None:
            try:
                v = np.asarray(np.sqrt(np.diag(asy_var))).reshape(-1)
                if v.shape[0] == len(cols):
                    asd_vec = v.tolist()
            except Exception:
                pass

        return est_vec, asd_vec

    qml_est_rows, qml_asd_rows = [], []
    ls_est_rows, ls_asd_rows = [], []
    qml_est_rows.append({"rep": "true", **{c: true_vec[i] for i, c in enumerate(cols)}})
    ls_est_rows.append({"rep": "true", **{c: true_vec[i] for i, c in enumerate(cols)}})
    qml_asd_rows.append({"rep": "true", **{c: None for c in cols}})
    ls_asd_rows.append({"rep": "true", **{c: None for c in cols}})

    for i, res in enumerate(results):
        q_est_vec, q_asd_vec = extract_est_asd(res.get("qml", {}))
        l_est_vec, l_asd_vec = extract_est_asd(res.get("ls", {}))
        qml_est_rows.append({"rep": i, **{c: q_est_vec[k] for k, c in enumerate(cols)}})
        qml_asd_rows.append({"rep": i, **{c: q_asd_vec[k] for k, c in enumerate(cols)}})
        ls_est_rows.append({"rep": i, **{c: l_est_vec[k] for k, c in enumerate(cols)}})
        ls_asd_rows.append({"rep": i, **{c: l_asd_vec[k] for k, c in enumerate(cols)}})

    df_qml_est = pd.DataFrame(qml_est_rows)
    df_qml_asd = pd.DataFrame(qml_asd_rows)
    df_ls_est = pd.DataFrame(ls_est_rows)
    df_ls_asd = pd.DataFrame(ls_asd_rows)

    csv_qml_est = out_root / f"{config_name}_QMLE_estimates.csv"
    csv_qml_asd = out_root / f"{config_name}_QMLE_asd.csv"
    csv_ls_est = out_root / f"{config_name}_LS_estimates.csv"
    csv_ls_asd = out_root / f"{config_name}_LS_asd.csv"
    df_qml_est.to_csv(csv_qml_est, index=False)
    df_qml_asd.to_csv(csv_qml_asd, index=False)
    df_ls_est.to_csv(csv_ls_est, index=False)
    df_ls_asd.to_csv(csv_ls_asd, index=False)

    print(f"Saved QMLE parameter estimates to {csv_qml_est}")
    print(f"Saved QMLE parameter ASDs      to {csv_qml_asd}")
    print(f"Saved LS   parameter estimates to {csv_ls_est}")
    print(f"Saved LS   parameter ASDs      to {csv_ls_asd}")


def main():
    parser = argparse.ArgumentParser(description="Run DGP1 simulation with configurable parallelism.")
    parser.add_argument("--n-jobs", type=int, default=50, help="Parallel workers for replication fitting.")
    parser.add_argument("--n-reps", type=int, default=500, help="Replications per configuration.")
    parser.add_argument("--seed-base", type=int, default=12345, help="Base seed for replication-level innovations.")
    parser.add_argument("--dgp-seed", type=int, default=None, help="Seed used once per case to construct fixed Phi/Theta/Sigma. Defaults to --seed-base for backward compatibility.")
    parser.add_argument("--N", type=int, default=6, help="Dimension N.")
    parser.add_argument("--T-values", type=str, default="500,1000", help="Comma-separated T list.")
    parser.add_argument("--dists", type=str, default="normal,t5", help="Comma-separated innovation distributions.")
    parser.add_argument("--sigma-types", type=str, default="identity,compound", help="Comma-separated covariance types.")
    parser.add_argument("--phi-design", type=str, default="fullrank_symmetric", choices=["rank2", "fullrank_stable", "fullrank_symmetric", "fullrank_symmetric_strong", "diag_balanced_pm05"], help="AR Phi spectrum design for DGP generation.")
    parser.add_argument("--lam-0", type=float, default=-0.8, help="True real-root lambda used in DGP1.")
    parser.add_argument("--gam-0", type=float, default=0.8, help="True complex-root modulus gamma used in DGP1.")
    parser.add_argument("--phi-0", type=float, default=float(np.pi / 4), help="True complex-root phase phi used in DGP1.")
    parser.add_argument("--init-mode", type=str, default="search", choices=["search", "true"], help="Initialization for fitted dynamic parameters.")
    parser.add_argument("--g-update", type=str, default="ols", choices=["ols", "gd"], help="Update method for the G block.")
    parser.add_argument("--g-gd-steps", type=int, default=100, help="Gradient-descent steps for the G block when --g-update=gd.")
    parser.add_argument("--g-gd-tol", type=float, default=1e-7, help="Relative loss tolerance for G-block gradient descent.")
    parser.add_argument("--out-dir", type=str, default="code/tests11/results/Exp1", help="Output directory.")
    args = parser.parse_args()

    diagnose_imports()
    print()
    t_values = parse_csv_list(args.T_values, int)
    dists = parse_csv_list(args.dists, str)
    sigma_types = parse_csv_list(args.sigma_types, str)
    commit_hash = git_commit_hash()
    dgp_seed = args.seed_base if args.dgp_seed is None else args.dgp_seed
    print(f"Using commit hash: {commit_hash}")
    print(
        f"n_jobs={args.n_jobs}, n_reps={args.n_reps}, "
        f"dgp_seed={dgp_seed}, seed_base={args.seed_base}, N={args.N}, phi_design={args.phi_design}, "
        f"lam_0={args.lam_0}, gam_0={args.gam_0}, phi_0={args.phi_0}, init_mode={args.init_mode}, "
        f"g_update={args.g_update}, g_gd_steps={args.g_gd_steps}, g_gd_tol={args.g_gd_tol}"
    )

    for sigma_type in sigma_types:
        for dist in dists:
            for t in t_values:
                run_config(
                    n_reps=args.n_reps,
                    T=t,
                    N=args.N,
                    innovation_dist=dist,
                    sigma_type=sigma_type,
                    out_dir=args.out_dir,
                    n_jobs=args.n_jobs,
                    seed_base=args.seed_base,
                    dgp_seed=dgp_seed,
                    phi_design=args.phi_design,
                    lam_0=args.lam_0,
                    gam_0=args.gam_0,
                    phi_0=args.phi_0,
                    init_mode=args.init_mode,
                    g_update=args.g_update,
                    g_gd_steps=args.g_gd_steps,
                    g_gd_tol=args.g_gd_tol,
                    commit_hash=commit_hash,
                )
    print("DGP1 experiments complete.")


if __name__ == "__main__":
    main()
