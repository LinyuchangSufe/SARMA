"""
Runner for DGP4 BIC model-selection experiments.

DGP4: N=6, true order (p,r,s)=(1,1,0), lambda in {0.5, 0.6, 0.7, 0.8}.
Candidates scanned by BIC: p,r,s <= 2, with max_dynamic_terms=N.
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

repo_root = Path(__file__).parent.parent.absolute()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from tests11.simulation_dgp1 import generate_dgp1, simulate_with_params
except Exception:
    from simulation_dgp1 import generate_dgp1, simulate_with_params

try:
    from src.sarma.estimator import SARMAEstimator
except Exception:
    try:
        from estimator import SARMAEstimator
    except Exception:
        SARMAEstimator = None

try:
    from src.sarma.selection import BIC_parallel_joblib, compute_bic_metrics
except Exception:
    try:
        from selection import BIC_parallel_joblib, compute_bic_metrics
    except Exception:
        BIC_parallel_joblib = None
        compute_bic_metrics = None

try:
    from src.sarma.optim import BCD_SARMA
except Exception:
    try:
        from optim import BCD_SARMA
    except Exception:
        BCD_SARMA = None


TRUE_ORDER = (1, 1, 0)
BIC_CRITERIA = {"current", "logdet_eff", "logdet_T"}


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


def diagnose_imports():
    if SARMAEstimator is not None:
        print("[OK] SARMAEstimator imported successfully")
        return
    print("[ERROR] SARMAEstimator could not be imported.")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _ascii_progress(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "] 0/0 (0.0%)"
    frac = min(max(done / total, 0.0), 1.0)
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} ({100.0 * frac:.1f}%)"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minute:02d}:{sec:02d}"
    return f"{minute:02d}:{sec:02d}"


class _SimpleTqdm:
    """Small tqdm-like fallback for remote environments without tqdm installed."""

    def __init__(self, total: int, desc: str = "", width: int = 30):
        self.total = max(0, int(total))
        self.desc = desc
        self.width = width
        self.n = 0
        self.start = time.time()
        self._render()

    def update(self, step: int = 1) -> None:
        self.n = min(self.total, self.n + int(step))
        self._render()

    def close(self) -> None:
        self._render()
        print(flush=True)

    def _render(self) -> None:
        frac = min(max(self.n / self.total, 0.0), 1.0) if self.total else 1.0
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        rate = self.n / elapsed if elapsed > 0 and self.n > 0 else 0.0
        if rate > 0 and self.n < self.total:
            eta = _format_duration((self.total - self.n) / rate)
        else:
            eta = "00:00"
        line = (
            f"\r{self.desc}: {100.0 * frac:5.1f}%|{bar}| "
            f"{self.n}/{self.total} [{_format_duration(elapsed)}<{eta}, {rate:.2f}it/s]"
        )
        print(line, end="", flush=True)


def _is_finite_order(p, r, s) -> bool:
    try:
        vals = [float(p), float(r), float(s)]
    except Exception:
        return False
    return all(np.isfinite(vals))


def classify_order(p, r, s, true_order: tuple[int, int, int] = TRUE_ORDER) -> str:
    """Classify selected order directly from (p,r,s), avoiding CSV bool/string issues."""
    if not _is_finite_order(p, r, s):
        return "error"
    p_i, r_i, s_i = int(p), int(r), int(s)
    p_true, r_true, s_true = true_order
    if (p_i, r_i, s_i) == true_order:
        return "true"
    if p_i < p_true or r_i < r_true or s_i < s_true:
        return "under"
    if p_i >= p_true and r_i >= r_true and s_i >= s_true:
        return "over"
    return "other"


def normalize_category(cat) -> str:
    text = str(cat).strip().lower()
    if text in {"true", "exact", "correct", "1", "yes"}:
        return "true"
    if text in {"under", "underfit", "underfitted"}:
        return "under"
    if text in {"over", "overfit", "overfitted"}:
        return "over"
    if text in {"error", "err", "failed", "nan", "none"}:
        return "error"
    return "other"


def _safe_float(value, default=np.nan) -> float:
    try:
        val = float(value)
    except Exception:
        return float(default)
    return float(val) if np.isfinite(val) else float(default)


def _safe_int(value, default: int = -1) -> int:
    try:
        val = float(value)
    except Exception:
        return int(default)
    if not np.isfinite(val):
        return int(default)
    return int(val)


def _load_existing_per_rep(per_rep_path: Path) -> pd.DataFrame:
    if not per_rep_path.exists():
        return pd.DataFrame(columns=["rep", "seed", "p", "r", "s", "category"])
    df = pd.read_csv(per_rep_path)
    need_cols = ["rep", "seed", "p", "r", "s", "category"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan
    out = df.copy()
    out["rep"] = pd.to_numeric(out["rep"], errors="coerce")
    out = out.dropna(subset=["rep"])
    out["rep"] = out["rep"].astype(int)
    out = out.sort_values("rep").drop_duplicates(subset=["rep"], keep="last").reset_index(drop=True)
    out["category"] = out.apply(
        lambda row: classify_order(row.get("p"), row.get("r"), row.get("s")),
        axis=1,
    )
    return out


def dgp4_irreducibility_metrics(params: dict, lam_val: float) -> dict:
    """Numerically summarize the N=3/N=6 DGP4 signal and irreducibility gap."""
    phi = np.asarray(params["Phi"], dtype=float)
    theta = np.asarray(params["Theta"], dtype=float)
    G1 = phi - theta
    if abs(lam_val) > 1e-12:
        G2 = theta @ (phi - theta) / float(lam_val)
    else:
        G2 = np.full_like(G1, np.nan)
    gap = G1 - G2

    def norm(x):
        return float(np.linalg.norm(x, ord="fro")) if np.all(np.isfinite(x)) else np.nan

    eig_phi = np.linalg.eigvals(phi)
    eig_theta = np.linalg.eigvals(theta)
    return {
        "G1_fro": norm(G1),
        "G2_fro": norm(G2),
        "G1_minus_G2_fro": norm(gap),
        "G2_over_G1_fro": float(norm(G2) / norm(G1)) if norm(G1) > 0 else np.nan,
        "irreducibility_gap_over_G1_fro": float(norm(gap) / norm(G1)) if norm(G1) > 0 else np.nan,
        "Phi_spectral_radius": float(np.max(np.abs(eig_phi))),
        "Theta_spectral_radius": float(np.max(np.abs(eig_theta))),
        "Phi_rank": int(np.linalg.matrix_rank(phi)),
        "Theta_rank": int(np.linalg.matrix_rank(theta)),
        "lambda": float(lam_val),
    }


def _array_to_text(value) -> str:
    if value is None:
        return ""
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return "[]"
        return np.array2string(arr.astype(float), precision=6, separator=" ", suppress_small=False)
    except Exception:
        return str(value)


def select_order_via_bic(
    y: np.ndarray,
    p_max: int = 2,
    r_max: int = 2,
    s_max: int = 2,
    *,
    P: int = 200,
    max_dynamic_terms: int | None = None,
    bic_step: float = 0.05,
    bic_grid_mode: str = "random",
    bic_n_random: int = 2000,
    bic_n_iter: int = 100,
    bic_stop_thres: float = 1e-6,
    bic_n_jobs_profiling: int = 1,
    init_mode: str = "search",
    bic_criterion: str = "current",
    bic_penalty_scale: float = 1.0,
    return_bic_table: bool = False,
    oracle_lam: float | None = None,
) -> dict:
    """Select order via BIC and return per-rep diagnostics."""
    if bic_criterion not in BIC_CRITERIA:
        raise ValueError(f"bic_criterion must be one of {sorted(BIC_CRITERIA)}, got {bic_criterion!r}.")

    if BIC_parallel_joblib is None and SARMAEstimator is None:
        return {"p": None, "r": None, "s": None, "category": "error", "error_message": "BIC import failed"}
    try:
        if BIC_parallel_joblib is not None:
            bic_res = BIC_parallel_joblib(
                y=y,
                p_m=p_max,
                r_m=r_max,
                s_m=s_max,
                P=P,
                n_jobs_BIC=1,
                max_dynamic_terms=y.shape[1] if max_dynamic_terms is None else max_dynamic_terms,
                step=bic_step,
                grid_mode=bic_grid_mode,
                n_iter=bic_n_iter,
                n_random=bic_n_random,
                stop_thres=bic_stop_thres,
                n_jobs_profiling=bic_n_jobs_profiling,
                bic_criterion=bic_criterion,
                penalty_scale=bic_penalty_scale,
                verbose=False,
            )
            p_sel, r_sel, s_sel = bic_res["ML_min_index"]
            table = bic_res["ML_BIC_table"].copy()
        else:
            est = SARMAEstimator(P=P, n_iter=bic_n_iter, stop_thres=bic_stop_thres)
            p_sel, r_sel, s_sel, _, _, _ = est._choose_orders_via_bic(y, n_jobs_BIC=1)
            table = None

        category = classify_order(p_sel, r_sel, s_sel)
        out = {
            "p": int(p_sel),
            "r": int(r_sel),
            "s": int(s_sel),
            "category": category,
            "bic_criterion": bic_criterion,
            "error_message": "",
        }

        if table is not None:
            finite_bic = pd.to_numeric(table["BIC"], errors="coerce")
            finite_bic = finite_bic[np.isfinite(finite_bic)]
            true_idx = TRUE_ORDER
            out["n_candidates"] = int(len(table))
            out["n_finite_candidates"] = int(len(finite_bic))
            out["n_error_candidates"] = int((table["Error"].astype(str).str.len() > 0).sum())
            out["nonfinite_candidate_rate"] = (
                float(1.0 - len(finite_bic) / len(table)) if len(table) else np.nan
            )
            out["best_bic"] = float(finite_bic.min()) if not finite_bic.empty else np.inf
            out["true_candidate_present"] = bool(true_idx in table.index)
            if true_idx in table.index:
                true_bic = pd.to_numeric(pd.Series([table.loc[true_idx, "BIC"]]), errors="coerce").iloc[0]
                out["true_bic"] = float(true_bic)
                out["gap_true_minus_best"] = float(true_bic - out["best_bic"])
                out["true_loss"] = _safe_float(table.loc[true_idx, "Loss"])
                out["true_logdet_sigma"] = _safe_float(table.loc[true_idx, "logdet_sigma"])
                out["true_trace_avg"] = _safe_float(table.loc[true_idx, "trace_avg"])
                out["true_converged"] = bool(table.loc[true_idx, "converged"])
                out["true_iter_no"] = _safe_int(table.loc[true_idx, "iter_no"])
            else:
                out["true_bic"] = np.nan
                out["gap_true_minus_best"] = np.nan
                out["true_candidate_present"] = False

            for col, suffix in [
                ("BIC_current", "current"),
                ("BIC_logdet_eff", "logdet_eff"),
                ("BIC_logdet_T", "logdet_T"),
            ]:
                vals = pd.to_numeric(table[col], errors="coerce")
                vals = vals[np.isfinite(vals)]
                if vals.empty:
                    out[f"selected_{suffix}"] = ""
                    out[f"selected_{suffix}_category"] = "error"
                    continue
                idx = tuple(int(x) for x in vals.idxmin())
                out[f"selected_{suffix}"] = f"({idx[0]},{idx[1]},{idx[2]})"
                out[f"selected_{suffix}_category"] = classify_order(*idx)

            if init_mode == "oracle_truth" and BCD_SARMA is not None and compute_bic_metrics is not None:
                try:
                    oracle_res = BCD_SARMA(
                        y,
                        TRUE_ORDER[0],
                        TRUE_ORDER[1],
                        TRUE_ORDER[2],
                        lmbd=np.array([oracle_lam if oracle_lam is not None else 0.5]),
                        eta=np.zeros((0, 2)),
                        Sigma=None,
                        esti_method="mle",
                        P=P,
                        n_iter=bic_n_iter,
                        stop_thres=bic_stop_thres,
                        verbose=False,
                        Cal_AsyVar=False,
                    )
                    oracle_metrics = compute_bic_metrics(
                        y,
                        TRUE_ORDER[0],
                        TRUE_ORDER[1],
                        TRUE_ORDER[2],
                        oracle_res,
                        criterion=bic_criterion,
                        penalty_scale=bic_penalty_scale,
                    )
                    out["oracle_true_bic"] = float(oracle_metrics["BIC"])
                    out["oracle_true_loss"] = float(oracle_metrics["Loss"])
                    out["oracle_true_logdet_sigma"] = float(oracle_metrics["logdet_sigma"])
                    out["oracle_true_trace_avg"] = float(oracle_metrics["trace_avg"])
                    out["gap_oracle_true_minus_best"] = float(oracle_metrics["BIC"] - out["best_bic"])
                    out["oracle_true_converged"] = bool(oracle_metrics["converged"])
                except Exception as exc:
                    out["oracle_true_error"] = repr(exc)

        if return_bic_table:
            out["_bic_table"] = table
        return out
    except Exception as exc:
        return {
            "p": None,
            "r": None,
            "s": None,
            "category": "error",
            "bic_criterion": bic_criterion,
            "error_message": repr(exc),
        }


def select_one_from_info(
    rep_info,
    T,
    innovation_dist,
    p_max=2,
    r_max=2,
    s_max=2,
    *,
    P=200,
    max_dynamic_terms=None,
    bic_step=0.05,
    bic_grid_mode="random",
    bic_n_random=2000,
    bic_n_iter=100,
    bic_stop_thres=1e-6,
    bic_n_jobs_profiling=1,
    init_mode="search",
    bic_criterion="current",
    bic_penalty_scale=1.0,
    return_bic_table=False,
    oracle_lam=None,
):
    seed = rep_info.get("seed")
    params = rep_info.get("params")
    phi = params.get("Phi")
    theta = params.get("Theta")
    sigma = params.get("Sigma")
    if simulate_with_params is None:
        return None, None, None, "error"
    y = simulate_with_params(
        Phi=phi,
        Theta=theta,
        Sigma=sigma,
        T=T,
        innovation_dist=innovation_dist,
        seed=seed,
        burn=1000,
    )
    return select_order_via_bic(
        y,
        p_max=p_max,
        r_max=r_max,
        s_max=s_max,
        P=P,
        max_dynamic_terms=max_dynamic_terms,
        bic_step=bic_step,
        bic_grid_mode=bic_grid_mode,
        bic_n_random=bic_n_random,
        bic_n_iter=bic_n_iter,
        bic_stop_thres=bic_stop_thres,
        bic_n_jobs_profiling=bic_n_jobs_profiling,
        init_mode=init_mode,
        bic_criterion=bic_criterion,
        bic_penalty_scale=bic_penalty_scale,
        return_bic_table=return_bic_table,
        oracle_lam=oracle_lam,
    )


def run_dgp3_config(
    lam_val: float,
    n_reps: int = 500,
    T: int = 500,
    N: int = 6,
    innovation_dist: str = "normal",
    sigma_type: str = "identity",
    out_dir: str = "code/tests11/results/Exp3",
    n_jobs: int = 100,
    seed_base: int = 12345,
    dgp_seed: int | None = None,
    p_max: int = 2,
    r_max: int = 2,
    s_max: int = 2,
    P: int = 200,
    max_dynamic_terms: int | None = None,
    bic_step: float = 0.05,
    bic_grid_mode: str = "random",
    bic_n_random: int = 2000,
    bic_n_iter: int = 100,
    bic_stop_thres: float = 1e-6,
    bic_n_jobs_profiling: int = 1,
    init_mode: str = "search",
    bic_criterion: str = "current",
    bic_penalty_scale: float = 1.0,
    save_bic_tables: str = "none",
    bic_top_k: int = 5,
    phi_design: str = "rank2",
    resume: bool = False,
    config_idx: int = 1,
    total_configs: int = 1,
    progress_every: int = 10,
    progress_style: str = "auto",
    commit_hash: str = "unknown",
):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if bic_criterion not in BIC_CRITERIA:
        raise ValueError(f"bic_criterion must be one of {sorted(BIC_CRITERIA)}, got {bic_criterion!r}.")
    if save_bic_tables not in {"none", "topk", "full"}:
        raise ValueError("save_bic_tables must be one of {'none','topk','full'}.")
    if progress_style not in {"auto", "tqdm", "line"}:
        raise ValueError("progress_style must be one of {'auto','tqdm','line'}.")

    p_true, r_true, s_true = TRUE_ORDER
    dgp_seed = int(seed_base if dgp_seed is None else dgp_seed)
    lam_0 = lam_val
    gam_0, phi_0 = 0.0, 0.0
    config_name = f"DGP4_N{N}_T{T}_lam{lam_val:.1f}_{innovation_dist}_{sigma_type}"
    per_rep_path = out_root / f"{config_name}_per_rep.csv"
    existing_df = _load_existing_per_rep(per_rep_path) if resume else pd.DataFrame(columns=["rep", "seed", "p", "r", "s", "category"])
    existing_reps = set(existing_df["rep"].tolist()) if not existing_df.empty else set()

    reps_to_run = []
    next_rep = 0
    while len(reps_to_run) < n_reps:
        if next_rep not in existing_reps:
            reps_to_run.append(next_rep)
        next_rep += 1

    total_after_this_batch = len(existing_reps) + len(reps_to_run)
    print(
        f"\n[{config_name}] ({config_idx}/{total_configs}) "
        f"running additional {len(reps_to_run)} reps (existing={len(existing_reps)}, total_after={total_after_this_batch})..."
    )
    progress_path = out_root / f"{config_name}_progress.json"
    _atomic_write_json(
        progress_path,
        {
            "config_name": config_name,
            "config_idx": int(config_idx),
            "total_configs": int(total_configs),
            "status": "running",
            "stage": "selecting_orders",
            "start_ts": t_start,
            "n_reps_batch": int(len(reps_to_run)),
            "n_reps_existing": int(len(existing_reps)),
            "n_reps_total_after": int(total_after_this_batch),
            "dgp_seed": int(dgp_seed),
            "seed_base": int(seed_base),
            "phi_design": phi_design,
            "n_done": 0,
            "pct_done": 0.0,
            "counts": {"under": 0, "exact": 0, "over": 0, "error": 0, "other": 0},
        },
    )

    _, base_params = generate_dgp1(
        T=1,
        N=N,
        p=p_true,
        r=r_true,
        s=s_true,
        lam_0=lam_0,
        gam_0=gam_0,
        phi_0=phi_0,
        innovation_dist=innovation_dist,
        sigma_type=sigma_type,
        phi_design=phi_design,
        seed=dgp_seed,
        burn=0,
    )
    dgp_diagnostics = dgp4_irreducibility_metrics(base_params, lam_val=lam_0)
    phi_diag_vals = [
        float(x)
        for x in np.asarray(base_params.get("Phi_diag_vals", []), dtype=float).ravel().tolist()
    ]
    reps = [{"rep": int(rep), "seed": int(seed_base + rep), "params": base_params.copy()} for rep in reps_to_run]

    selections: list[dict] = []
    candidate_tables: list[pd.DataFrame] = []
    cat_counts = {"under": 0, "exact": 0, "over": 0, "error": 0, "other": 0}
    progress_every = max(1, int(progress_every))
    progress_every = min(progress_every, max(1, int(len(reps))))

    # tqdm is useful on interactive TTYs; line-by-line progress is clearer in redirected logs.
    use_tqdm = (
        progress_style == "tqdm"
        or ((tqdm is not None) and progress_style == "auto" and bool(sys.stdout.isatty()))
    )
    pbar = None
    if use_tqdm:
        desc = f"{config_idx}/{total_configs} {config_name}"
        if tqdm is not None:
            pbar = tqdm(total=len(reps), desc=desc, dynamic_ncols=True)
        else:
            pbar = _SimpleTqdm(total=len(reps), desc=desc)
    else:
        if tqdm is not None and progress_style == "auto":
            print("  stdout is non-interactive; using line progress logs (tqdm disabled).", flush=True)

    def _consume_result(rep_info, result):
        result = dict(result)
        table = result.pop("_bic_table", None)
        p_sel, r_sel, s_sel = result.get("p"), result.get("r"), result.get("s")
        cat = classify_order(p_sel, r_sel, s_sel)
        result["category"] = cat
        result["rep"] = int(rep_info["rep"])
        result["seed"] = int(rep_info["seed"])
        result["dgp_seed"] = int(dgp_seed)
        selections.append(result)

        if table is not None and save_bic_tables != "none":
            table_out = table.copy().reset_index()
            table_out.insert(0, "dgp_seed", int(dgp_seed))
            table_out.insert(0, "seed", int(rep_info["seed"]))
            table_out.insert(0, "rep", int(rep_info["rep"]))
            table_out["candidate_category"] = table_out.apply(
                lambda row: classify_order(row["p"], row["r"], row["s"]),
                axis=1,
            )
            if save_bic_tables == "topk":
                table_out["_rank"] = pd.to_numeric(table_out["BIC"], errors="coerce").rank(method="first")
                selected_mask = (
                    (table_out["p"] == p_sel)
                    & (table_out["r"] == r_sel)
                    & (table_out["s"] == s_sel)
                )
                true_mask = (
                    (table_out["p"] == p_true)
                    & (table_out["r"] == r_true)
                    & (table_out["s"] == s_true)
                )
                table_out = table_out[
                    (table_out["_rank"] <= int(bic_top_k)) | selected_mask | true_mask
                ].copy()
                table_out = table_out.drop(columns=["_rank"])
            for col in ["lmbd", "eta", "initial_lmbd", "initial_eta"]:
                if col in table_out.columns:
                    table_out[col] = table_out[col].map(_array_to_text)
            candidate_tables.append(table_out)

        if cat == "under":
            cat_counts["under"] += 1
        elif cat == "true":
            cat_counts["exact"] += 1
        elif cat == "over":
            cat_counts["over"] += 1
        elif cat == "error":
            cat_counts["error"] += 1
        else:
            cat_counts["other"] += 1

    def _update_progress(done_batch: int, *, force: bool = False):
        if (not force) and done_batch != len(reps) and (done_batch % progress_every != 0):
            return
        done_total = len(existing_reps) + done_batch
        _atomic_write_json(
            progress_path,
            {
                "config_name": config_name,
                "config_idx": int(config_idx),
                "total_configs": int(total_configs),
                "status": "running" if done_batch < len(reps) else "completed",
                "stage": "selecting_orders" if done_batch < len(reps) else "done",
                "start_ts": t_start,
                "elapsed_sec": float(time.time() - t_start),
                "n_reps_batch": int(len(reps)),
                "n_reps_existing": int(len(existing_reps)),
                "n_reps_total_after": int(total_after_this_batch),
                "dgp_seed": int(dgp_seed),
                "seed_base": int(seed_base),
                "phi_design": phi_design,
                "n_done_batch": int(done_batch),
                "pct_done_batch": float(done_batch / len(reps)) if len(reps) > 0 else 1.0,
                "n_done": int(done_total),
                "pct_done": float(done_total / total_after_this_batch) if total_after_this_batch > 0 else 1.0,
                "counts": cat_counts,
            },
        )
        if pbar is None:
            print(
                f"  [{config_idx}/{total_configs}] {config_name} | "
                f"batch {_ascii_progress(done_batch, len(reps))} | "
                f"total {_ascii_progress(done_total, total_after_this_batch)} | "
                f"under/exact/over/error={cat_counts['under']}/{cat_counts['exact']}/{cat_counts['over']}/{cat_counts['error']}",
                flush=True,
            )

    def _run_one(rep_info):
        result = select_one_from_info(
            rep_info,
            T,
            innovation_dist,
            p_max,
            r_max,
            s_max,
            P=P,
            max_dynamic_terms=max_dynamic_terms,
            bic_step=bic_step,
            bic_grid_mode=bic_grid_mode,
            bic_n_random=bic_n_random,
            bic_n_iter=bic_n_iter,
            bic_stop_thres=bic_stop_thres,
            bic_n_jobs_profiling=bic_n_jobs_profiling,
            init_mode=init_mode,
            bic_criterion=bic_criterion,
            bic_penalty_scale=bic_penalty_scale,
            return_bic_table=save_bic_tables != "none",
            oracle_lam=lam_0,
        )
        return rep_info, result

    def _iter_tasks():
        return (delayed(_run_one)(rep_info) for rep_info in reps)

    if n_jobs == 1:
        for i, rep_info in enumerate(reps, start=1):
            _, result = _run_one(rep_info)
            _consume_result(rep_info, result)
            if pbar is not None:
                pbar.update(1)
            _update_progress(i)
    else:
        print(f"  Selecting orders in parallel with n_jobs={n_jobs}...")
        try:
            results_iter = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(_iter_tasks())
            for i, packed in enumerate(results_iter, start=1):
                rep_info, result = packed
                _consume_result(rep_info, result)
                if pbar is not None:
                    pbar.update(1)
                _update_progress(i)
        except TypeError:
            print("  joblib version does not support return_as=generator_unordered; progress updates only after completion.")
            results = Parallel(n_jobs=n_jobs)(_iter_tasks())
            for i, packed in enumerate(results, start=1):
                rep_info, result = packed
                _consume_result(rep_info, result)
                if pbar is not None:
                    pbar.update(1)
            _update_progress(len(selections), force=True)

    if pbar is not None:
        pbar.close()
    _update_progress(len(selections), force=True)

    batch_df = pd.DataFrame(selections)
    for col in ["rep", "seed", "p", "r", "s", "category"]:
        if col not in batch_df.columns:
            batch_df[col] = np.nan
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
    else:
        combined_df = batch_df.copy()
    combined_df = combined_df.sort_values("rep").drop_duplicates(subset=["rep"], keep="last").reset_index(drop=True)
    combined_df["category"] = combined_df.apply(
        lambda row: classify_order(row.get("p"), row.get("r"), row.get("s")),
        axis=1,
    )
    combined_df.to_csv(per_rep_path, index=False)

    if candidate_tables:
        candidate_df = pd.concat(candidate_tables, ignore_index=True)
        suffix = "full" if save_bic_tables == "full" else f"top{int(bic_top_k)}"
        candidate_path = out_root / f"{config_name}_BIC_candidates_{suffix}.csv"
        candidate_df.to_csv(candidate_path, index=False)

    pkl_path = out_root / f"{config_name}_reps.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(
            {
                "reps": reps,
                "selections": selections,
                "n_existing_before": int(len(existing_reps)),
                "n_batch": int(len(reps)),
                "n_total_after": int(len(combined_df)),
                "config": {
                    "lam_0": lam_0,
                    "T": T,
                    "N": N,
                    "p": p_true,
                    "r": r_true,
                    "s": s_true,
                    "innovation_dist": innovation_dist,
                    "sigma_type": sigma_type,
                    "n_reps_requested": int(n_reps),
                    "n_reps_batch": int(len(reps)),
                    "n_reps_total_after": int(len(combined_df)),
                    "n_jobs": n_jobs,
                    "dgp_seed": int(dgp_seed),
                    "seed_base": seed_base,
                    "p_max": p_max,
                    "r_max": r_max,
                    "s_max": s_max,
                    "P": P,
                    "max_dynamic_terms": N if max_dynamic_terms is None else int(max_dynamic_terms),
                    "bic_step": bic_step,
                    "bic_grid_mode": bic_grid_mode,
                    "bic_n_random": int(bic_n_random),
                    "bic_n_iter": int(bic_n_iter),
                    "bic_stop_thres": float(bic_stop_thres),
                    "bic_n_jobs_profiling": int(bic_n_jobs_profiling),
                    "bic_criterion": bic_criterion,
                    "bic_penalty_scale": float(bic_penalty_scale),
                    "save_bic_tables": save_bic_tables,
                    "bic_top_k": int(bic_top_k),
                    "phi_design": phi_design,
                    "Phi_diag_vals": phi_diag_vals,
                    "init_mode": init_mode,
                    "resume": bool(resume),
                    "progress_style": progress_style,
                    "commit_hash": commit_hash,
                    "dgp_diagnostics": dgp_diagnostics,
                },
            },
            fh,
        )

    meta_path = out_root / f"{config_name}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "config_name": config_name,
                "config_idx": int(config_idx),
                "total_configs": int(total_configs),
                "N": int(N),
                "T": int(T),
                "lam_0": float(lam_0),
                "order_true": {"p": p_true, "r": r_true, "s": s_true},
                "order_search_max": {"p_max": p_max, "r_max": r_max, "s_max": s_max},
                "P": int(P),
                "max_dynamic_terms": int(N) if max_dynamic_terms is None else int(max_dynamic_terms),
                "bic_step": float(bic_step),
                "bic_grid_mode": bic_grid_mode,
                "bic_n_random": int(bic_n_random),
                "bic_n_iter": int(bic_n_iter),
                "bic_stop_thres": float(bic_stop_thres),
                "bic_criterion": bic_criterion,
                "bic_penalty_scale": float(bic_penalty_scale),
                "save_bic_tables": save_bic_tables,
                "bic_top_k": int(bic_top_k),
                "phi_design": phi_design,
                "n_reps_requested": int(n_reps),
                "n_reps_batch": int(len(reps)),
                "n_reps_existing_before": int(len(existing_reps)),
                "n_reps_total_after": int(len(combined_df)),
                "n_jobs": int(n_jobs),
                "n_jobs_BIC": 1,
                "n_jobs_profiling": int(bic_n_jobs_profiling),
                "progress_every": int(progress_every),
                "progress_style": progress_style,
                "init_mode": init_mode,
                "resume": bool(resume),
                "dgp_seed": int(dgp_seed),
                "seed_base": int(seed_base),
                "seed_first_batch": int(seed_base + min(reps_to_run)) if reps_to_run else None,
                "seed_last_batch": int(seed_base + max(reps_to_run)) if reps_to_run else None,
                "Phi_diag_vals": phi_diag_vals,
                "innovation_dist": innovation_dist,
                "sigma_type": sigma_type,
                "commit_hash": commit_hash,
                "dgp_diagnostics": dgp_diagnostics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    counts = {
        "T": T,
        "N": N,
        "lam": lam_val,
        "innovation_dist": innovation_dist,
        "sigma_type": sigma_type,
        "n_reps": int(len(combined_df)),
        "n_reps_batch": int(len(reps)),
        "n_reps_requested": int(n_reps),
        "n_reps_existing_before": int(len(existing_reps)),
        "dgp_seed": int(dgp_seed),
        "seed_base": int(seed_base),
        "bic_criterion": bic_criterion,
        "bic_penalty_scale": float(bic_penalty_scale),
        "save_bic_tables": save_bic_tables,
        "phi_design": phi_design,
    }
    under = int((combined_df["category"] == "under").sum())
    true = int((combined_df["category"] == "true").sum())
    over = int((combined_df["category"] == "over").sum())
    error = int((combined_df["category"] == "error").sum())
    # Keep both naming schemes for compatibility:
    # - legacy: underfitted/correct/overfitted
    # - paper-facing: under/exact/over
    counts["underfitted"] = under
    counts["correct"] = true
    counts["overfitted"] = over
    counts["under"] = under
    counts["exact"] = true
    counts["over"] = over
    counts["error"] = error
    counts["other"] = int((combined_df["category"] == "other").sum())
    for key, value in dgp_diagnostics.items():
        counts[f"dgp_{key}"] = value
    total_valid = under + true + over
    if total_valid > 0:
        counts["freq_under"] = under / total_valid
        counts["freq_true"] = true / total_valid
        counts["freq_over"] = over / total_valid
    else:
        counts["freq_under"] = np.nan
        counts["freq_true"] = np.nan
        counts["freq_over"] = np.nan

    df = pd.DataFrame([counts])
    csv_out = out_root / f"{config_name}_BIC_summary.csv"
    df.to_csv(csv_out, index=False)

    elapsed_sec = float(time.time() - t_start)
    _atomic_write_json(
        progress_path,
        {
            "config_name": config_name,
            "config_idx": int(config_idx),
            "total_configs": int(total_configs),
            "status": "completed",
            "stage": "done",
            "start_ts": t_start,
            "elapsed_sec": elapsed_sec,
            "n_reps_batch": int(len(reps)),
            "n_reps_existing": int(len(existing_reps)),
            "n_reps_total_after": int(len(combined_df)),
            "dgp_seed": int(dgp_seed),
            "seed_base": int(seed_base),
            "seed_first_batch": int(seed_base + min(reps_to_run)) if reps_to_run else None,
            "seed_last_batch": int(seed_base + max(reps_to_run)) if reps_to_run else None,
            "phi_design": phi_design,
            "n_done_batch": int(len(reps)),
            "n_done": int(len(combined_df)),
            "pct_done_batch": 1.0,
            "pct_done": 1.0,
            "underfitted": int(under),
            "correct": int(true),
            "overfitted": int(over),
            "under": int(under),
            "exact": int(true),
            "over": int(over),
            "error": int(error),
        },
    )

    print(f"  Saved BIC summary to: {csv_out}")
    if candidate_tables:
        print(f"  Saved BIC candidate diagnostics to: {candidate_path}")
    print(f"    Underfitted: {under}/{len(combined_df)}, Correct: {true}/{len(combined_df)}, Overfitted: {over}/{len(combined_df)}")


def main():
    parser = argparse.ArgumentParser(description="Run DGP4 BIC simulations with configurable parallelism.")
    parser.add_argument("--N", type=int, default=6, help="Dimension of DGP4; use 3 for paper replication and 6 for the N=6 extension.")
    parser.add_argument("--n-jobs", type=int, default=100, help="Parallel workers over replications.")
    parser.add_argument("--n-reps", type=int, default=500, help="Replications per configuration.")
    parser.add_argument("--seed-base", type=int, default=12345, help="Base seed for replication-level innovations.")
    parser.add_argument("--dgp-seed", type=int, default=None, help="Seed used once per case to construct fixed Phi/Theta/Sigma. Defaults to --seed-base for backward compatibility.")
    parser.add_argument("--T-values", type=str, default="500,1000", help="Comma-separated T values.")
    parser.add_argument("--lam-values", type=str, default="0.5,0.6,0.7,0.8", help="Comma-separated lambda values.")
    parser.add_argument("--dists", type=str, default="normal,t5", help="Innovation distributions.")
    parser.add_argument("--sigma-types", type=str, default="identity,compound", help="Covariance types.")
    parser.add_argument("--p-max", type=int, default=2, help="Maximum p in BIC search.")
    parser.add_argument("--r-max", type=int, default=2, help="Maximum r in BIC search.")
    parser.add_argument("--s-max", type=int, default=2, help="Maximum s in BIC search.")
    parser.add_argument("--P", type=int, default=200, help="Truncation length for BIC/estimation helper matrices.")
    parser.add_argument("--bic-grid-mode", type=str, default="random", choices=["random", "full"], help="Initializer search mode inside BIC.")
    parser.add_argument("--bic-step", type=float, default=0.05, help="Grid step for initializer pools.")
    parser.add_argument("--bic-n-random", type=int, default=2000, help="Number of random initializer candidates when grid mode is random.")
    parser.add_argument("--bic-n-iter", type=int, default=100, help="BCD iterations for each SARMA fit in BIC.")
    parser.add_argument("--bic-stop-thres", type=float, default=1e-6, help="Convergence threshold for BIC inner fits.")
    parser.add_argument("--bic-max-dynamic-terms", type=int, default=6, help="Constraint r+2s<=max_dynamic_terms in BIC candidate orders.")
    parser.add_argument("--bic-n-jobs-profiling", type=int, default=1, help="Profiling parallelism inside a single BIC order fit.")
    parser.add_argument("--bic-criterion", type=str, default="current", choices=sorted(BIC_CRITERIA), help="Criterion used to select the BIC minimizer.")
    parser.add_argument("--bic-penalty-scale", type=float, default=1.0, help="Multiply the BIC penalty by this nonnegative finite scale.")
    parser.add_argument("--save-bic-tables", type=str, default="none", choices=["none", "topk", "full"], help="Save per-rep candidate BIC diagnostics.")
    parser.add_argument("--bic-top-k", type=int, default=5, help="When --save-bic-tables=topk, save top-k candidates plus true/selected candidates.")
    parser.add_argument("--phi-design", type=str, default="rank2", choices=["rank2", "fullrank_stable", "fullrank_symmetric", "fullrank_symmetric_strong"], help="AR Phi spectrum design for DGP generation.")
    parser.add_argument("--progress-every", type=int, default=10, help="Write progress.json every k finished replications.")
    parser.add_argument("--progress-style", type=str, default="auto", choices=["auto", "tqdm", "line"], help="Progress display style. Use tqdm in an attached screen TTY; use line for redirected logs.")
    parser.add_argument("--resume", action="store_true", help="Resume mode: run additional replications and append to existing per-rep results.")
    parser.add_argument("--init-mode", type=str, default="search", choices=["search", "oracle_truth"], help="search: normal BIC; oracle_truth: normal BIC plus true-order oracle-initialized diagnostic.")
    parser.add_argument("--out-dir", type=str, default="code/tests11/results/Exp3", help="Output directory.")
    args = parser.parse_args()

    diagnose_imports()
    print()
    commit_hash = git_commit_hash()
    dgp_seed = args.seed_base if args.dgp_seed is None else args.dgp_seed
    t_values = parse_csv_list(args.T_values, int)
    lam_values = parse_csv_list(args.lam_values, float)
    dists = parse_csv_list(args.dists, str)
    sigma_types = parse_csv_list(args.sigma_types, str)
    print("=" * 70)
    print("Running DGP4 BIC Model Selection Experiments")
    print("=" * 70)
    print(
        f"commit={commit_hash}, n_jobs={args.n_jobs}, n_reps={args.n_reps}, "
        f"N={args.N}, dgp_seed={dgp_seed}, seed_base={args.seed_base}, "
        f"p_max={args.p_max}, r_max={args.r_max}, s_max={args.s_max}, "
        f"bic_grid_mode={args.bic_grid_mode}, bic_n_random={args.bic_n_random}, "
        f"bic_criterion={args.bic_criterion}, bic_penalty_scale={args.bic_penalty_scale}, "
        f"save_bic_tables={args.save_bic_tables}, "
        f"phi_design={args.phi_design}, progress_every={args.progress_every}, "
        f"progress_style={args.progress_style}, "
        f"resume={args.resume}, init_mode={args.init_mode}"
    )

    configs = [
        (lam, T, dist, sigma_type)
        for lam in lam_values
        for T in t_values
        for dist in dists
        for sigma_type in sigma_types
    ]
    total_configs = len(configs)

    for idx, (lam, T, dist, sigma_type) in enumerate(configs, start=1):
        run_dgp3_config(
            lam_val=lam,
            n_reps=args.n_reps,
            T=T,
            N=args.N,
            innovation_dist=dist,
            sigma_type=sigma_type,
            out_dir=args.out_dir,
            n_jobs=args.n_jobs,
            seed_base=args.seed_base,
            dgp_seed=dgp_seed,
            p_max=args.p_max,
            r_max=args.r_max,
            s_max=args.s_max,
            P=args.P,
            max_dynamic_terms=args.bic_max_dynamic_terms,
            bic_step=args.bic_step,
            bic_grid_mode=args.bic_grid_mode,
            bic_n_random=args.bic_n_random,
            bic_n_iter=args.bic_n_iter,
            bic_stop_thres=args.bic_stop_thres,
            bic_n_jobs_profiling=args.bic_n_jobs_profiling,
            bic_criterion=args.bic_criterion,
            bic_penalty_scale=args.bic_penalty_scale,
            save_bic_tables=args.save_bic_tables,
            bic_top_k=args.bic_top_k,
            phi_design=args.phi_design,
            init_mode=args.init_mode,
            resume=args.resume,
            config_idx=idx,
            total_configs=total_configs,
            progress_every=args.progress_every,
            progress_style=args.progress_style,
            commit_hash=commit_hash,
        )

    print("\n" + "=" * 70)
    print("DGP4 BIC selection experiments complete.")


if __name__ == "__main__":
    main()
