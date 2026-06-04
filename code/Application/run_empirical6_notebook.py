from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional runtime dependency
    threadpool_limits = None


SCRIPT_PATH = Path(__file__).resolve()
CODE_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = CODE_DIR.parents[0]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from OtherModel.IOLS_VARMA1 import varma11_iols_fit_no_c, varma11_iols_forecast1
from src.sarma.estimator import SARMAEstimator


DEFAULT_METHODS = ["SARMA_LS", "SARMA_MLE", "VAR_LS", "VARMA_IOLS", "VARMA_MLE"]


@dataclass
class MethodRunResult:
    method: str
    origin: int
    train_end: int
    target_index: int
    status: str
    error_message: str
    fit_seconds: float
    predict_seconds: float
    forecast: np.ndarray
    diagnostics: Dict[str, object] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce 6D Empirical example notebook with full-sample BIC SARMA order."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=CODE_DIR / "data" / "FRED-MD.csv",
        help="Path to FRED-MD.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_DIR / "Application" / "results" / "empirical6_notebook",
        help="Root directory for run artifacts.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--order-source",
        choices=["bic_full_sample", "fixed"],
        default="bic_full_sample",
        help="How to choose SARMA order used in rolling refits.",
    )
    parser.add_argument("--fixed-sarma-p", type=int, default=1)
    parser.add_argument("--fixed-sarma-r", type=int, default=1)
    parser.add_argument("--fixed-sarma-s", type=int, default=0)
    parser.add_argument(
        "--sarma-label-mode",
        choices=["notebook_preserved", "corrected"],
        default="notebook_preserved",
        help=(
            "Notebook-preserved keeps legacy mapping where SARMA_LS stores MLE forecast and "
            "SARMA_MLE stores LS forecast."
        ),
    )
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument(
        "--origins",
        type=int,
        default=None,
        help="Number of rolling origins. Default equals test-size.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--rolling-method",
        choices=["rolling", "recursive"],
        default="rolling",
    )
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--bic-jobs", type=int, default=3)
    parser.add_argument("--bic-penalty-scale", type=float, default=1.0)
    parser.add_argument("--sarma-profile-jobs", type=int, default=1)
    parser.add_argument("--var-lag", type=int, default=2)
    parser.add_argument("--varma-n-jobs", type=int, default=50)
    parser.add_argument("--iols-max-iter", type=int, default=200)
    parser.add_argument("--iols-tol", type=float, default=1e-8)
    parser.add_argument("--iols-burn-in", type=int, default=5)
    parser.add_argument("--iols-ridge", type=float, default=1e-8)
    parser.add_argument("--maxiter-mle", type=int, default=200)
    parser.add_argument(
        "--verbose-sarma",
        action="store_true",
        help="Emit SARMA estimator logging in run.log.",
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash(repo_root: Path) -> str:
    env_commit = os.environ.get("SARMA_COMMIT")
    if env_commit:
        return env_commit
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_status_porcelain(repo_root: Path) -> List[str]:
    env_status = os.environ.get("SARMA_GIT_STATUS_JSON")
    if env_status:
        try:
            parsed = json.loads(env_status)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [line for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def make_run_id(args: argparse.Namespace, git_hash: str) -> str:
    if args.run_id:
        return args.run_id
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"empirical6_{git_hash}_{stamp}"


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("empirical6_notebook")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.__stdout__)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def is_missing(value: object) -> bool:
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def to_jsonable(value: object) -> object:
    if value is None or is_missing(value):
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    return value


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def json_array(value: object) -> str:
    if value is None or is_missing(value):
        return ""
    return json.dumps(np.asarray(value).tolist())


def bic_table_for_csv(bic_table: pd.DataFrame) -> pd.DataFrame:
    table = bic_table.reset_index().copy()
    for column in ["lmbd", "eta"]:
        if column in table.columns:
            table[column] = table[column].apply(json_array)
    return table


def fred_md_transform(x: pd.Series, code: int) -> pd.Series:
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 4:
        return np.log(x)
    if code == 5:
        return np.log(x).diff()
    if code == 6:
        return np.log(x).diff().diff()
    raise ValueError(f"Unknown transform code: {code}")


def load_and_transform_data(data_path: Path) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    labels = ["IP", "UR", "CPI", "RW", "CONS", "FFR"]
    vars_ = ["INDPRO", "UNRATE", "CPIAUCSL", "W875RX1", "DPCERA3M086SBEA", "FEDFUNDS"]
    transform_codes = {
        "INDPRO": 5,
        "UNRATE": 2,
        "CPIAUCSL": 6,
        "W875RX1": 5,
        "DPCERA3M086SBEA": 5,
        "FEDFUNDS": 2,
    }

    df_raw = pd.read_csv(data_path, header=0, index_col=0)
    df = df_raw[vars_]
    df = df.iloc[1:]

    df_trans = pd.DataFrame(index=df.index)
    for var_name in vars_:
        df_trans[var_name] = fred_md_transform(df[var_name], transform_codes[var_name])
    df_trans = df_trans.dropna(how="any")

    df_std = (df_trans - df_trans.mean()) / df_trans.std(ddof=0)
    df_std.columns = labels
    y = df_std.to_numpy(dtype=float)
    if np.isnan(y).any():
        raise ValueError("Transformed data contains missing values.")

    metadata = {
        "raw_columns": vars_,
        "label_columns": labels,
        "transform_codes": transform_codes,
        "index_start": str(df_std.index[0]),
        "index_end": str(df_std.index[-1]),
        "raw_rows_after_drop": int(df_std.shape[0]),
    }
    return df_std, y, metadata


def run_full_sample_bic(
    y: np.ndarray,
    args: argparse.Namespace,
    logger: logging.Logger,
    run_dir: Path,
) -> Dict[str, object]:
    logger.info(
        "Running SARMAEstimator full-sample fit with BIC order selection "
        "(n_jobs_BIC=%d, bic_penalty_scale=%.3f)",
        args.bic_jobs,
        args.bic_penalty_scale,
    )
    start = time.perf_counter()
    estimator = SARMAEstimator(seed=args.seed, verbose=args.verbose_sarma)
    if args.verbose_sarma:
        estimator.fit(
            y,
            n_jobs_BIC=args.bic_jobs,
            n_jobs_profiling=args.sarma_profile_jobs,
            bic_penalty_scale=args.bic_penalty_scale,
        )
    else:
        with open(os.devnull, "w", encoding="utf-8") as null_fh, redirect_stdout(null_fh), redirect_stderr(null_fh):
            estimator.fit(
                y,
                n_jobs_BIC=args.bic_jobs,
                n_jobs_profiling=args.sarma_profile_jobs,
                bic_penalty_scale=args.bic_penalty_scale,
            )
    elapsed = time.perf_counter() - start

    bic_result = estimator._history.get("bic")
    if not isinstance(bic_result, dict):
        raise RuntimeError("Estimator did not expose BIC history.")
    if "ML_min_index" not in bic_result:
        raise RuntimeError("BIC result missing ML_min_index.")

    selected_order = tuple(int(v) for v in bic_result["ML_min_index"])
    bic_table = bic_result.get("ML_BIC_table")
    selected_row: Dict[str, object] = {}
    if isinstance(bic_table, pd.DataFrame):
        bic_table_for_csv(bic_table).to_csv(run_dir / "bic_table.csv", index=False)
        selected_row = bic_table.loc[selected_order].to_dict()

    selection = {
        "order_source": "bic_full_sample",
        "selected_order": list(selected_order),
        "selected_lmbd": to_jsonable(bic_result.get("ML_lmbd_value")),
        "selected_eta": to_jsonable(bic_result.get("ML_eta_value")),
        "selected_row": to_jsonable(selected_row),
        "elapsed_seconds": elapsed,
        "n_jobs_BIC": args.bic_jobs,
        "bic_penalty_scale": args.bic_penalty_scale,
    }
    save_json(run_dir / "bic_selection.json", selection)
    logger.info(
        "Selected BIC order on full sample: (p,r,s)=%s in %.2f seconds",
        selected_order,
        elapsed,
    )
    return selection


def build_varmax_start_params_from_ab(mod: VARMAX, a_mat: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
    start_params = np.array(mod.start_params, dtype=float)
    names = mod.param_names
    cols = list(mod.endog_names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    n_dim = len(cols)

    if a_mat.shape != (n_dim, n_dim) or b_mat.shape != (n_dim, n_dim):
        raise ValueError(f"Expected A,B with shape ({n_dim},{n_dim}), got {a_mat.shape}, {b_mat.shape}.")

    for dep_i, dep in enumerate(cols):
        for reg_j, reg in enumerate(cols):
            key_ar = f"L1.{reg}.{dep}"
            if key_ar in name_to_idx:
                start_params[name_to_idx[key_ar]] = a_mat[dep_i, reg_j]
            key_ma = f"L1.e({reg}).{dep}"
            if key_ma in name_to_idx:
                start_params[name_to_idx[key_ma]] = b_mat[dep_i, reg_j]
    return start_params


def varmax_fit_forecast_with_iols_start(
    y: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    steps: int = 1,
    maxiter: int = 200,
) -> np.ndarray:
    df = pd.DataFrame(np.asarray(y, dtype=float), columns=[f"y{i + 1}" for i in range(y.shape[1])])
    mod = VARMAX(df, order=(1, 1), trend="n")
    start_params = build_varmax_start_params_from_ab(mod, a_mat, b_mat)
    res = mod.fit(start_params=start_params, method="lbfgs", maxiter=maxiter, disp=False)
    fc = np.asarray(res.forecast(steps=steps), dtype=float)
    return fc[-1, :]


def rolling_history(y_train: np.ndarray, y_test: np.ndarray, t: int, method: str) -> np.ndarray:
    if method == "rolling":
        if t > y_train.shape[0]:
            y_all = np.vstack((y_train, y_test[:t]))
            return y_all[-y_train.shape[0] :, :]
        return np.vstack((y_train[t:], y_test[:t]))
    if method == "recursive":
        return np.vstack((y_train, y_test[:t]))
    raise ValueError("method must be 'rolling' or 'recursive'")


def sarma_diag_from_fit(branch_name: str, fit_result: Dict[str, object]) -> Dict[str, object]:
    return {
        "sarma_branch_actual": branch_name,
        "sarma_initial_lmbd": json_array(fit_result.get("initial_lmbd")),
        "sarma_initial_eta": json_array(fit_result.get("initial_eta")),
        "sarma_profile_loss": float(fit_result.get("profile_loss", np.nan)),
        "sarma_profile_candidate_index": int(fit_result.get("profile_candidate_index", -1)),
        "sarma_profile_candidate_count": int(fit_result.get("profile_candidate_count", -1)),
        "sarma_final_lmbd": json_array(fit_result.get("lmbd")),
        "sarma_final_eta": json_array(fit_result.get("eta")),
        "sarma_loss": float(fit_result.get("Loss", np.nan)),
        "sarma_iter_no": int(fit_result.get("iter_no", -1)),
        "sarma_converged": bool(fit_result.get("converged", False)),
        "sarma_theta_diff": float(fit_result.get("theta_diff", np.nan)),
        "sarma_loss_diff": float(fit_result.get("loss_diff", np.nan)),
    }


def run_sarma_both(
    history: np.ndarray,
    order: Tuple[int, int, int],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, object]]:
    p, r, s = order
    est = SARMAEstimator(seed=args.seed, verbose=args.verbose_sarma)
    fit_start = time.perf_counter()
    if args.verbose_sarma:
        est.fit(history, p, r, s, n_jobs_profiling=args.sarma_profile_jobs)
    else:
        with open(os.devnull, "w", encoding="utf-8") as null_fh, redirect_stdout(null_fh), redirect_stderr(null_fh):
            est.fit(history, p, r, s, n_jobs_profiling=args.sarma_profile_jobs)
    fit_seconds = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    mle_forecast = np.asarray(est.predict(history), dtype=float)[0]
    pred_seconds_mle = time.perf_counter() - pred_start

    ls_fit = est._history["LS_result"]
    mle_fit = est._history["MLE_result"]
    est._fitted.G = np.asarray(ls_fit["G"])
    est._fitted.lmbd = np.asarray(ls_fit["lmbd"])
    est._fitted.eta = np.asarray(ls_fit["eta"])
    est._fitted.A = np.asarray(ls_fit["A"])

    pred_start = time.perf_counter()
    ls_forecast = np.asarray(est.predict(history), dtype=float)[0]
    pred_seconds_ls = time.perf_counter() - pred_start

    return {
        "MLE": {
            "forecast": mle_forecast,
            "fit_seconds": fit_seconds,
            "predict_seconds": pred_seconds_mle,
            "diagnostics": sarma_diag_from_fit("MLE", mle_fit),
        },
        "LS": {
            "forecast": ls_forecast,
            "fit_seconds": fit_seconds,
            "predict_seconds": pred_seconds_ls,
            "diagnostics": sarma_diag_from_fit("LS", ls_fit),
        },
    }


def run_var_ls(history: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    fit_start = time.perf_counter()
    model = VAR(history)
    results = model.fit(args.var_lag)
    fit_seconds = time.perf_counter() - fit_start
    pred_start = time.perf_counter()
    forecast = np.asarray(results.forecast(history[-args.var_lag :], steps=1), dtype=float)[0]
    predict_seconds = time.perf_counter() - pred_start
    return {
        "forecast": forecast,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "diagnostics": {},
    }


def _threadpool_scope():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


def run_varma_both(history: np.ndarray, args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    with _threadpool_scope():
        fit_iols_start = time.perf_counter()
        a_iols, b_iols, u_iols, info = varma11_iols_fit_no_c(
            history,
            max_iter=args.iols_max_iter,
            tol=args.iols_tol,
            burn_in=args.iols_burn_in,
            ridge=args.iols_ridge,
        )
        fit_iols_seconds = time.perf_counter() - fit_iols_start

        pred_iols_start = time.perf_counter()
        y_iols = np.asarray(varma11_iols_forecast1(history, a_iols, b_iols, u_iols), dtype=float)
        pred_iols_seconds = time.perf_counter() - pred_iols_start

        fit_mle_start = time.perf_counter()
        y_mle = varmax_fit_forecast_with_iols_start(
            history,
            a_iols,
            b_iols,
            steps=args.horizon,
            maxiter=args.maxiter_mle,
        )
        fit_mle_seconds = time.perf_counter() - fit_mle_start

    iols_diag = {
        "varma_iols_converged": bool(info.get("converged", False)),
        "varma_iols_n_iter": int(info.get("n_iter", -1)),
        "varma_iols_final_resid_diff": float(info.get("final_resid_diff", np.nan)),
    }
    return {
        "IOLS": {
            "forecast": y_iols,
            "fit_seconds": fit_iols_seconds,
            "predict_seconds": pred_iols_seconds,
            "diagnostics": iols_diag,
        },
        "MLE": {
            "forecast": np.asarray(y_mle, dtype=float),
            "fit_seconds": fit_iols_seconds + fit_mle_seconds,
            "predict_seconds": 0.0,
            "diagnostics": iols_diag,
        },
    }


def run_varma_parallel(
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    origins = args.origins
    assert origins is not None
    n_dim = y_train.shape[1]

    out_forecasts = {
        "VARMA_IOLS": np.full((origins, n_dim), np.nan, dtype=float),
        "VARMA_MLE": np.full((origins, n_dim), np.nan, dtype=float),
    }
    out_rows: List[Dict[str, object]] = []

    def _task(origin: int) -> Dict[str, object]:
        history = rolling_history(y_train, y_test, origin, args.rolling_method)
        try:
            payload = run_varma_both(history, args)
            return {"origin": origin, "status": "ok", "payload": payload}
        except Exception:
            return {
                "origin": origin,
                "status": "failed",
                "error_message": traceback.format_exc(limit=3).strip(),
            }

    if args.varma_n_jobs == 1:
        task_outputs = [_task(origin) for origin in range(origins)]
    else:
        task_outputs = Parallel(n_jobs=args.varma_n_jobs, backend="loky", verbose=0)(
            delayed(_task)(origin) for origin in range(origins)
        )

    for out in task_outputs:
        origin = int(out["origin"])
        train_end = y_train.shape[0] + origin - 1
        target = y_test[origin]
        if out["status"] == "ok":
            payload = out["payload"]
            for method, key in [("VARMA_IOLS", "IOLS"), ("VARMA_MLE", "MLE")]:
                if method not in args.methods:
                    continue
                result = attach_origin_metadata(method, payload[key], origin, train_end, origin)
                out_forecasts[method][origin] = result.forecast
                out_rows.append(result_row(result, target))
        else:
            logger.warning("VARMA failed at origin %d", origin)
            for method in ["VARMA_IOLS", "VARMA_MLE"]:
                if method not in args.methods:
                    continue
                failed = failed_result(method, origin, train_end, origin, n_dim)
                failed.error_message = str(out.get("error_message", failed.error_message))
                out_rows.append(result_row(failed, target))

    return out_forecasts, out_rows


def attach_origin_metadata(
    method: str,
    payload: Dict[str, object],
    origin: int,
    train_end: int,
    target_index: int,
) -> MethodRunResult:
    return MethodRunResult(
        method=method,
        origin=origin,
        train_end=train_end,
        target_index=target_index,
        status="ok",
        error_message="",
        fit_seconds=float(payload["fit_seconds"]),
        predict_seconds=float(payload["predict_seconds"]),
        forecast=np.asarray(payload["forecast"], dtype=float),
        diagnostics=dict(payload.get("diagnostics", {})),
    )


def failed_result(method: str, origin: int, train_end: int, target_index: int, n_dim: int) -> MethodRunResult:
    return MethodRunResult(
        method=method,
        origin=origin,
        train_end=train_end,
        target_index=target_index,
        status="failed",
        error_message=traceback.format_exc(limit=3).strip(),
        fit_seconds=np.nan,
        predict_seconds=np.nan,
        forecast=np.full(n_dim, np.nan, dtype=float),
        diagnostics={},
    )


def result_row(result: MethodRunResult, target: np.ndarray) -> Dict[str, object]:
    error = result.forecast - target if np.isfinite(result.forecast).all() else np.full_like(target, np.nan)
    mse_t = float(np.mean(error ** 2)) if np.isfinite(error).all() else np.nan
    mae_t = float(np.mean(np.abs(error))) if np.isfinite(error).all() else np.nan
    row = {
        "method": result.method,
        "origin": result.origin,
        "train_end": result.train_end,
        "target_index": result.target_index,
        "status": result.status,
        "error_message": result.error_message,
        "fit_seconds": result.fit_seconds,
        "predict_seconds": result.predict_seconds,
        "mse_t": mse_t,
        "mae_t": mae_t,
    }
    row.update(result.diagnostics)
    return row


def evaluate_methods(
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    logger: logging.Logger,
    sarma_order: Tuple[int, int, int],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    n_dim = y_train.shape[1]
    origins = args.origins
    assert origins is not None

    forecasts = {
        method: np.full((origins, n_dim), np.nan, dtype=float)
        for method in args.methods
    }
    rows: List[Dict[str, object]] = []

    for origin in range(origins):
        history = rolling_history(y_train, y_test, origin, args.rolling_method)
        target = y_test[origin]
        train_end = y_train.shape[0] + origin - 1
        logger.info(
            "Origin %d/%d | history shape=%s | target row=%d",
            origin + 1,
            origins,
            history.shape,
            y_train.shape[0] + origin,
        )

        if "SARMA_LS" in args.methods or "SARMA_MLE" in args.methods:
            try:
                sarma = run_sarma_both(history, sarma_order, args)
                if args.sarma_label_mode == "notebook_preserved":
                    mapping = {
                        "SARMA_LS": sarma["MLE"],
                        "SARMA_MLE": sarma["LS"],
                    }
                else:
                    mapping = {
                        "SARMA_LS": sarma["LS"],
                        "SARMA_MLE": sarma["MLE"],
                    }
                for method in ["SARMA_LS", "SARMA_MLE"]:
                    if method not in args.methods:
                        continue
                    result = attach_origin_metadata(method, mapping[method], origin, train_end, origin)
                    forecasts[method][origin] = result.forecast
                    rows.append(result_row(result, target))
            except Exception:
                logger.exception("SARMA failed at origin %d", origin)
                for method in ["SARMA_LS", "SARMA_MLE"]:
                    if method in args.methods:
                        rows.append(result_row(failed_result(method, origin, train_end, origin, n_dim), target))

        if "VAR_LS" in args.methods:
            try:
                var_payload = run_var_ls(history, args)
                result = attach_origin_metadata("VAR_LS", var_payload, origin, train_end, origin)
                forecasts["VAR_LS"][origin] = result.forecast
                rows.append(result_row(result, target))
            except Exception:
                logger.exception("VAR_LS failed at origin %d", origin)
                rows.append(result_row(failed_result("VAR_LS", origin, train_end, origin, n_dim), target))

    if "VARMA_IOLS" in args.methods or "VARMA_MLE" in args.methods:
        logger.info("Running VARMA tasks in parallel with n_jobs=%d", args.varma_n_jobs)
        varma_forecasts, varma_rows = run_varma_parallel(y_train, y_test, args, logger)
        for method in ["VARMA_IOLS", "VARMA_MLE"]:
            if method in args.methods:
                forecasts[method] = varma_forecasts[method]
        rows.extend(varma_rows)

    return forecasts, pd.DataFrame(rows)


def winner_counts(per_step_errors: pd.DataFrame, metric_col: str) -> Counter:
    winners: List[str] = []
    for _, group in per_step_errors.groupby("origin", sort=True):
        valid = group[np.isfinite(group[metric_col])]
        if valid.empty:
            continue
        winners.append(valid.loc[valid[metric_col].idxmin(), "method"])
    return Counter(winners)


def summarize_metrics(
    forecasts: Dict[str, np.ndarray],
    y_true: np.ndarray,
    per_step_errors: pd.DataFrame,
    methods: List[str],
) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    mse_winners = winner_counts(per_step_errors, "mse_t")
    mae_winners = winner_counts(per_step_errors, "mae_t")

    for method in methods:
        preds = forecasts[method]
        mask = np.isfinite(preds).all(axis=1)
        method_rows = per_step_errors[per_step_errors["method"] == method]
        if mask.any():
            err = preds[mask] - y_true[mask]
            mse = float(np.mean(((err) ** 2).ravel(order="F")))
            mae = float(np.mean(np.abs(err).ravel(order="F")))
            # Notebook cell 29 logic:
            # np.sqrt(np.mean(err^2)) then another sqrt after DataFrame mean.
            rmsfe = float(np.sqrt(np.sqrt(np.mean(err ** 2))))
            mafe = float(np.mean(np.linalg.norm(err, ord=1, axis=1)))
        else:
            mse = np.nan
            mae = np.nan
            rmsfe = np.nan
            mafe = np.nan

        summary_rows.append(
            {
                "method": method,
                "origins_attempted": int(len(method_rows)),
                "origins_successful": int(mask.sum()),
                "origins_failed": int((~mask).sum()),
                "MSE": mse,
                "MAE": mae,
                "RMSFE": rmsfe,
                "MAFE": mafe,
                "win_count_mse": int(mse_winners.get(method, 0)),
                "win_count_mae": int(mae_winners.get(method, 0)),
                "fit_seconds_total": float(method_rows["fit_seconds"].fillna(0.0).sum()),
                "predict_seconds_total": float(method_rows["predict_seconds"].fillna(0.0).sum()),
                "fit_seconds_mean": float(method_rows["fit_seconds"].mean()),
                "predict_seconds_mean": float(method_rows["predict_seconds"].mean()),
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> int:
    args = parse_args()
    git_hash = git_commit_hash(REPO_ROOT)
    run_id = make_run_id(args, git_hash)
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(run_dir)
    logger.info("Starting empirical 6D notebook-style run: %s", run_id)

    df_std, y, data_meta = load_and_transform_data(args.data_path)
    if args.horizon != 1:
        raise ValueError("This runner currently supports horizon=1 only.")
    if args.test_size < 1 or args.test_size >= y.shape[0]:
        raise ValueError("test_size must be between 1 and T-1.")
    if args.origins is None:
        args.origins = args.test_size
    if args.origins < 1 or args.origins > args.test_size:
        raise ValueError("origins must be between 1 and test_size.")

    y_train = y[: -args.test_size]
    y_test = y[-args.test_size :]
    logger.info("Loaded dataset %s; transformed shape=(%d, %d)", args.data_path, y.shape[0], y.shape[1])
    logger.info("Methods: %s", ", ".join(args.methods))
    logger.info(
        "Rolling setup: train_size=%d, test_size=%d, origins=%d, horizon=%d, method=%s",
        y_train.shape[0],
        args.test_size,
        args.origins,
        args.horizon,
        args.rolling_method,
    )

    bic_selection: Optional[Dict[str, object]] = None
    if args.order_source == "bic_full_sample":
        bic_selection = run_full_sample_bic(y, args, logger, run_dir)
        sarma_order = tuple(int(v) for v in bic_selection["selected_order"])
    else:
        sarma_order = (args.fixed_sarma_p, args.fixed_sarma_r, args.fixed_sarma_s)
        logger.info("Using fixed SARMA order from CLI: (p,r,s)=%s", sarma_order)
    logger.info("Rolling SARMA refits will use order: (p,r,s)=%s", sarma_order)

    config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "cwd": str(REPO_ROOT),
        "data_path": str(args.data_path.resolve()),
        "data_sha256": compute_sha256(args.data_path),
        "data_shape": [int(y.shape[0]), int(y.shape[1])],
        "columns": data_meta["label_columns"],
        "raw_columns": data_meta["raw_columns"],
        "transform_codes": data_meta["transform_codes"],
        "index_start": data_meta["index_start"],
        "index_end": data_meta["index_end"],
        "methods": args.methods,
        "train_size": int(y_train.shape[0]),
        "test_size": int(args.test_size),
        "origins": int(args.origins),
        "rolling_window_size": int(y_train.shape[0]),
        "horizon": int(args.horizon),
        "rolling_method": args.rolling_method,
        "seed": int(args.seed),
        "git_commit": git_hash,
        "git_status_porcelain": git_status_porcelain(REPO_ROOT),
        "order_source": args.order_source,
        "sarma_order": list(sarma_order),
        "sarma_label_mode": args.sarma_label_mode,
        "bic_selection": bic_selection,
        "bic_jobs": args.bic_jobs,
        "bic_penalty_scale": args.bic_penalty_scale,
        "sarma_profile_jobs": args.sarma_profile_jobs,
        "var_lag": args.var_lag,
        "varma_n_jobs": args.varma_n_jobs,
        "iols_max_iter": args.iols_max_iter,
        "iols_tol": args.iols_tol,
        "iols_burn_in": args.iols_burn_in,
        "iols_ridge": args.iols_ridge,
        "maxiter_mle": args.maxiter_mle,
    }
    save_json(run_dir / "config.json", config)

    start = time.perf_counter()
    forecasts, per_step_errors = evaluate_methods(y_train, y_test, args, logger, sarma_order)
    elapsed = time.perf_counter() - start

    y_true = y_test[: args.origins]
    summary = summarize_metrics(forecasts, y_true, per_step_errors, args.methods)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)
    save_json(run_dir / "metrics_summary.json", {"results": summary.to_dict(orient="records")})
    per_step_errors.to_csv(run_dir / "per_step_errors.csv", index=False)
    sarma_diag_df = per_step_errors[per_step_errors["method"].isin(["SARMA_LS", "SARMA_MLE"])].copy()
    if not sarma_diag_df.empty:
        sarma_diag_df.to_csv(run_dir / "sarma_diagnostics.csv", index=False)
    np.savez_compressed(
        run_dir / "forecasts.npz",
        y_test=y_true,
        **{method: forecasts[method] for method in args.methods},
    )

    logger.info("Completed run in %.2f seconds", elapsed)
    for row in summary.to_dict(orient="records"):
        logger.info(
            "%s | MSE=%.6f | MAE=%.6f | RMSFE=%.6f | MAFE=%.6f | success=%d/%d",
            row["method"],
            row["MSE"],
            row["MAE"],
            row["RMSFE"],
            row["MAFE"],
            row["origins_successful"],
            row["origins_attempted"],
        )
    logger.info("Artifacts written to %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
