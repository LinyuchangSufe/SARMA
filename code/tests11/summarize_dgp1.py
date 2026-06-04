"""
Summarize DGP1 results into compact, diagnostic-long, and supplement-full tables.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_SPECS = [
    ("normal", "identity", "normal_a0"),
    ("t5", "identity", "t5_a0"),
    ("normal", "compound", "normal_a05"),
    ("t5", "compound", "t5_a05"),
]

COMPACT_MAIN = [
    ("$\\lambda$", "lmbd_1"),
    ("$\\gamma$", "eta_1_gamma"),
    ("$\\phi$", "eta_1_phi"),
    ("$g_{1,11}$", "G1_r1_c1"),
    ("$g_{2,11}$", "G2_r1_c1"),
    ("$g_{3,11}$", "G3_r1_c1"),
    ("$g_{4,11}$", "G4_r1_c1"),
    ("$\\sigma_{11}$", "Sigma_r1_c1"),
    ("$\\sigma_{21}$", "Sigma_r2_c1"),
    ("$\\sigma_{31}$", "Sigma_r3_c1"),
]


def parse_csv_list(text: str, cast):
    vals = [token.strip() for token in str(text).split(",")]
    return [cast(v) for v in vals if v]


def _sort_key(param_col: str):
    m = re.match(r"^lmbd_(\d+)$", param_col)
    if m:
        return (0, int(m.group(1)), 0, 0)
    m = re.match(r"^eta_(\d+)_(gamma|phi)$", param_col)
    if m:
        part = 0 if m.group(2) == "gamma" else 1
        return (1, int(m.group(1)), part, 0)
    m = re.match(r"^G(\d+)_r(\d+)_c(\d+)$", param_col)
    if m:
        return (2, int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"^Sigma_r(\d+)_c(\d+)$", param_col)
    if m:
        return (3, int(m.group(1)), int(m.group(2)), 0)
    return (9, param_col, 0, 0)


def parse_param_meta(param_col: str, override_label: str | None = None):
    if override_label is not None:
        label = override_label
    else:
        label = param_col

    m = re.match(r"^lmbd_(\d+)$", param_col)
    if m:
        idx = int(m.group(1))
        return {
            "param_label": override_label or f"$\\lambda_{{{idx}}}$",
            "param_type": "omega",
            "block": "lambda",
            "row": idx,
            "col": np.nan,
        }

    m = re.match(r"^eta_(\d+)_(gamma|phi)$", param_col)
    if m:
        idx = int(m.group(1))
        part = m.group(2)
        part_symbol = "\\gamma" if part == "gamma" else "\\phi"
        return {
            "param_label": override_label or f"${part_symbol}_{{{idx}}}$",
            "param_type": "omega",
            "block": part,
            "row": idx,
            "col": np.nan,
        }

    m = re.match(r"^G(\d+)_r(\d+)_c(\d+)$", param_col)
    if m:
        g_block = int(m.group(1))
        row = int(m.group(2))
        col = int(m.group(3))
        return {
            "param_label": override_label or f"$g_{{{g_block},{row}{col}}}$",
            "param_type": "G",
            "block": g_block,
            "row": row,
            "col": col,
        }

    m = re.match(r"^Sigma_r(\d+)_c(\d+)$", param_col)
    if m:
        row = int(m.group(1))
        col = int(m.group(2))
        return {
            "param_label": override_label or f"$\\sigma_{{{row}{col}}}$",
            "param_type": "Sigma",
            "block": "Sigma",
            "row": row,
            "col": col,
        }

    return {
        "param_label": label,
        "param_type": "other",
        "block": "other",
        "row": np.nan,
        "col": np.nan,
    }


def read_csv_files(result_dir: Path, N: int, T: int, dist: str, sigma_type: str, method: str):
    candidates = [
        f"DGP1_N{N}_T{T}_{dist}_{sigma_type}",
        f"DGP1_T{T}_{dist}_{sigma_type}",
    ]
    for config_str in candidates:
        est_file = result_dir / f"{config_str}_{method}_estimates.csv"
        asd_file = result_dir / f"{config_str}_{method}_asd.csv"
        if est_file.exists() and asd_file.exists():
            return pd.read_csv(est_file), pd.read_csv(asd_file), config_str
    return None, None, None


def compute_stats(df_est: pd.DataFrame, df_asd: pd.DataFrame, param_col: str):
    if param_col not in df_est.columns or param_col not in df_asd.columns:
        return {
            "true": np.nan,
            "mean_est": np.nan,
            "bias": np.nan,
            "esd": np.nan,
            "asd": np.nan,
            "esd_asd_ratio": np.nan,
            "n_est": 0,
            "n_asd": 0,
        }

    true_val = float(df_est.iloc[0][param_col])
    est_vals = df_est.iloc[1:][param_col].dropna().astype(float).values
    asd_vals = df_asd.iloc[1:][param_col].dropna().astype(float).values

    if len(est_vals) == 0:
        return {
            "true": true_val,
            "mean_est": np.nan,
            "bias": np.nan,
            "esd": np.nan,
            "asd": np.nan,
            "esd_asd_ratio": np.nan,
            "n_est": 0,
            "n_asd": int(len(asd_vals)),
        }

    mean_est = float(np.mean(est_vals))
    bias = (mean_est - true_val) * 100.0
    esd = np.std(est_vals, ddof=1) * 100.0 if len(est_vals) > 1 else np.nan
    asd = np.mean(asd_vals) * 100.0 if len(asd_vals) > 0 else np.nan
    ratio = esd / asd if np.isfinite(esd) and np.isfinite(asd) and abs(asd) > 1e-12 else np.nan
    return {
        "true": true_val,
        "mean_est": mean_est,
        "bias": float(bias),
        "esd": float(esd) if np.isfinite(esd) else np.nan,
        "asd": float(asd) if np.isfinite(asd) else np.nan,
        "esd_asd_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "n_est": int(len(est_vals)),
        "n_asd": int(len(asd_vals)),
    }


def discover_full_params(result_dir: Path, N: int, t_values: list[int], method: str):
    found = set()
    for T in t_values:
        for dist, sigma_type, _ in PANEL_SPECS:
            df_est, _, _ = read_csv_files(result_dir, N, T, dist, sigma_type, method)
            if df_est is None:
                continue
            for col in df_est.columns:
                if col == "rep":
                    continue
                found.add(col)
    return sorted(found, key=_sort_key)


def build_tables(result_dir: Path, N: int, t_values: list[int], method: str):
    compact_map = {param_col: label for label, param_col in COMPACT_MAIN}
    compact_params = [param_col for _, param_col in COMPACT_MAIN]
    full_params = discover_full_params(result_dir, N, t_values, method)
    full_map = {param_col: None for param_col in full_params}

    cache = {}

    def load_once(T: int, dist: str, sigma_type: str):
        key = (T, dist, sigma_type)
        if key not in cache:
            cache[key] = read_csv_files(result_dir, N, T, dist, sigma_type, method)
        return cache[key]

    def build_for_params(param_cols: list[str], label_map: dict[str, str | None], include_ratio: bool):
        rows_wide = []
        rows_long = []
        for param_col in param_cols:
            meta = parse_param_meta(param_col, label_map.get(param_col))
            for T in t_values:
                wide_row = {
                    "param_label": meta["param_label"],
                    "param_col": param_col,
                    "param_type": meta["param_type"],
                    "block": meta["block"],
                    "row": meta["row"],
                    "col": meta["col"],
                    "T": int(T),
                }
                for dist, sigma_type, panel_key in PANEL_SPECS:
                    df_est, df_asd, config_used = load_once(T, dist, sigma_type)
                    stats = {
                        "true": np.nan,
                        "mean_est": np.nan,
                        "bias": np.nan,
                        "esd": np.nan,
                        "asd": np.nan,
                        "esd_asd_ratio": np.nan,
                        "n_est": 0,
                        "n_asd": 0,
                    }
                    if df_est is not None and df_asd is not None:
                        stats = compute_stats(df_est, df_asd, param_col)

                    wide_row[f"{panel_key}_bias"] = stats["bias"]
                    wide_row[f"{panel_key}_esd"] = stats["esd"]
                    wide_row[f"{panel_key}_asd"] = stats["asd"]

                    long_row = {
                        "param_label": meta["param_label"],
                        "param_col": param_col,
                        "param_type": meta["param_type"],
                        "block": meta["block"],
                        "row": meta["row"],
                        "col": meta["col"],
                        "T": int(T),
                        "dist": dist,
                        "sigma_type": sigma_type,
                        "panel_key": panel_key,
                        "method": method,
                        "config_used": config_used,
                        "true": stats["true"],
                        "mean_est": stats["mean_est"],
                        "bias": stats["bias"],
                        "esd": stats["esd"],
                        "asd": stats["asd"],
                        "n_est": stats["n_est"],
                        "n_asd": stats["n_asd"],
                    }
                    if include_ratio:
                        long_row["esd_asd_ratio"] = stats["esd_asd_ratio"]
                    rows_long.append(long_row)

                rows_wide.append(wide_row)
        wide_df = pd.DataFrame(rows_wide)
        long_df = pd.DataFrame(rows_long)
        return wide_df, long_df

    compact_wide, compact_long = build_for_params(compact_params, compact_map, include_ratio=True)
    full_wide, full_long = build_for_params(full_params, full_map, include_ratio=True)
    return compact_wide, compact_long, full_wide, full_long


def main():
    parser = argparse.ArgumentParser(description="Summarize DGP1 into compact/long/supplement tables.")
    parser.add_argument("--result-dir", type=Path, default=Path("code/tests11/results/Exp1"))
    parser.add_argument("--output-dir", type=Path, default=Path("code/tests11/results/Exp1"))
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--T-values", type=str, default="500,1000")
    parser.add_argument("--method", type=str, default="QMLE", choices=["QMLE", "LS"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_values = parse_csv_list(args.T_values, int)
    method_tag = args.method.lower()

    compact_wide, compact_long, full_wide, full_long = build_tables(
        result_dir=args.result_dir, N=args.N, t_values=t_values, method=args.method
    )

    compact_csv = args.output_dir / f"DGP1_table_{method_tag}_compact.csv"
    long_csv = args.output_dir / f"DGP1_table_{method_tag}_long.csv"
    supp_long_csv = args.output_dir / f"DGP1_table_{method_tag}_supplement_full_long.csv"
    supp_wide_csv = args.output_dir / f"DGP1_table_{method_tag}_supplement_full_wide.csv"

    compact_wide.to_csv(compact_csv, index=False)
    compact_long.to_csv(long_csv, index=False)
    full_long.to_csv(supp_long_csv, index=False)
    full_wide.to_csv(supp_wide_csv, index=False)

    print(f"Wrote compact summary: {compact_csv}")
    print(f"Wrote diagnostic long: {long_csv}")
    print(f"Wrote supplement full long: {supp_long_csv}")
    print(f"Wrote supplement full wide: {supp_wide_csv}")


if __name__ == "__main__":
    main()
