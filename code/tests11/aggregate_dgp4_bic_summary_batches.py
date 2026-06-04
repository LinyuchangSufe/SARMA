"""
Aggregate DGP4 BIC summary batches into 500-rep paper-style tables.

The runner writes one ``*_BIC_summary.csv`` per simulation case and per batch.
This script sums the under/exact/over counts across multiple batch directories
before converting them to percentages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PANELS = [
    ("normal", "identity", "normal_a0"),
    ("t5", "identity", "t5_a0"),
    ("normal", "compound", "normal_a05"),
    ("t5", "compound", "t5_a05"),
]


def parse_csv_list(text: str, cast):
    return [cast(token.strip()) for token in str(text).split(",") if token.strip()]


def collect_summary_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_dir in input_dirs:
        files.extend(sorted(input_dir.rglob("*_BIC_summary.csv")))
    if not files:
        raise FileNotFoundError("No *_BIC_summary.csv files found.")
    return files


def load_grouped(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].copy()
        row["source_file"] = str(path)
        rows.append(row)
    if not rows:
        raise ValueError("All summary files were empty.")

    raw = pd.DataFrame(rows)
    for col in ["N", "T", "lam", "n_reps", "under", "exact", "over", "error", "other"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    group_cols = [
        "N",
        "T",
        "lam",
        "innovation_dist",
        "sigma_type",
        "bic_criterion",
        "bic_penalty_scale",
        "phi_design",
        "dgp_seed",
        "dgp_lambda",
    ]
    group_cols = [col for col in group_cols if col in raw.columns]
    count_cols = [col for col in ["n_reps", "under", "exact", "over", "error", "other"] if col in raw.columns]

    grouped = raw.groupby(group_cols, dropna=False, as_index=False)[count_cols].sum()
    grouped["n_batches"] = raw.groupby(group_cols, dropna=False).size().to_numpy()
    denom = grouped["n_reps"].replace(0, np.nan)
    grouped["pct_under"] = grouped["under"] * 100.0 / denom
    grouped["pct_exact"] = grouped["exact"] * 100.0 / denom
    grouped["pct_over"] = grouped["over"] * 100.0 / denom
    grouped = grouped.sort_values(["lam", "T", "innovation_dist", "sigma_type"]).reset_index(drop=True)
    return grouped


def build_wide(long_df: pd.DataFrame, N: int, t_values: list[int], lam_values: list[float]) -> pd.DataFrame:
    sub = long_df[long_df["N"].astype(int).eq(N)].copy()
    sub = sub[sub["T"].astype(int).isin(t_values)]
    sub = sub[sub["lam"].round(6).isin([round(v, 6) for v in lam_values])]

    lookup = {}
    for _, row in sub.iterrows():
        key = (round(float(row["lam"]), 6), int(row["T"]), row["innovation_dist"], row["sigma_type"])
        lookup[key] = (row["pct_under"], row["pct_exact"], row["pct_over"], row["n_reps"])

    rows = []
    for lam in lam_values:
        for T in t_values:
            out = {"lam": float(lam), "T": int(T)}
            n_reps_seen = []
            for dist, sigma_type, panel_key in PANELS:
                under, exact, over, n_reps = lookup.get(
                    (round(float(lam), 6), int(T), dist, sigma_type),
                    (np.nan, np.nan, np.nan, np.nan),
                )
                out[f"{panel_key}_under"] = under
                out[f"{panel_key}_exact"] = exact
                out[f"{panel_key}_over"] = over
                n_reps_seen.append(n_reps)
            out["n_reps_min"] = np.nanmin(n_reps_seen) if not all(pd.isna(v) for v in n_reps_seen) else np.nan
            out["n_reps_max"] = np.nanmax(n_reps_seen) if not all(pd.isna(v) for v in n_reps_seen) else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def fmt(x) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.1f}"


def to_latex(wide_df: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{rrrrrrrrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{Normal ($a=0$)} & \multicolumn{3}{c}{$t_5$ ($a=0$)} & \multicolumn{3}{c}{Normal ($a=0.5$)} & \multicolumn{3}{c}{$t_5$ ($a=0.5$)} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}",
        r"\multicolumn{1}{c}{$\lambda$}& \multicolumn{1}{c}{$T$} & \multicolumn{1}{l}{Under} & \multicolumn{1}{l}{Exact} & \multicolumn{1}{l}{Over} & \multicolumn{1}{l}{Under} & \multicolumn{1}{l}{Exact} & \multicolumn{1}{l}{Over} & \multicolumn{1}{l}{Under} & \multicolumn{1}{l}{Exact} & \multicolumn{1}{l}{Over} & \multicolumn{1}{l}{Under} & \multicolumn{1}{l}{Exact} & \multicolumn{1}{l}{Over} \\",
        r"\midrule",
    ]
    for _, row in wide_df.sort_values(["lam", "T"]).iterrows():
        vals = [f"{float(row['lam']):.1f}", f"{int(row['T'])}"]
        for _, _, panel_key in PANELS:
            vals.extend(
                [
                    fmt(row[f"{panel_key}_under"]),
                    fmt(row[f"{panel_key}_exact"]),
                    fmt(row[f"{panel_key}_over"]),
                ]
            )
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate batched DGP4 BIC summaries.")
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--T-values", type=str, default="500,750,1000")
    parser.add_argument("--lam-values", type=str, default="0.6,0.7,0.8,0.9")
    parser.add_argument("--prefix", type=str, default="DGP4_BIC_strong_symmetric_n500")
    parser.add_argument(
        "--caption",
        type=str,
        default="Percentages of underfitted, correctly selected and overfitted cases under DGP4.",
    )
    parser.add_argument("--label", type=str, default="tab_BIC")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = collect_summary_files(args.input_dir)
    long_df = load_grouped(files)
    wide_df = build_wide(
        long_df,
        N=args.N,
        t_values=parse_csv_list(args.T_values, int),
        lam_values=parse_csv_list(args.lam_values, float),
    )

    long_path = args.output_dir / f"{args.prefix}_long_N{args.N}.csv"
    wide_path = args.output_dir / f"{args.prefix}_table_N{args.N}.csv"
    tex_path = args.output_dir / f"{args.prefix}_table_N{args.N}.tex"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    tex_path.write_text(to_latex(wide_df, args.caption, args.label), encoding="utf-8")

    print(f"Read {len(files)} summary files")
    print(f"Wrote {long_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
