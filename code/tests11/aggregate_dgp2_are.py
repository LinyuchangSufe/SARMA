"""
Aggregate DGP2 ARE outputs into paper-style CSV/LaTeX tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def parse_csv_list(text: str, cast):
    vals = [token.strip() for token in str(text).split(",")]
    return [cast(v) for v in vals if v]


def _design_norm(design_raw: str) -> str:
    if design_raw in {"2a", "a"}:
        return "2a"
    if design_raw in {"2b", "b"}:
        return "2b"
    raise ValueError(f"Unknown design token: {design_raw}")


def parse_name(path: Path):
    name = path.name
    suffix = "_ARE_per_rep.csv"
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    parts = stem.split("_")
    # Supported:
    # old: DGP22a_T5000_a0.4_a0.0_normal
    # new: DGP22a_N6_T5000_a0.4_a0.0_normal
    if len(parts) not in {5, 6}:
        return None
    token0 = parts[0]
    if not token0.startswith("DGP2"):
        return None
    design_token = token0.replace("DGP2", "", 1)
    design = _design_norm(design_token)
    idx = 1
    N = None
    if parts[idx].startswith("N"):
        N = int(parts[idx][1:])
        idx += 1
    if idx + 3 >= len(parts):
        return None
    if not parts[idx].startswith("T"):
        return None
    T = int(parts[idx][1:])
    param_token = parts[idx + 1]
    a_token = parts[idx + 2]
    dist = parts[idx + 3]
    if not param_token or param_token[0] not in {"a", "b"}:
        return None
    if not a_token.startswith("a"):
        return None
    try:
        param_val = float(param_token[1:])
        a_val = float(a_token[1:])
    except ValueError:
        return None
    if dist not in {"normal", "t5"}:
        return None
    return {
        "N": int(N) if N is not None else 6,
        "T": int(T),
        "design": design,
        "param_val": param_val,
        "a": a_val,
        "innovation_dist": dist,
    }


def _mean_are(df: pd.DataFrame, are_column: str) -> tuple[float, str]:
    column = are_column if are_column in df.columns else "ARE"
    if column not in df.columns:
        raise ValueError(f"ARE column not found. Requested {are_column!r}; available columns: {list(df.columns)}")
    if "rep" in df.columns:
        mean_rows = df[df["rep"].astype(str) == "mean"]
        if len(mean_rows) > 0:
            return float(mean_rows.iloc[0][column]), column
    return float(pd.to_numeric(df[column], errors="coerce").mean()), column


def load_long(input_dir: Path, are_column: str) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*_ARE_per_rep.csv")):
        meta = parse_name(path)
        if meta is None:
            continue
        df = pd.read_csv(path)
        mean_are, used_column = _mean_are(df, are_column)
        rows.append(
            {
                "source_file": path.name,
                "are_column": used_column,
                "N": int(meta["N"]),
                "T": int(meta["T"]),
                "design": meta["design"],
                "param_val": float(meta["param_val"]),
                "a": float(meta["a"]),
                "innovation_dist": meta["innovation_dist"],
                "ARE": mean_are,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No DGP2 ARE files matched in {input_dir}")
    out = pd.DataFrame(rows)
    out = out.sort_values(["T", "innovation_dist", "a", "design", "param_val"]).reset_index(drop=True)
    return out


def build_compact(
    long_df: pd.DataFrame,
    N: int,
    T: int,
    report_a: list[float],
    lam_values: list[float],
    gam_values: list[float],
) -> pd.DataFrame:
    sub = long_df[(long_df["N"] == N) & (long_df["T"] == T)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for N={N}, T={T}.")

    lookup = {}
    for _, row in sub.iterrows():
        key = (
            row["innovation_dist"],
            round(float(row["a"]), 6),
            row["design"],
            round(float(row["param_val"]), 6),
        )
        lookup[key] = float(row["ARE"])

    rows = []
    for dist in ["normal", "t5"]:
        dist_label = "Normal" if dist == "normal" else "$t_5$"
        for a in report_a:
            row = {"dist": dist_label, "a": float(a)}
            for lam in lam_values:
                row[f"lambda_{lam:.1f}"] = lookup.get((dist, round(float(a), 6), "2a", round(float(lam), 6)), np.nan)
            for gam in gam_values:
                row[f"gamma_{gam:.1f}"] = lookup.get((dist, round(float(a), 6), "2b", round(float(gam), 6)), np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _fmt(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.3f}"


def to_latex(compact_df: pd.DataFrame, lam_values: list[float], gam_values: list[float], caption: str, label: str) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{0.85}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{crrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"&& \multicolumn{4}{c}{$\lambda$} & \multicolumn{4}{c}{$\gamma$} \\")
    lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    lam_hdr = " & ".join([f"{v:.1f}" for v in lam_values])
    gam_hdr = " & ".join([f"{v:.1f}" for v in gam_values])
    lines.append(f"&\\multicolumn{{1}}{{c}}{{$a$}} & {lam_hdr} & {gam_hdr} \\\\")
    lines.append(r"\midrule")

    for dist_label in ["Normal", "$t_5$"]:
        sub = compact_df[compact_df["dist"] == dist_label].sort_values("a")
        first = True
        for _, row in sub.iterrows():
            dist_cell = dist_label if first else ""
            first = False
            vals = [dist_cell, f"{float(row['a']):.1f}"]
            for lam in lam_values:
                vals.append(_fmt(row[f"lambda_{lam:.1f}"]))
            for gam in gam_values:
                vals.append(_fmt(row[f"gamma_{gam:.1f}"]))
            lines.append(" & ".join(vals) + r" \\")
        if dist_label == "Normal":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Aggregate DGP2 ARE results to CSV/LaTeX.")
    parser.add_argument("--input-dir", type=Path, default=Path("code/tests11/results/Exp2"))
    parser.add_argument("--output-dir", type=Path, default=Path("code/tests11/results/Exp2"))
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--T", type=int, default=5000)
    parser.add_argument("--a-report", type=str, default="0.0,0.6,0.9")
    parser.add_argument("--lambda-values", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument("--gamma-values", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument(
        "--are-column",
        type=str,
        default="ARE_QML_over_LS",
        help="Column to aggregate. Defaults to the paper-table direction; falls back to ARE for old files.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "$\\text{ARE}(\\widehat{\\bm \\alpha}_{LS}, \\widehat{\\bm \\alpha})$ under DGP2(a) and DGP2(b)."
        ),
    )
    parser.add_argument("--label", type=str, default="tab_ARE")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_a = parse_csv_list(args.a_report, float)
    lam_values = parse_csv_list(args.lambda_values, float)
    gam_values = parse_csv_list(args.gamma_values, float)

    long_df = load_long(args.input_dir, are_column=args.are_column)
    long_out = args.output_dir / f"DGP2_ARE_long_N{args.N}.csv"
    long_df.to_csv(long_out, index=False)

    compact_df = build_compact(long_df, N=args.N, T=args.T, report_a=report_a, lam_values=lam_values, gam_values=gam_values)
    compact_out = args.output_dir / f"DGP2_ARE_table_N{args.N}.csv"
    compact_df.to_csv(compact_out, index=False)

    tex = to_latex(compact_df, lam_values=lam_values, gam_values=gam_values, caption=args.caption, label=args.label)
    tex_out = args.output_dir / f"DGP2_ARE_table_N{args.N}.tex"
    tex_out.write_text(tex, encoding="utf-8")

    print(f"Wrote long CSV: {long_out}")
    print(f"Wrote compact CSV: {compact_out}")
    print(f"Wrote LaTeX: {tex_out}")


if __name__ == "__main__":
    main()
