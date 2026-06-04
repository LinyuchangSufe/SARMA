"""
Merge split DGP1 summary tables and export compact/full CSV and TeX tables.

This is for runs split across T grids. By default, if the same T appears in
multiple inputs, later inputs override earlier ones. This lets us combine the
old T=500/1000 run with the newer T=750/1000 run while keeping T=1000 from the
newer run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PANEL_KEYS = ["normal_a0", "t5_a0", "normal_a05", "t5_a05"]
PANEL_LABELS = [r"Normal ($a=0$)", r"$t_5$ ($a=0$)", r"Normal ($a=0.5$)", r"$t_5$ ($a=0.5$)"]
STATS = ["bias", "esd", "asd"]
ID_COLS = ["param_label", "param_col", "param_type", "block", "row", "col", "T"]


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def merge_tables(paths: list[Path], t_values: list[int]) -> pd.DataFrame:
    parts = []
    for priority, path in enumerate(paths):
        df = pd.read_csv(path)
        df["_priority"] = priority
        parts.append(df)
    if not parts:
        raise ValueError("No input tables")

    merged = pd.concat(parts, ignore_index=True)
    merged = merged[merged["T"].isin(t_values)].copy()
    if merged.empty:
        raise ValueError(f"No rows remain after filtering T={t_values}")

    key = ["param_col", "T"]
    merged = merged.sort_values(key + ["_priority"]).drop_duplicates(key, keep="last")
    merged = merged.drop(columns=["_priority"])

    # Preserve the parameter order from the first input and the requested T order.
    param_order = []
    for path in paths:
        df = pd.read_csv(path)
        for param_col in df["param_col"]:
            if param_col not in param_order:
                param_order.append(param_col)
    param_rank = {param_col: i for i, param_col in enumerate(param_order)}
    t_rank = {T: i for i, T in enumerate(t_values)}
    merged["_param_rank"] = merged["param_col"].map(param_rank)
    merged["_t_rank"] = merged["T"].map(t_rank)
    merged = merged.sort_values(["_param_rank", "_t_rank"]).drop(columns=["_param_rank", "_t_rank"])
    return merged.reset_index(drop=True)


def compact_to_tex(df: pd.DataFrame, method_label: str, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\renewcommand{\arraystretch}{1}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{crrrrrrrrrrrrr}",
        r"\toprule",
        (
            r"&       & \multicolumn{3}{c}{"
            + PANEL_LABELS[0]
            + r"} & \multicolumn{3}{c}{"
            + PANEL_LABELS[1]
            + r"} & \multicolumn{3}{c}{"
            + PANEL_LABELS[2]
            + r"} & \multicolumn{3}{c}{"
            + PANEL_LABELS[3]
            + r"} \\"
        ),
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}",
        (
            r"& \multicolumn{1}{c}{$T$} & \multicolumn{1}{c}{Bias} & \multicolumn{1}{c}{ESD} & \multicolumn{1}{c}{ASD} "
            r"& \multicolumn{1}{c}{Bias} & \multicolumn{1}{c}{ESD} & \multicolumn{1}{c}{ASD} "
            r"& \multicolumn{1}{c}{Bias} & \multicolumn{1}{c}{ESD} & \multicolumn{1}{c}{ASD} "
            r"& \multicolumn{1}{c}{Bias} & \multicolumn{1}{c}{ESD} & \multicolumn{1}{c}{ASD} \\"
        ),
        r"\midrule",
    ]

    first_for_param: set[str] = set()
    for _, row in df.iterrows():
        param = str(row["param_col"])
        label_text = str(row["param_label"]) if param not in first_for_param else ""
        first_for_param.add(param)
        values = [label_text, str(int(row["T"]))]
        for panel in PANEL_KEYS:
            for stat in STATS:
                values.append(fmt(row[f"{panel}_{stat}"]))
        lines.append(" & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge split DGP1 summary tables.")
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--T-values", type=str, default="500,750,1000")
    args = parser.parse_args()

    t_values = parse_ints(args.T_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for method in ["qmle", "ls"]:
        compact_inputs = [d / f"DGP1_table_{method}_compact.csv" for d in args.input_dir]
        long_inputs = [d / f"DGP1_table_{method}_long.csv" for d in args.input_dir]
        full_long_inputs = [d / f"DGP1_table_{method}_supplement_full_long.csv" for d in args.input_dir]
        full_wide_inputs = [d / f"DGP1_table_{method}_supplement_full_wide.csv" for d in args.input_dir]
        for path in compact_inputs + long_inputs + full_long_inputs + full_wide_inputs:
            if not path.exists():
                raise FileNotFoundError(path)

        compact = merge_tables(compact_inputs, t_values)
        long = pd.concat([pd.read_csv(p).assign(_priority=i) for i, p in enumerate(long_inputs)], ignore_index=True)
        long = long[long["T"].isin(t_values)].sort_values(["param_col", "T", "dist", "sigma_type", "_priority"])
        long = long.drop_duplicates(["param_col", "T", "dist", "sigma_type"], keep="last").drop(columns=["_priority"])

        full_wide = merge_tables(full_wide_inputs, t_values)
        full_long = pd.concat([pd.read_csv(p).assign(_priority=i) for i, p in enumerate(full_long_inputs)], ignore_index=True)
        full_long = full_long[full_long["T"].isin(t_values)].sort_values(["param_col", "T", "dist", "sigma_type", "_priority"])
        full_long = full_long.drop_duplicates(["param_col", "T", "dist", "sigma_type"], keep="last").drop(columns=["_priority"])

        compact.to_csv(args.output_dir / f"DGP1_table_{method}_compact.csv", index=False)
        long.to_csv(args.output_dir / f"DGP1_table_{method}_long.csv", index=False)
        full_wide.to_csv(args.output_dir / f"DGP1_table_{method}_supplement_full_wide.csv", index=False)
        full_long.to_csv(args.output_dir / f"DGP1_table_{method}_supplement_full_long.csv", index=False)

        method_label = "QMLE" if method == "qmle" else "LSE"
        caption = rf"Biases ($\times 100$), ESDs ($\times 100$) and ASDs ($\times 100$) of the {method_label} for DGP1."
        tex = compact_to_tex(compact, method_label, caption, f"tab_DGP1_{method}")
        (args.output_dir / f"DGP1_table_{method}_compact.tex").write_text(tex, encoding="utf-8")

    print(f"Wrote complete DGP1 tables to: {args.output_dir}")


if __name__ == "__main__":
    main()
