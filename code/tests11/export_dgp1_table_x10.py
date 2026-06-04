"""
Export DGP1 compact tables with bias/ESD/ASD scaled by 10 instead of 100.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from export_complete_dgp1_tables import compact_to_tex


STAT_SUFFIXES = ("_bias", "_esd", "_asd")


def rescale_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stat_cols = [col for col in out.columns if col.endswith(STAT_SUFFIXES)]
    out[stat_cols] = out[stat_cols].apply(pd.to_numeric, errors="coerce") / 10.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create x10-scaled DGP1 compact tables.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", type=str, default="qmle,ls")
    parser.add_argument("--suffix", type=str, default="x10")
    args = parser.parse_args()

    output_dir = args.input_dir if args.output_dir is None else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [token.strip().lower() for token in args.methods.split(",") if token.strip()]

    for method in methods:
        input_csv = args.input_dir / f"DGP1_table_{method}_compact.csv"
        df = pd.read_csv(input_csv)
        scaled = rescale_compact_table(df)

        csv_path = output_dir / f"DGP1_table_{method}_compact_{args.suffix}.csv"
        tex_path = output_dir / f"DGP1_table_{method}_compact_{args.suffix}.tex"
        scaled.to_csv(csv_path, index=False)

        method_label = "QMLE" if method == "qmle" else "LSE"
        caption = rf"Biases ($\times 10$), ESDs ($\times 10$) and ASDs ($\times 10$) of the {method_label} for DGP1."
        tex = compact_to_tex(scaled, method_label, caption, f"tab_DGP1_{method}_{args.suffix}")
        tex_path.write_text(tex, encoding="utf-8")
        print(f"Wrote {csv_path}")
        print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
