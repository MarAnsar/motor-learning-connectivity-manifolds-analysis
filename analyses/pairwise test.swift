#!/usr/bin/env python3
"""
Pairwise epoch comparisons (paired t-tests) restricted to FDR-significant ROIs.

This script:
1) Reads rmANOVA results (per ROI) and selects ROIs that are significant
   after FDR for a chosen effect (e.g., "Day", "Epoch", or "Day×Epoch").
2) For each input data file (long format with Subject-level eccentricity),
   runs paired t-tests between specified epoch pairs, *only* for the significant ROIs.
3) Applies FDR correction to the resulting p-values per file.

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# ---------------------------
# Utilities
# ---------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Read TSV or CSV based on file extension."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def get_significant_rois(anova_path: Path, effect: str, alpha: float) -> List[str]:
    """
    Load rmANOVA results and return ROIs with p-FDR <= alpha for the chosen effect.
    """
    df = read_table(anova_path)
    required_cols = {"Region", "Effect"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"ANOVA file must have columns {required_cols}; found {df.columns.tolist()}")

    # prefer 'p-FDR', fall back to 'p-unc' if needed
    pcol = "p-FDR" if "p-FDR" in df.columns else ("p-unc" if "p-unc" in df.columns else None)
    if pcol is None:
        raise ValueError("ANOVA file must contain 'p-FDR' or 'p-unc' column for p-values.")

    # Standardize effect naming like "Day×Epoch"
    eff_norm = effect.replace("*", "×").replace(" x ", "×").replace("X", "×")
    sub = df[df["Effect"].astype(str).str.replace("*", "×") == eff_norm].copy()
    if sub.empty:
        raise ValueError(f"No rows found for Effect == '{eff_norm}' in ANOVA file.")

    sig = sub.loc[sub[pcol] <= alpha, "Region"].astype(str).unique().tolist()
    return sig


def paired_ttests_for_file(
    data_path: Path,
    sig_rois: List[str],
    epoch_pairs: List[Tuple[str, str]],
    epoch_col: str = "Epoch",
    subj_col: str = "Subject",
    region_col: str = "Region",
    value_col: str = "Eccentricity",
    min_n: int = 2,
) -> pd.DataFrame:
    """
    Run paired t-tests between epoch pairs for significant ROIs only.
    Returns a tidy DataFrame with columns:
      Region, Epochs, N, T, P, P_FDR (filled later), file
    """
    df = read_table(data_path)

    # Validate columns
    req = {region_col, subj_col, epoch_col, value_col}
    if not req.issubset(df.columns):
        raise ValueError(f"{data_path.name}: required columns {req} not all present. Found {df.columns.tolist()}")

    # Filter to significant ROIs
    df = df[df[region_col].astype(str).isin(sig_rois)].copy()
    if df.empty:
        return pd.DataFrame(columns=["Region", "Epochs", "N", "T", "P", "P_FDR", "file"])

    out_rows = []
    rois = df[region_col].astype(str).unique()

    for roi in rois:
        d_roi = df[df[region_col].astype(str) == roi]

        for ep1, ep2 in epoch_pairs:
            d_pair = d_roi[d_roi[epoch_col].isin([ep1, ep2])]
            wide = d_pair.pivot(index=subj_col, columns=epoch_col, values=value_col)
            # Drop subjects with missing values in either epoch
            wide = wide.dropna(subset=[ep1, ep2], how="any")

            if wide.shape[0] >= min_n:
                t_stat, p_val = ttest_rel(wide[ep1], wide[ep2])
                out_rows.append({
                    "Region": roi,
                    "Epochs": f"{ep1}-{ep2}",
                    "N": int(wide.shape[0]),
                    "T": float(t_stat),
                    "P": float(p_val),
                    "P_FDR": np.nan,
                    "file": data_path.name,
                })

    return pd.DataFrame(out_rows)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Paired epoch comparisons restricted to FDR-significant ROIs from rmANOVA."
    )
    ap.add_argument("--in-dir", required=True, help="Directory of input TSV/CSV files (long format).")
    ap.add_argument("--anova-results", required=True, help="rmANOVA results TSV/CSV with Region, Effect, p-FDR (or p-unc).")
    ap.add_argument("--effect", default="Epoch",
                    help="Effect name to select significant ROIs (e.g., 'Day', 'Epoch', 'Day×Epoch').")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold on FDR/p-values (default: 0.05).")
    ap.add_argument("--epochs", nargs="+", default=["Base", "Early", "Late", "Washout"],
                    help="Available epoch names (used only for validation).")
    ap.add_argument(
        "--pairs", nargs="+",
        default=["Base", "Early", "Early", "Late", "Late", "Washout", "Base", "Late", "Base", "Washout", "Early", "Washout"],
        help=("Flattened list of epoch pairs, e.g.: "
              "--pairs Base Early Early Late Late Washout Base Late Base Washout Early Washout")
    )
    ap.add_argument("--out-dir", required=True, help="Directory to write pairwise results.")
    # Optional column names
    ap.add_argument("--region-col", default="Region")
    ap.add_argument("--subject-col", default="Subject")
    ap.add_argument("--epoch-col", default="Epoch")
    ap.add_argument("--value-col", default="Eccentricity")
    return ap.parse_args()


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate pairs
    if len(args.pairs) % 2 != 0:
        raise ValueError("--pairs must contain an even number of epoch names (flattened pairs).")
    epoch_pairs = [(args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)]

    # Load significant ROIs from rmANOVA results
    sig_rois = get_significant_rois(Path(args.anova_results), effect=args.effect, alpha=args.alpha)
    if not sig_rois:
        print("[info] No significant ROIs found for the chosen effect; nothing to test.")
        return

    # Iterate over input files
    data_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".tsv", ".csv")])
    if not data_files:
        raise FileNotFoundError(f"No .tsv/.csv files found in: {in_dir}")

    for f in data_files:
        try:
            res = paired_ttests_for_file(
                data_path=f,
                sig_rois=sig_rois,
                epoch_pairs=epoch_pairs,
                epoch_col=args.epoch_col,
                subj_col=args.subject_col,
                region_col=args.region_col,
                value_col=args.value_col,
            )
            if res.empty:
                print(f"[info] {f.name}: no tests run (no overlapping significant ROIs or insufficient subjects).")
                continue

            # FDR across all tests in this file
            _, p_fdr, _, _ = multipletests(res["P"].to_numpy(), alpha=args.alpha, method="fdr_bh")
            res["P_FDR"] = p_fdr

            out_path = out_dir / f"pairwise_{f.stem}.tsv"
            res.to_csv(out_path, sep="\t", index=False)
            print(f"[ok] Saved: {out_path}")
        except Exception as e:
            print(f"[warn] Skipping {f.name}: {e}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
