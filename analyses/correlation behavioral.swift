#!/usr/bin/env python3
"""
Per-ROI Pearson correlation with FDR correction.

Given a long-format TSV that contains (at least) these columns:
  - ROI column (e.g., "roi")
  - X variable (e.g., "distance")
  - Y variable (e.g., "Error_deg")

…the script computes Pearson's r and p-value for each ROI, applies
Benjamini–Hochberg FDR correction across ROIs, preserves the original
ROI order as it first appears in the input, and saves a results TSV.

"""

from __future__ import annotations

import argparse
import sys
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-ROI Pearson correlation with FDR (BH) correction.")
    p.add_argument("--infile", required=True, help="Input TSV file (long format).")
    p.add_argument("--outfile", required=True, help="Output TSV file for correlation results.")
    p.add_argument("--roi-col", default="roi", help='Column name for ROI (default: "roi").')
    p.add_argument("--x-col", default="distance", help='Column name for X variable (default: "distance").')
    p.add_argument("--y-col", default="Error_deg", help='Column name for Y variable (default: "Error_deg").')
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load
    try:
        df = pd.read_csv(args.infile, sep="\t")
    except Exception as e:
        print(f"✖ Failed to read --infile: {args.infile}\n{e}", file=sys.stderr)
        sys.exit(1)

    # --- Column checks
    for col in (args.roi_col, args.x_col, args.y_col):
        if col not in df.columns:
            print(f"✖ Missing required column: {col}", file=sys.stderr)
            sys.exit(1)

    # Preserve original ROI order as they first appear
    roi_order = df[args.roi_col].drop_duplicates(keep="first").tolist()

    # Ensure numeric (drop rows that can't be coerced)
    df = df.copy()
    df[args.x_col] = pd.to_numeric(df[args.x_col], errors="coerce")
    df[args.y_col] = pd.to_numeric(df[args.y_col], errors="coerce")
    df = df.dropna(subset=[args.x_col, args.y_col, args.roi_col])

    if df.empty:
        print("✖ No valid rows after coercion to numeric.", file=sys.stderr)
        sys.exit(1)

    # --- Per-ROI Pearson r
    # Use apply to keep groups; skip ROIs with < 2 valid points
    rows = []
    for roi, g in df.groupby(args.roi_col, sort=False):
        x = g[args.x_col].values
        y = g[args.y_col].values
        if len(x) < 2:
            # Not enough data to compute correlation
            rows.append({args.roi_col: roi, "r": float("nan"), "p": float("nan")})
            continue
        r, p = pearsonr(x, y)
        rows.append({args.roi_col: roi, "r": r, "p": p})

    results = pd.DataFrame(rows)

    if results["p"].notna().any():
        # FDR (BH) over available p-values
        mask = results["p"].notna().values
        pvals = results.loc[mask, "p"].values
        _, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
        results["p_fdr"] = pd.NA
        results.loc[mask, "p_fdr"] = p_fdr
    else:
        results["p_fdr"] = pd.NA

    # Reorder to match original ROI order
    results[args.roi_col] = pd.Categorical(results[args.roi_col], categories=roi_order, ordered=True)
    results = results.sort_values(args.roi_col)

    # Save
    try:
        results.to_csv(args.outfile, sep="\t", index=False)
    except Exception as e:
        print(f"✖ Failed to write --outfile: {args.outfile}\n{e}", file=sys.stderr)
        sys.exit(1)

    # Small console summary
    n_roi = results.shape[0]
    n_valid = results["p"].notna().sum()
    print(f"✅ Saved correlations for {n_roi} ROI(s) "
          f"(valid tests: {n_valid}) -> {args.outfile}")


if __name__ == "__main__":
    main()
