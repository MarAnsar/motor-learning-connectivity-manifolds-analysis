#!/usr/bin/env python3
"""
Day × Epoch rmANOVA per ROI (supports eccentricity data)

This script runs a 2×4 repeated-measures ANOVA (within-subject factors:
    - Day   : 2 ordered levels (default: D1, D2)
    - Epoch : 4 ordered levels (default: base, early, late, washout)
for each ROI independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pingouin as pg


# -------------------------
# I/O helpers
# -------------------------

def load_long_table(path: Path) -> pd.DataFrame:
    """Read TSV or CSV based on file extension."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def ensure_long_schema(
    df: pd.DataFrame,
    day_order: List[str] | None,
    epoch_order: List[str] | None,
) -> pd.DataFrame:
    """Validate columns and coerce Day/Epoch to ordered categoricals."""
    required = ["Subject", "Day", "Epoch", "Region", "Value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    out = df.copy()
    out["Subject"] = out["Subject"].astype(str)
    out["Region"] = out["Region"].astype(str)
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")

    if day_order is None:
        day_order = sorted(out["Day"].dropna().astype(str).unique().tolist())
    if epoch_order is None:
        epoch_order = sorted(out["Epoch"].dropna().astype(str).unique().tolist())

    out["Day"] = pd.Categorical(out["Day"].astype(str), categories=day_order, ordered=True)
    out["Epoch"] = pd.Categorical(out["Epoch"].astype(str), categories=epoch_order, ordered=True)

    out = out.dropna(subset=["Value"])
    return out


# -------------------------
# Stats core
# -------------------------

def run_rm_anova_day_epoch_per_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ROI: run rmANOVA with factors Day (2) and Epoch (4).

    Returns one tidy DataFrame with columns:
        Region, Effect, F, p-unc, p-FDR, (optional: eps, spher, W-spher, ddof1, ddof2, np2)
    """
    rows: List[Dict] = []

    for region, sub in df.groupby("Region", sort=False):
        try:
            aov = pg.rm_anova(
                dv="Value",
                within=["Day", "Epoch"],
                subject="Subject",
                data=sub,
                detailed=True
            )
            # Keep only the within effects (Day, Epoch, Day * Epoch)
            aov = aov[aov["Source"].isin(["Day", "Epoch", "Day * Epoch"])].copy()
            for _, r in aov.iterrows():
                row = {
                    "Region": region,
                    "Effect": r["Source"].replace(" * ", "×"),
                    "F": r.get("F", np.nan),
                    "p-unc": r.get("p-unc", np.nan),
                }
                # Optional extras if available
                for c in ["np2", "eps", "spher", "W-spher", "ddof1", "ddof2"]:
                    if c in aov.columns:
                        row[c] = r.get(c, np.nan)
                rows.append(row)
        except Exception as e:
            rows.append({
                "Region": region,
                "Effect": "ANOVA_failed",
                "F": np.nan,
                "p-unc": np.nan,
                "error": str(e)
            })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # FDR per effect across ROIs
    out_parts = []
    for eff, grp in res.groupby("Effect", sort=False):
        p = grp["p-unc"].to_numpy()
        mask = ~np.isnan(p)
        if mask.any():
            _, p_fdr = pg.multicomp(p[mask], method="fdr_bh")
            grp = grp.copy()
            grp.loc[mask, "p-FDR"] = p_fdr
        else:
            grp["p-FDR"] = np.nan
        out_parts.append(grp)

    return pd.concat(out_parts, ignore_index=True)


# -------------------------
# CLI
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run Day × Epoch rmANOVA per ROI (e.g., on eccentricity data)."
    )
    ap.add_argument(
        "--in-file",
        required=True,
        help="Path to long-format TSV/CSV with columns: Subject, Day, Epoch, Region, Value.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write results.",
    )
    ap.add_argument(
        "--day-order",
        nargs="*",
        default=["D1", "D2"],
        help="Ordered levels for Day (default: D1 D2).",
    )
    ap.add_argument(
        "--epoch-order",
        nargs="*",
        default=["base", "early", "late", "washout"],
        help="Ordered levels for Epoch (default: base early late washout).",
    )
    ap.add_argument(
        "--outfile-name",
        default="rmANOVA_DayxEpoch_perROI.tsv",
        help="Output filename (TSV).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_long_table(in_path)
    df = ensure_long_schema(df, args.day_order, args.epoch_order)

    results = run_rm_anova_day_epoch_per_roi(df)
    out_file = out_dir / args.outfile_name
    results.to_csv(out_file, sep="\t", index=False)

    print(f"✅ Done. Results saved to: {out_file}")


if __name__ == "__main__":
    main()
