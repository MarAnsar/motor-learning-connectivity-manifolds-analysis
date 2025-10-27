#!/usr/bin/env python3
"""
Average Day1 & Day2 seed-based FC tables and run paired epoch contrasts (with FDR).
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
from typing import Iterable, Tuple, List
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Average Day1/Day2 seed FC and run paired epoch contrasts with FDR."
    )
    p.add_argument(
        "--in-dir",
        required=True,
        help="Directory containing FC_<SEED>_Day1.tsv and FC_<SEED>_Day2.tsv.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write outputs (Avg TSVs and pairwise TSVs).",
    )
    p.add_argument(
        "--seeds",
        required=True,
        help=(
            "Comma-separated seed labels used in filenames, e.g.: "
            '"7Networks_LH_SalVentAttn_Med_2,7Networks_RH_SalVentAttn_Med_2,7Networks_LH_SomMot_26"'
        ),
    )
    p.add_argument(
        "--contrasts",
        default="Base>Early,Early>Late,Base>Late,Base>Washout,Late>Washout,Early>Washout",
        help=(
            "Comma-separated epoch contrasts as Second>First (order matters for the paired t-test): "
            'e.g. "Base>Early,Early>Late".'
        ),
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for FDR correction (default: 0.05).",
    )
    return p.parse_args()


# ---------------------------
# Helpers
# ---------------------------

REQUIRED_COLS = {"Subject", "Epoch", "Region", "Value"}
CANONICAL_EPOCHS = {"base": "Base", "early": "Early", "late": "Late", "washout": "Washout"}


def canonicalize_epoch_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize epoch labels to 'Base', 'Early', 'Late', 'Washout' if possible."""
    if "Epoch" not in df.columns:
        return df
    df = df.copy()
    df["Epoch"] = (
        df["Epoch"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: CANONICAL_EPOCHS.get(x, x.title()))
    )
    return df


def ensure_required_columns(df: pd.DataFrame, path_hint: str) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path_hint}: missing required columns: {sorted(missing)}; "
            f"found columns: {sorted(df.columns)}"
        )


def load_seed_file(in_dir: str, seed: str, day_label: str) -> pd.DataFrame:
    """Load one seed file for a given day (Day1 or Day2)."""
    fname = f"FC_{seed}_{day_label}.tsv"
    fpath = os.path.join(in_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Missing file: {fpath}")
    df = pd.read_csv(fpath, sep="\t")
    ensure_required_columns(df, fpath)
    df = canonicalize_epoch_names(df)
    return df


def average_day1_day2(in_dir: str, out_dir: str, seed: str) -> str:
    """
    Concatenate Day1 + Day2 and compute subject-wise mean across days
    per (Region, Subject, Epoch) on "Value".
    Return path to the averaged TSV.
    """
    df_day1 = load_seed_file(in_dir, seed, "Day1")
    df_day2 = load_seed_file(in_dir, seed, "Day2")

    combined = pd.concat([df_day1, df_day2], ignore_index=True)
    avg = (
        combined.groupby(["Region", "Subject", "Epoch"], sort=False)["Value"]
        .mean()
        .reset_index()
    )
    avg_path = os.path.join(out_dir, f"FC_{seed}_Avg.tsv")
    avg.to_csv(avg_path, sep="\t", index=False, float_format="%.17g")
    print(f"[avg] {seed}: saved {avg_path}")
    return avg_path


def parse_contrasts(spec: str) -> List[Tuple[str, str]]:
    """
    Parse "Base>Early,Early>Late" into [("Base","Early"),("Early","Late")].
    Remember: order is Second > First (as requested).
    """
    out: List[Tuple[str, str]] = []
    for part in [s for s in spec.split(",") if s.strip()]:
        if ">" not in part:
            raise ValueError(f"Bad contrast format (expected Second>First): {part}")
        left, right = [t.strip() for t in part.split(">", 1)]
        out.append((right, left))  # store as (First, Second) for pivot clarity
    return out


def run_pairwise_for_seed(avg_path: str, out_dir: str, seed: str, contrasts: Iterable[Tuple[str, str]], alpha: float):
    """
    For each Region, run paired t-tests within subjects on average file (FC_<seed>_Avg.tsv).
    contrasts is an iterable of tuples (First, Second) in natural order for pivoting.
    Results are saved to pairwise_<seed>.tsv with FDR-corrected p-values.
    """
    df = pd.read_csv(avg_path, sep="\t")
    ensure_required_columns(df, avg_path)
    df = canonicalize_epoch_names(df)

    results = []
    for region in df["Region"].unique():
        dfr = df[df["Region"] == region]
        for (first, second) in contrasts:
            # Pivot to Subject rows, two epoch columns
            pair = dfr[dfr["Epoch"].isin([first, second])]
            piv = pair.pivot(index="Subject", columns="Epoch", values="Value").dropna()
            if piv.shape[0] <= 1:
                continue  # need at least 2 subjects
            # ttest_rel uses arrays in "Second > First" sense if we pass (second, first)
            t_stat, p_val = ttest_rel(piv[second], piv[first])
            results.append(
                {
                    "Seed": seed,
                    "Region": region,
                    "Epochs": f"{second} > {first}",
                    "N": int(piv.shape[0]),
                    "T Value": float(t_stat),
                    "P Value": float(p_val),
                }
            )

    if not results:
        print(f"[pairwise] {seed}: no valid results (insufficient data).")
        return

    out_df = pd.DataFrame(results)
    # FDR (BH)
    _, p_fdr, _, _ = multipletests(out_df["P Value"].values, alpha=alpha, method="fdr_bh")
    out_df["FDR-corrected P Value"] = p_fdr

    out_path = os.path.join(out_dir, f"pairwise_{seed}.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[pairwise] {seed}: saved {out_path}")


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("No seeds provided via --seeds.", file=sys.stderr)
        sys.exit(1)

    try:
        # parse contrasts as “Second>First” then store internally as (First, Second)
        # to simplify pivoting and keep the test direction “Second > First”.
        contrasts = parse_contrasts(args.contrasts)
    except ValueError as e:
        print(f"Contrast parse error: {e}", file=sys.stderr)
        sys.exit(1)

    for seed in seeds:
        # 1) Average Day1 + Day2
        try:
            avg_path = average_day1_day2(args.in_dir, args.out_dir, seed)
        except Exception as e:
            print(f"[avg:{seed}] ERROR: {e}")
            continue

        # 2) Pairwise tests with FDR
        try:
            run_pairwise_for_seed(avg_path, args.out_dir, seed, contrasts, args.alpha)
        except Exception as e:
            print(f"[pairwise:{seed}] ERROR: {e}")

    print("✅ Averaging & pairwise testing complete.")


if __name__ == "__main__":
    main()
