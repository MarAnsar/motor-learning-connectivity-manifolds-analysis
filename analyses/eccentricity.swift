#!/usr/bin/env python3
"""
Compute per-ROI eccentricity from PCA gradients — separately for Day 1 and Day 2.

Eccentricity (here) = Euclidean distance of each region’s gradient vector
(e.g., [g1, g2, g3]) from the centroid (mean vector) of all regions within the
same Subject × Session × Epoch.

Expected on disk (default naming):
  <data_dir>/
    sub-01/
      sub-01_ses-01_task-rotation_base_gradient.tsv
      sub-01_ses-01_task-rotation_early_gradient.tsv
      sub-01_ses-01_task-rotation_late_gradient.tsv
      sub-01_ses-01_task-washout_washout_gradient.tsv
      sub-01_ses-02_task-rotation_base_gradient.tsv
      ...
    sub-02/
      ...

Each file must be a TSV with columns:
  Region, g1, g2, g3, ...  (we only use the first 3 by default)

Outputs (per session):
  <out_dir>/eccentricity_3PCA_full_<session>.tsv
  <out_dir>/eccentricity_3PCA_simplified_<session>.tsv

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


# ----------------------
# I/O and parsing
# ----------------------

def find_subjects(root: Path) -> List[str]:
    """Return ['sub-XX', ...] for subdirectories present under root."""
    subs = sorted([p.name for p in root.glob("sub-*") if p.is_dir()])
    if not subs:
        raise FileNotFoundError(f"No 'sub-*' folders found under: {root}")
    return subs


def load_one_gradient_tsv(path: Path, gradients: Sequence[str]) -> pd.DataFrame:
    """Load a single gradient TSV and ensure expected columns exist and are numeric."""
    df = pd.read_csv(path, sep="\t")
    expected = ["Region", *gradients]
    if not all(col in df.columns for col in expected):
        print(f"[warn] {path.name}: missing expected columns. Found: {list(df.columns)}; expected at least: {expected}")
    # Coerce gradient columns to numeric
    for g in gradients:
        if g in df.columns:
            df[g] = pd.to_numeric(df[g], errors="coerce")
        else:
            df[g] = np.nan
    return df[["Region", *gradients]]


def collect_session_rows(
    data_dir: Path,
    subject: str,
    session: str,
    epochs: Sequence[str],
    gradients: Sequence[str],
    rot_pattern: str,
    wo_pattern: str,
) -> List[pd.DataFrame]:
    """
    Collect rows for one Subject × Session across requested epochs.
    Handles rotation vs washout filename patterns.
    """
    rows = []
    for ep in epochs:
        if ep.lower() == "washout":
            fname = wo_pattern.format(sub=subject, ses=session, epoch=ep)
        else:
            fname = rot_pattern.format(sub=subject, ses=session, epoch=ep)

        fpath = data_dir / subject / fname
        if not fpath.exists():
            print(f"[warn] Missing file: {fpath}")
            continue

        df = load_one_gradient_tsv(fpath, gradients)
        df["sub"] = subject
        df["ses"] = session
        df["epoch"] = ep
        rows.append(df)
    return rows


# ----------------------
# Eccentricity
# ----------------------

def compute_eccentricity(df: pd.DataFrame, gradients: Sequence[str]) -> pd.DataFrame:
    """
    Compute Euclidean distance from centroid within each sub × ses × epoch group.
    """
    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        G = g.loc[:, gradients].astype(float).to_numpy()
        center = np.nanmean(G, axis=0)  # centroid in gradient space
        ecc = np.linalg.norm(G - center, axis=1)
        g = g.copy()
        g["eccentricity"] = ecc
        return g

    return df.groupby(["sub", "ses", "epoch"], as_index=False, group_keys=False).apply(_per_group)


# ----------------------
# CLI
# ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Compute per-ROI eccentricity per session (Day 1 & Day 2).")
    ap.add_argument("--data-dir", required=True, help="Root directory containing sub-*/ gradient TSVs.")
    ap.add_argument("--out-dir", required=True, help="Directory to write output TSVs.")
    ap.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02"],
                    help="Sessions to process (e.g., ses-01 ses-02).")
    ap.add_argument("--epochs", nargs="+", default=["base", "early", "late", "washout"],
                    help="Epochs to include.")
    ap.add_argument("--gradients", nargs="+", default=["g1", "g2", "g3"],
                    help="Gradient columns to use for eccentricity.")
    ap.add_argument("--rotation-pattern", default="{sub}_{ses}_task-rotation_{epoch}_gradient.tsv",
                    help="Filename pattern for rotation epochs (base/early/late).")
    ap.add_argument("--washout-pattern", default="{sub}_{ses}_task-washout_washout_gradient.tsv",
                    help="Filename pattern for washout epoch.")
    ap.add_argument("--full-name", default="eccentricity_3PCA_full_{ses}.tsv",
                    help="Output filename (full) per session.")
    ap.add_argument("--simple-name", default="eccentricity_3PCA_simplified_{ses}.tsv",
                    help="Output filename (simplified) per session.")
    return ap.parse_args()


# ----------------------
# Main
# ----------------------

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(data_dir)

    for ses in args.sessions:
        print(f"[info] Processing session: {ses}")
        per_session_frames = []

        for sub in subjects:
            rows = collect_session_rows(
                data_dir=data_dir,
                subject=sub,
                session=ses,
                epochs=args.epochs,
                gradients=args.gradients,
                rot_pattern=args.rotation-pattern if hasattr(args, "rotation-pattern") else args.rotation_pattern,  # safety
                wo_pattern=args.washout-pattern if hasattr(args, "washout-pattern") else args.washout_pattern,
            )
            if rows:
                per_session_frames.append(pd.concat(rows, ignore_index=True))

        if not per_session_frames:
            print(f"[warn] No data loaded for {ses}. Skipping.")
            continue

        session_df = pd.concat(per_session_frames, ignore_index=True)

        # Compute eccentricity within sub × ses × epoch
        ecc_df = compute_eccentricity(session_df, gradients=args.gradients)

        # Save: full (keep gradients), simplified (drop gradients)
        full_path = out_dir / args.full_name.format(ses=ses)
        simp_path = out_dir / args.simple_name.format(ses=ses)

        ecc_df.to_csv(full_path, sep="\t", index=False)

        simplified = ecc_df.drop(columns=[g for g in args.gradients if g in ecc_df.columns])
        simplified.to_csv(simp_path, sep="\t", index=False)

        print(f"[ok] Saved: {full_path}")
        print(f"[ok] Saved: {simp_path}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
