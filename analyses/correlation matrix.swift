#!/usr/bin/env python3
"""
Per-subject epoch×epoch correlation matrices from flattened PCA vectors.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def list_subject_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {root}")
    return files


def subject_id_from_filename(fname: Path) -> str:
    """
    Extract a subject ID from the filename.
    Default: take the first underscore-separated token (e.g., 'sub-01_vectors.tsv' -> 'sub-01').
    Modify here if your naming differs.
    """
    stem = fname.stem  # e.g., 'sub-01_vectors'
    return stem.split("_")[0]


def load_epoch_matrix(tsv_path: Path, index_col: str = "epoch") -> pd.DataFrame:
    """
    Load a TSV with an 'epoch' column. Returns a DataFrame indexed by epoch,
    with numeric feature columns (flattened PCA features).
    """
    df = pd.read_csv(tsv_path, sep="\t")
    if index_col not in df.columns:
        raise ValueError(f"Required column '{index_col}' not found in {tsv_path}")
    df = df.set_index(index_col)

    # Ensure all remaining columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all(axis=None):
        raise ValueError(f"All values are NaN after numeric coercion in {tsv_path}.")
    # Drop columns that are entirely NaN (if any)
    df = df.dropna(axis=1, how="all")
    # Fill any remaining NaNs per column with column mean (conservative)
    df = df.fillna(df.mean(numeric_only=True))
    return df


def epoch_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute epoch×epoch Pearson correlation (rows=epochs, columns=features).
    Returns a square DataFrame with epoch labels on rows/cols.
    """
    # np.corrcoef expects observations as rows; we already have epochs as rows
    corr = np.corrcoef(df.values)
    return pd.DataFrame(corr, index=df.index, columns=df.index)


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute per-subject epoch×epoch correlation matrices from flattened PCA vectors."
    )
    ap.add_argument("--input-dir", required=True,
                    help="Directory containing per-subject TSVs (rows=epochs, columns=flattened PCA features).")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write per-subject correlation matrices (TSV).")
    ap.add_argument("--pattern", default="*_vectors.tsv",
                    help="Glob pattern for subject files (default: '*_vectors.tsv').")
    ap.add_argument("--index-col", default="epoch",
                    help="Column to use as epoch index (default: 'epoch').")
    return ap.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_subject_files(in_dir, args.pattern)

    for f in files:
        subj_id = subject_id_from_filename(f)
        df = load_epoch_matrix(f, index_col=args.index_col)
        cmat = epoch_correlation(df)

        out_path = out_dir / f"{subj_id}_corr_matrix.tsv"
        cmat.to_csv(out_path, sep="\t")
        print(f"[ok] {subj_id}: saved {out_path}")

    print("✅ All correlation matrices saved successfully.")

    # -------------------------------------------------------------------------
    # NOTE (group-level):
    # After running this script, compute the GROUP correlation matrix by:
    # 1) Reading all '<subj>_corr_matrix.tsv' files 
    # 2) Stacking them into a 3D array and taking the elementwise mean:
    #       group = np.mean(np.stack([df.values for df in subjects], axis=0), axis=0)
    #    (ensure the same epoch order across subjects by reindexing on a common label list)
    # 3) Save 'group' as a TSV with the same epoch labels for rows/columns.
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    main()
