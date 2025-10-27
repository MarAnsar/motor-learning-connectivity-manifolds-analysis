#!/usr/bin/env python3
"""
Fix a labeled connectivity matrix, apply dominant-set sparsification, and
export the sparsified node strengths.

"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
from brainspace.gradient.utils import dominant_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fix labeled connectivity matrix, apply dominant-set sparsification, and export node strengths."
    )
    p.add_argument("--infile", required=True, help="Input TSV matrix (first column = row labels; header = column labels).")
    p.add_argument("--outdir", required=True, help="Directory to write outputs.")
    p.add_argument(
        "--keep-fraction",
        type=float,
        default=0.10,
        help="Fraction of strongest entries to keep per row (e.g., 0.10 keeps top 10%%).",
    )
    p.add_argument(
        "--symmetrize",
        action="store_true",
        help="If set, enforce symmetry by 0.5*(W + W.T) when matrix is not already symmetric (within small tolerance).",
    )
    p.add_argument(
        "--coerce-int-labels",
        action="store_true",
        help="If set, coerce row/column labels to integers (e.g., '032'→32). Disable if your labels are not numeric.",
    )
    p.add_argument(
        "--ds-is-thresh",
        action="store_true",
        help="Pass is_thresh=True to brainspace.utils.dominant_set. By default we use is_thresh=False to mirror legacy behavior.",
    )
    p.add_argument(
        "--prefix",
        default="reference_cmat",
        help='Filename prefix for outputs (default: "reference_cmat").',
    )
    return p.parse_args()


def load_labeled_matrix_tsv(path: str, coerce_int_labels: bool) -> pd.DataFrame:
    """Load TSV where first column is row labels; header row are column labels."""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
    except Exception as e:
        print(f"✖ Failed to read TSV: {path}\n{e}", file=sys.stderr)
        sys.exit(1)

    if df.shape[1] < 2:
        print("✖ Input must have labels + at least one numeric column.", file=sys.stderr)
        sys.exit(1)

    # Set row labels to the first column; strip whitespace for safety
    df = df.set_index(df.columns[0])
    df.index = df.index.str.strip()
    df.columns = df.columns.str.strip()

    if coerce_int_labels:
        try:
            df.index = df.index.astype(int)
            df.columns = df.columns.astype(int)
        except Exception as e:
            print("✖ Could not coerce labels to integers. Disable --coerce-int-labels if labels are not numeric.",
                  file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)

    # Convert cell values to float
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # Ensure columns match rows (same set), then reorder columns to row order
    row_ids = pd.Index(df.index)
    col_ids = pd.Index(df.columns)

    rows_only = row_ids.difference(col_ids)
    cols_only = col_ids.difference(row_ids)
    if len(rows_only) or len(cols_only):
        raise ValueError(
            "Row/column label sets differ.\n"
            f"Rows only (first 10): {list(rows_only)[:10]}\n"
            f"Cols only (first 10): {list(cols_only)[:10]}"
        )

    df = df.loc[row_ids, row_ids]
    return df


def maybe_symmetrize(df: pd.DataFrame, do_symmetrize: bool, atol=1e-8, rtol=1e-8) -> pd.DataFrame:
    """Optionally symmetrize if not already symmetric within tolerance."""
    A = df.to_numpy(dtype=float)
    if do_symmetrize and not np.allclose(A, A.T, atol=atol, rtol=rtol):
        A = 0.5 * (A + A.T)
        df = pd.DataFrame(A, index=df.index, columns=df.columns)
    return df


def run_dominant_set_and_strength(df: pd.DataFrame, keep_fraction: float, is_thresh: bool) -> pd.Series:
    """
    Apply dominant-set row-wise sparsification and return node strength (row sums).
    We mirror the legacy call: dominant_set(W, k=keep_fraction, is_thresh=False, as_sparse=False).
    """
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("--keep-fraction must be in (0, 1].")

    X = dominant_set(
        df.values,
        k=keep_fraction,
        is_thresh=is_thresh,     # default False to match original behavior
        as_sparse=False
    )
    strength = pd.Series(np.sum(X, axis=1), index=df.index, name="strength")
    return strength


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fixed_matrix_file = os.path.join(args.outdir, f"{args.prefix}_fixed.tsv")
    strength_file     = os.path.join(args.outdir, "strength_values.tsv")

    # 1) Load & fix
    df = load_labeled_matrix_tsv(args.infile, coerce_int_labels=args.coerce_int_labels)
    df_fixed = maybe_symmetrize(df, do_symmetrize=args.symmetrize)

    # 2) Save fixed matrix for QC/reuse
    df_fixed.to_csv(fixed_matrix_file, sep="\t")

    # 3) Dominant-set sparsification + node strength
    strength = run_dominant_set_and_strength(
        df_fixed, keep_fraction=args.keep_fraction, is_thresh=args.ds_is_thresh
    )

    # 4) Save strengths
    strength.to_csv(strength_file, sep="\t", header=True)

    print("✅ Saved:")
    print(f" - Fixed matrix: {fixed_matrix_file}")
    print(f" - Node strength: {strength_file}")


if __name__ == "__main__":
    main()
