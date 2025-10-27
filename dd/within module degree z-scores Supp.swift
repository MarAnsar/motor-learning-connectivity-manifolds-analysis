#!/usr/bin/env python3
"""
Compute within-module degree z-scores 
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    from brainspace.gradient.utils import dominant_set
except Exception as e:
    print("✖ brainspace is required: pip install brainspace", file=sys.stderr)
    raise

try:
    import bct  # bctpy
except Exception as e:
    print("✖ bctpy is required: pip install bctpy", file=sys.stderr)
    raise


# -----------------------
# CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Within-module degree z-scores after dominant-set sparsification."
    )
    p.add_argument("--cmat", required=True,
                   help="TSV connectivity matrix with integer ROI IDs on rows/cols.")
    p.add_argument("--communities", required=True,
                   help="TSV with columns [id, module] mapping ROI → module label.")
    p.add_argument("--keep-fraction", type=float, default=0.10,
                   help="Fraction of strongest entries to keep per row (e.g., 0.10).")
    p.add_argument("--symmetrize", action="store_true",
                   help="If set, symmetrize matrix as 0.5*(W+W.T) before sparsification.")
    p.add_argument("--binarize", action="store_true",
                   help="If set, binarize the sparsified matrix (nonzero → 1).")
    p.add_argument("--out", required=True, help="Output TSV path.")
    return p.parse_args()


# -----------------------
# I/O helpers
# -----------------------
def load_labeled_cmat(path: str) -> pd.DataFrame:
    """
    Load TSV where first column is row labels and header row are column labels.
    Coerce labels to integers; values to float; reorder columns to match row order.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.shape[1] < 2:
        raise ValueError("Connectivity file must have labels + at least one numeric column.")

    # First column → index
    df = df.set_index(df.columns[0])
    df.index = df.index.str.strip().astype(int)
    df.columns = df.columns.str.strip().astype(int)

    # Values to float
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # Ensure exact same label set, then align columns to row order
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
    return df.loc[row_ids, row_ids]


def load_communities(path: str) -> pd.DataFrame:
    """
    Load ROI→module mapping and ensure expected columns exist.
    Returns a DataFrame with columns: id(int), module(original), module_int(int).
    """
    com = pd.read_csv(path, sep="\t", dtype={"id": int, "module": str})
    if not {"id", "module"}.issubset(com.columns):
        raise ValueError("--communities TSV must have columns: id, module")
    # Normalize whitespace
    com["module"] = com["module"].astype(str).str.strip()
    # Factorize module labels to consecutive integers (1..K)
    mod_codes, mod_uniques = pd.factorize(com["module"], sort=True)
    com["module_int"] = mod_codes + 1  # 1-based for bctpy
    return com[["id", "module", "module_int"]]


# -----------------------
# Core logic
# -----------------------
def dominant_set_sparsify(W: np.ndarray, keep_fraction: float) -> np.ndarray:
    """
    Apply dominant-set sparsification row-wise, keeping the top-k fraction.
    """
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("--keep-fraction must be in (0, 1].")
    return dominant_set(W, k=keep_fraction, is_thresh=False, as_sparse=False)


def module_degree_z(W: np.ndarray, modules: np.ndarray) -> np.ndarray:
    """
    Compute within-module degree z-scores using bctpy.
    W: (N,N) weighted or binary symmetric matrix
    modules: (N,) integer module labels (1..K)
    """
    # bctpy expects integers starting at 1
    if modules.min() < 1:
        raise ValueError("module labels must start at 1.")
    z = bct.module_degree_zscore(W, modules, 2)  # '2' = undirected/weighted
    return np.nan_to_num(z, nan=0.0)


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1) Load matrix + communities
    df = load_labeled_cmat(args.cmat)
    com = load_communities(args.communities)

    # Keep only ROIs present in both matrix and communities
    ids = df.index.tolist()
    keep_ids = sorted(set(ids) & set(com["id"]))
    if not keep_ids:
        raise ValueError("No overlap between matrix ROI IDs and community ROI IDs.")

    df = df.loc[keep_ids, keep_ids].copy()
    com = com.set_index("id").loc[keep_ids].reset_index()

    W = df.to_numpy(float)

    # 2) Optional symmetrize
    if args.symmetrize and not np.allclose(W, W.T, atol=1e-8, rtol=1e-8):
        W = 0.5 * (W + W.T)

    # 3) Dominant-set sparsify
    Ws = dominant_set_sparsify(W, keep_fraction=args.keep_fraction)

    # 4) Optional binarize
    if args.binarize:
        Ws = (Ws != 0).astype(float)

    # 5) Within-module degree z-score
    Z = module_degree_z(Ws, com["module_int"].to_numpy(int))

    # 6) Save
    out_df = pd.DataFrame(
        {
            "id": keep_ids,
            "module": com["module"].to_list(),
            "module_degree": Z,
        }
    )
    out_df.to_csv(args.out, sep="\t", index=False)

    # Summary
    print(f"✅ Saved within-module degree z-scores to: {args.out}")
    print(f"N ROIs = {len(keep_ids)} | mean z = {np.mean(Z):.4f} | range = ({np.min(Z):.4f}, {np.max(Z):.4f})")


if __name__ == "__main__":
    main()
