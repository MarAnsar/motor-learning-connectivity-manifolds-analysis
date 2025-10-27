#!/usr/bin/env python3
"""
Compute node participation coefficients between TWO predefined ROI communities
after row-wise dominant-set sparsification of a labeled connectivity matrix.

ROI ID sets
-----------
Pass the two communities via text files containing one integer ROI ID per line.
Example file contents:

    # file: net_a_ids.txt
    32
    33
    ...
    270

    # file: net_b_ids.txt
    92
    93
    ...
    318

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
    print("✖ brainspace is required (pip install brainspace).", file=sys.stderr)
    raise

try:
    import bct  # bctpy
except Exception as e:
    print("✖ bctpy is required (pip install bctpy).", file=sys.stderr)
    raise


# -----------------------
# CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Participation (two communities) after dominant-set sparsification."
    )
    p.add_argument("--cmat", required=True, help="Input TSV: labeled connectivity matrix (IDs on rows/cols).")
    p.add_argument("--net-a-ids", required=True, help="Text file with ROI IDs (one per line) for community A.")
    p.add_argument("--net-b-ids", required=True, help="Text file with ROI IDs (one per line) for community B.")
    p.add_argument("--label-a", default="A", help='Label for community A (default: "A").')
    p.add_argument("--label-b", default="B", help='Label for community B (default: "B").')
    p.add_argument("--keep-fraction", type=float, default=0.10,
                   help="Fraction of strongest entries to keep per row (e.g., 0.10 keeps top 10%).")
    p.add_argument("--symmetrize", action="store_true",
                   help="Symmetrize matrix as 0.5*(W + W.T) if not perfectly symmetric.")
    p.add_argument("--out", required=True, help="Output TSV path for participation values.")
    return p.parse_args()


# -----------------------
# I/O helpers
# -----------------------
def read_id_list(path: str) -> set[int]:
    """Read integer IDs from a text file (one per line)."""
    ids = set()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.add(int(s))
    if not ids:
        raise ValueError(f"No IDs found in: {path}")
    return ids


def load_labeled_cmat(path: str) -> pd.DataFrame:
    """
    Load TSV where first column is row labels and the header row are column labels.
    Coerce labels to integers; values to float; reorder columns to match row order.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.shape[1] < 2:
        raise ValueError("Connectivity file must have labels + at least one numeric column.")

    # First column becomes index (row labels)
    df = df.set_index(df.columns[0])
    df.index = df.index.str.strip().astype(int)
    df.columns = df.columns.str.strip().astype(int)

    # Values to float
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # Ensure same label set and reorder columns to row order
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


# -----------------------
# Core logic
# -----------------------
def dominant_set_sparsify(W: np.ndarray, keep_fraction: float) -> np.ndarray:
    """
    Apply dominant-set sparsification row-wise, keeping the top-k fraction.
    Mirrors: dominant_set(W, k=keep_fraction, is_thresh=False, as_sparse=False).
    """
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("--keep-fraction must be in (0, 1].")
    X = dominant_set(W, k=keep_fraction, is_thresh=False, as_sparse=False)
    return X


def compute_two_community_participation(W: np.ndarray, ids: list[int],
                                        set_a: set[int], set_b: set[int],
                                        label_a: str, label_b: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep ROIs that belong to exactly one of the two sets (exclusive),
    build community vector, then compute participation coefficients.
    Returns:
        keep_idx (np.ndarray): indices (into original W/ids) of kept ROIs
        P        (np.ndarray): participation coefficients (kept ROIs only)
    """
    # Identify ROIs that are in A xor B
    mask_keep = np.array([(i in set_a) ^ (i in set_b) for i in ids], dtype=bool)
    keep_idx = np.where(mask_keep)[0]
    if keep_idx.size == 0:
        raise ValueError("No ROIs belong exclusively to one of the two communities.")

    # Slice matrix and build 1/2 community assignment for bctpy
    Wk = W[keep_idx][:, keep_idx]
    kept_ids = [ids[i] for i in keep_idx]
    comm = np.array([1 if i in set_a else 2 for i in kept_ids], dtype=int)

    # Weighted participation (row-wise)
    P = bct.participation_coef(Wk, comm, 'out')  # shape (N_kept,)
    return keep_idx, P


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1) Load inputs
    df = load_labeled_cmat(args.cmat)
    ids = df.index.to_list()
    A = df.to_numpy(float)

    # (Optional) symmetrize tiny mismatch
    if args.symmetrize and not np.allclose(A, A.T, atol=1e-8, rtol=1e-8):
        A = 0.5 * (A + A.T)

    # Read communities
    set_a = read_id_list(args.net_a_ids)
    set_b = read_id_list(args.net_b_ids)

    # 2) Dominant-set sparsify (keep top fraction per row)
    X = dominant_set_sparsify(A, keep_fraction=args.keep_fraction)

    # 3) Participation for ROIs in exactly one of the two communities
    keep_idx, P = compute_two_community_participation(
        X, ids, set_a, set_b, args.label_a, args.label_b
    )

    kept_ids = [ids[i] for i in keep_idx]
    kept_labels = [args.label_a if i in set_a else args.label_b for i in kept_ids]

    out_df = pd.DataFrame(
        {"id": kept_ids, "community": kept_labels, "participation": P},
        columns=["id", "community", "participation"],
    )
    out_df.to_csv(args.out, sep="\t", index=False)

    # Small summary
    n_a = sum(1 for i in kept_ids if i in set_a)
    n_b = sum(1 for i in kept_ids if i in set_b)
    print("✅ Saved participation (two communities):", args.out)
    print(f"   kept ROIs: {len(kept_ids)} | {args.label_a}: {n_a} | {args.label_b}: {n_b}")
    print(f"   participation range: [{float(np.min(P)):.3f}, {float(np.max(P)):.3f}]")


if __name__ == "__main__":
    main()
