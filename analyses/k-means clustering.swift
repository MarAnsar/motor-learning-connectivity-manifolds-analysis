#!/usr/bin/env python3
"""
K-means clustering on eccentricity data (wide table).

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def read_table(path: Path) -> pd.DataFrame:
    """Read TSV/CSV based on file extension."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="K-means clustering on eccentricity data (wide format).")
    ap.add_argument("--input", required=True, type=Path, help="Input TSV/CSV (wide table with 'Region' + numeric columns).")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    ap.add_argument("--k", type=int, default=2, help="Number of clusters (default: 2).")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for KMeans (default: 42).")
    ap.add_argument("--n-init", type=int, default=10, help="n_init for KMeans (default: 10).")
    ap.add_argument("--x-col", type=str, default=None, help="Column name for x-axis in optional plot.")
    ap.add_argument("--y-col", type=str, default=None, help="Column name for y-axis in optional plot.")
    ap.add_argument("--save-fig", action="store_true", help="If set, saves PNG and SVG plots when x/y columns are provided.")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data
    df = read_table(args.input)
    if "Region" not in df.columns:
        raise ValueError("Input table must contain a 'Region' column.")

    # Separate out numeric features
    feature_cols = [c for c in df.columns if c != "Region"]
    if not feature_cols:
        raise ValueError("No numeric columns found besides 'Region'.")

    numeric_data = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if numeric_data.isna().any().any():
        # Simple strategy: drop rows with NaNs in features
        keep_mask = ~numeric_data.isna().any(axis=1)
        dropped = (~keep_mask).sum()
        if dropped > 0:
            print(f"[info] Dropping {dropped} rows with NaNs in feature columns.")
        df = df.loc[keep_mask].reset_index(drop=True)
        numeric_data = numeric_data.loc[keep_mask].reset_index(drop=True)

    # ---- KMeans clustering
    kmeans = KMeans(n_clusters=args.k, random_state=args.random_state, n_init=args.n_init)
    labels = kmeans.fit_predict(numeric_data) + 1  # make labels 1-based
    df["Cluster"] = labels

    # ---- Save results
    out_tsv = args.out_dir / f"eccentricity_kmeans_{args.k}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"✅ K-means (k={args.k}) results saved to: {out_tsv}")

    # ---- Optional scatter plot
    if args.x_col and args.y_col:
        if args.x_col not in df.columns or args.y_col not in df.columns:
            missing = [c for c in [args.x_col, args.y_col] if c not in df.columns]
            raise ValueError(f"Requested plot columns not found: {missing}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x=args.x_col,
            y=args.y_col,
            hue="Cluster",
            palette="coolwarm",
            s=100
        )
        plt.xlabel(args.x_col.replace("_", " "))
        plt.ylabel(args.y_col.replace("_", " "))
        plt.title(f"K-Means Clustering (k={args.k}) on Eccentricity Data")
        plt.legend(title="Cluster")
        plt.grid(True)

        if args.save-fig:  # noqa: E999 (hyphen in flag name)
            base = args.out_dir / f"kmeans_{args.k}_{args.x_col}_vs_{args.y_col}"
            plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.savefig(base.with_suffix(".svg"), bbox_inches="tight")
            print(f"🖼️  Plot saved to: {base}.png / {base}.svg")

        plt.show()


if __name__ == "__main__":
    main()
