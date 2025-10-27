#!/usr/bin/env python3
"""
Evaluate clustering quality with Silhouette and Davies–Bouldin (DBI).

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ---------------------------
# Utilities
# ---------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Read TSV/CSV based on extension."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def numeric_matrix(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """Return only numeric columns, optionally dropping known non-numerics first."""
    keep = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return keep.select_dtypes(include=[np.number])


def make_line_plot(x, y, best_k, title, ylabel, out_png: Path, out_svg: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(x, y, marker="o", linestyle="-", color="black", linewidth=0.8)
    if best_k is not None:
        ax.axvline(x=best_k, linestyle="--", color="red", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)

    # light spines for a clean look
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    ax.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Compute Silhouette and DBI over k.")
    ap.add_argument("--in-dir", required=True,
                    help="Folder containing per-k clustering files.")
    ap.add_argument("--out-dir", required=True,
                    help="Folder to write metrics and plots.")
    ap.add_argument("--k", nargs="+", type=int, required=True,
                    help="k values to evaluate (e.g., --k 2 3 4 5 6).")
    ap.add_argument("--pattern", default="eccentricity_kmeans_{k}.tsv",
                    help="Filename pattern inside --in-dir (use {k} placeholder).")
    ap.add_argument("--label-col", default="Cluster",
                    help="Column name with cluster labels. Default: Cluster")
    ap.add_argument("--id-col", default="Region",
                    help="Non-numeric identifier column to drop before scoring. Default: Region")
    return ap.parse_args()


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = sorted(set(args.k))
    rows = []

    for k in ks:
        fpath = in_dir / args.pattern.format(k=k)
        if not fpath.exists():
            print(f"[warn] Missing file for k={k}: {fpath}")
            continue

        df = read_table(fpath)

        # Validate presence of label column
        if args.label_col not in df.columns:
            raise ValueError(f"{fpath.name}: label column '{args.label_col}' not found in columns {df.columns.tolist()}")

        # Extract numeric feature matrix
        X = numeric_matrix(df, drop_cols=[args.id_col, args.label_col])
        if X.shape[1] == 0:
            raise ValueError(f"{fpath.name}: no numeric feature columns found after dropping [{args.id_col}, {args.label_col}]")

        labels = df[args.label_col].to_numpy()
        # Basic sanity check
        if len(np.unique(labels)) < 2:
            print(f"[warn] k={k} has <2 unique clusters in '{args.label_col}'. Skipping.")
            continue

        # Compute metrics
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)

        rows.append({"k": k, "Silhouette": sil, "DBI": dbi})

    if not rows:
        raise RuntimeError("No metrics computed. Check your --k list and file pattern/paths.")

    metrics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    metrics_path = out_dir / "clustering_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    print(f"[ok] Metrics saved -> {metrics_path}")

    # Choose "best" k per metric
    best_k_sil = metrics.loc[metrics["Silhouette"].idxmax(), "k"]
    best_k_dbi = metrics.loc[metrics["DBI"].idxmin(), "k"]

    # Individual plots
    make_line_plot(
        x=metrics["k"].tolist(),
        y=metrics["Silhouette"].tolist(),
        best_k=best_k_sil,
        title="Silhouette",
        ylabel="Silhouette score",
        out_png=out_dir / "silhouette_plot.png",
        out_svg=out_dir / "silhouette_plot.svg",
    )
    make_line_plot(
        x=metrics["k"].tolist(),
        y=metrics["DBI"].tolist(),
        best_k=best_k_dbi,
        title="Davies–Bouldin Index",
        ylabel="DBI (lower is better)",
        out_png=out_dir / "dbi_plot.png",
        out_svg=out_dir / "dbi_plot.svg",
    )

    # Combined 1×2 panel
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    # Silhouette
    axes[0].plot(metrics["k"], metrics["Silhouette"], marker="o", color="black", linewidth=0.8)
    axes[0].axvline(best_k_sil, linestyle="--", color="red", linewidth=0.8)
    axes[0].set_title("Silhouette", fontsize=13)
    axes[0].set_xlabel("k", fontsize=12)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_xticks(metrics["k"])
    axes[0].tick_params(labelsize=11)

    # DBI
    axes[1].plot(metrics["k"], metrics["DBI"], marker="o", color="black", linewidth=0.8)
    axes[1].axvline(best_k_dbi, linestyle="--", color="red", linewidth=0.8)
    axes[1].set_title("Davies–Bouldin Index", fontsize=13)
    axes[1].set_xlabel("k", fontsize=12)
    axes[1].set_ylabel("Index (lower is better)", fontsize=12)
    axes[1].set_xticks(metrics["k"])
    axes[1].tick_params(labelsize=11)

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    combined_png = out_dir / "clustering_validation.png"
    combined_svg = out_dir / "clustering_validation.svg"
    fig.savefig(combined_png, dpi=300, bbox_inches="tight")
    fig.savefig(combined_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Saved combined plots -> {combined_png} / {combined_svg}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
