#!/usr/bin/env python3
"""
AMN+SMN Representational Analysis (RSM/RDM, Clustering, 3D MDS)

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import spatial, cluster
from sklearn.manifold import MDS

# Needed in some Matplotlib versions to enable 3D projection import side effects
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -----------------------------
# Helpers
# -----------------------------

def read_labeled_square_matrix(path: Path) -> pd.DataFrame:
    """Read a square matrix (CSV/TSV) with an index column for labels."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, index_col=0)
    else:
        # default to TSV for .tsv / .txt
        df = pd.read_csv(path, sep="\t", index_col=0)

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Input matrix must be square; got {df.shape}.")
    return df


def save_fig(fig: plt.Figure, out_png: Path, out_svg: Path, dpi: int = 300) -> None:
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_svg, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main analysis steps
# -----------------------------

def plot_rsm(df: pd.DataFrame, title: str, out_base: Path, cmap: str, annot: bool, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df, ax=ax, cmap=cmap, square=True, annot=annot, fmt=fmt)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="x", labelsize=12, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    save_fig(fig, out_base.with_name(out_base.name + "_RSM_heatmap.png"),
                  out_base.with_name(out_base.name + "_RSM_heatmap.svg"))


def plot_rdm(rdm: pd.DataFrame, title: str, out_base: Path, cmap: str, annot: bool, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(rdm, ax=ax, cmap=cmap, square=True, annot=annot, fmt=fmt)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="x", labelsize=12, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    save_fig(fig, out_base.with_name(out_base.name + "_RDM_heatmap.png"),
                  out_base.with_name(out_base.name + "_RDM_heatmap.svg"))


def cluster_and_plot(rdm: pd.DataFrame, conditions: list[str], out_base: Path, cmap: str) -> None:
    # condensed distances for linkage
    dvec = spatial.distance.squareform(rdm.values, checks=False)
    Z = cluster.hierarchy.linkage(dvec, method="average")

    # Dendrogram
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    cluster.hierarchy.dendrogram(Z, labels=conditions, leaf_rotation=45, leaf_font_size=10, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=14)
    fig.tight_layout()
    save_fig(fig, out_base.with_name(out_base.name + "_dendrogram.png"),
                  out_base.with_name(out_base.name + "_dendrogram.svg"))

    # Clustermap
    g = sns.clustermap(
        rdm, cmap=cmap, row_linkage=Z, col_linkage=Z,
        figsize=(8, 8), square=True
    )
    # clustermap has its own Figure
    g.savefig(out_base.with_name(out_base.name + "_clustermap.png"), dpi=300, bbox_inches="tight")
    g.savefig(out_base.with_name(out_base.name + "_clustermap.svg"), dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def mds_3d_plot(rdm: pd.DataFrame, conditions: list[str], out_base: Path, random_state: int) -> None:
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=random_state)
    coords = mds.fit_transform(rdm.values)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # color by index just for visual separation (categorical colormap)
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    s=100, c=np.arange(len(conditions)), cmap="viridis", alpha=0.9)

    for i, lab in enumerate(conditions):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], lab, fontsize=9)

    ax.set_title("3D MDS of Condition Dissimilarities", fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_box_aspect([1, 1, 1])

    fig.tight_layout()
    save_fig(fig, out_base.with_name(out_base.name + "_MDS_3D.png"),
                  out_base.with_name(out_base.name + "_MDS_3D.svg"))


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="AMN+SMN RSA: RSM/RDM heatmaps, clustering, and 3D MDS."
    )
    ap.add_argument("--matrix", required=True,
                    help="Path to group correlation matrix (.csv or .tsv) with index column for labels.")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write outputs.")
    ap.add_argument("--basename", default="amn_smn_group",
                    help="Base name for output files (default: amn_smn_group).")

    ap.add_argument("--cmap", default="viridis", help="Matplotlib/Seaborn colormap name (default: viridis).")
    ap.add_argument("--annot", action="store_true", help="Annotate heatmaps with numeric values.")
    ap.add_argument("--fmt", default=".2f", help="Annotation format (default: .2f).")

    ap.add_argument("--mds-seed", type=int, default=42, help="Random seed for MDS (default: 42).")
    return ap.parse_args()


def main():
    args = parse_args()

    matrix_path = Path(args.matrix)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_base = out_dir / args.basename

    # 1) Load correlation matrix (RSM)
    rsm = read_labeled_square_matrix(matrix_path)
    conditions = rsm.index.tolist()

    # 2) Plot/save RSM
    plot_rsm(rsm, title="AMN+SMN Ensemble: Correlation Matrix (RSM)",
             out_base=out_base, cmap=args.cmap, annot=args.annot, fmt=args.fmt)

    # 3) Compute and plot RDM
    rdm = 1.0 - rsm
    plot_rdm(rdm, title="AMN+SMN Ensemble: Dissimilarity Matrix (RDM = 1 - R)",
             out_base=out_base, cmap=args.cmap, annot=args.annot, fmt=args.fmt)

    # Also save the RDM table for convenience
    rdm_out = out_base.with_name(out_base.name + "_RDM.tsv")
    rdm.to_csv(rdm_out, sep="\t")

    # 4) Clustering
    cluster_and_plot(rdm, conditions, out_base, cmap=args.cmap)

    # 5) 3D MDS
    mds_3d_plot(rdm, conditions, out_base, random_state=args.mds_seed)

    print("✅ Done. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
