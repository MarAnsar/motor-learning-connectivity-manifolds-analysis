#!/usr/bin/env python3
"""
Epoch-wise eccentricity summary charts for k=2 clusters.

For each cluster (1 and 2), this script:
  • Reads per-epoch TSV files (Base/Early/Late/Washout for Day 1 and Day 2).
  • For each epoch file:
      - Computes a per-subject mean (average across regions for that subject).
      - Computes a single "connecting value" = average across subjects.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Plot style knobs (kept subtle/tidy) ────────────────────────────────────────
SCATTER_SIZE = 12    # small dots
LINE_WIDTH   = 0.5   # thin connecting lines
SPINE_WIDTH  = 0.5   # thin frame
MARKER_SIZE  = 3.0   # small markers on connecting lines
COLOR_DAY1   = "#467137"  # green
COLOR_DAY2   = "#F7931E"  # orange
FRAME_COLOR1 = "#ff2800"  # frame color for cluster 1
FRAME_COLOR2 = "#00F0FF"  # frame color for cluster 2


def parse_args():
    ap = argparse.ArgumentParser(description="Make epoch-wise connecting charts for k=2 clusters.")
    ap.add_argument(
        "--base-path",
        default="PATH/TO/your/average4regions",
        help="Directory containing 8 epoch TSVs per cluster (see docstring).",
    )
    ap.add_argument(
        "--out-dir",
        default="PATH/TO/output/figures",
        help="Directory to save the output figures.",
    )
    ap.add_argument(
        "--group1-suffix",
        default="group1.tsv",
        help="Filename suffix for Cluster 1 files (default: 'group1.tsv').",
    )
    ap.add_argument(
        "--group2-suffix",
        default="group2.tsv",
        help="Filename suffix for Cluster 2 files (default: 'group2.tsv').",
    )
    return ap.parse_args()


# Filenames by epoch (stem only; cluster suffix is appended)
EPOCHS_DAY1 = ["Base_Day1", "Early_Day1", "Late_Day1", "Washout_Day1"]
EPOCHS_DAY2 = ["Base_Day2", "Early_Day2", "Late_Day2", "Washout_Day2"]
X_LABELS    = ["Base", "Early", "Late", "Washout", "Base", "Early", "Late", "Washout"]


def load_epoch_subject_means(tsv_path: str) -> np.ndarray:
    """
    Returns a 1D array of per-subject means for the given epoch file.

    Expected TSV columns:
      'Region', sub-01, sub-02, ...

    We compute:
      1) per subject mean across regions (column-wise mean of numeric columns)
      2) return that vector (length = #subjects)
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    df = pd.read_csv(tsv_path, sep="\t")
    if "Region" not in df.columns:
        raise ValueError(f"'Region' column not found in: {tsv_path}")

    numeric = df.drop(columns=["Region"], errors="ignore")

    # Keep only numeric subject columns
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    # Drop any all-NaN columns (e.g., if a subject is missing entirely)
    numeric = numeric.dropna(axis=1, how="all")

    # Per-subject mean across regions
    subj_means = numeric.mean(axis=0).to_numpy(dtype=float)  # shape: (n_subjects,)
    # Drop NaNs if any remain (subjects with all-NaN after coercion)
    subj_means = subj_means[~np.isnan(subj_means)]
    return subj_means


def compute_connecting_values(base_path: str, group_suffix: str):
    """
    For the 8 epoch files in a cluster, return:
      - subj_means_all: dict mapping epoch-stem -> 1D array of per-subject means
      - connecting_day1: list of 4 scalars (avg of subject means for Day1 epochs)
      - connecting_day2: list of 4 scalars (avg of subject means for Day2 epochs)
    """
    subj_means_all = {}
    connecting_day1 = []
    connecting_day2 = []

    for stem in EPOCHS_DAY1 + EPOCHS_DAY2:
        fname = f"{stem}_{group_suffix}"
        fpath = os.path.join(base_path, fname)
        subj_means = load_epoch_subject_means(fpath)
        subj_means_all[stem] = subj_means

    # Day 1 (order matters for plotting positions 1..4)
    for stem in EPOCHS_DAY1:
        connecting_day1.append(float(np.mean(subj_means_all[stem])))

    # Day 2 (positions 5..8)
    for stem in EPOCHS_DAY2:
        connecting_day2.append(float(np.mean(subj_means_all[stem])))

    return subj_means_all, connecting_day1, connecting_day2


def plot_cluster(
    cluster_id: int,
    subj_means_all: dict,
    connecting_day1: list,
    connecting_day2: list,
    out_dir: str,
):
    """
    Make the epoch-wise chart for one cluster.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect all jitter points to define y-limits
    all_points = []
    for stem in (EPOCHS_DAY1 + EPOCHS_DAY2):
        all_points.extend(subj_means_all[stem].tolist())
    if not all_points:
        raise RuntimeError("No data points to plot.")

    y_min, y_max = float(np.min(all_points)), float(np.max(all_points))
    # Build four evenly spaced y ticks
    y_ticks = [y_min + i * (y_max - y_min) / 4.0 for i in range(5)]

    # Begin plotting
    plt.figure(figsize=(12, 6))

    # Scatter Day 1 (green) and Day 2 (orange) with light jitter
    for idx, stem in enumerate(EPOCHS_DAY1 + EPOCHS_DAY2, start=1):
        vals = subj_means_all[stem]
        jitter = np.random.normal(0, 0.10, size=vals.shape[0])
        xs = np.full_like(vals, idx, dtype=float) + jitter
        color = COLOR_DAY1 if stem in EPOCHS_DAY1 else COLOR_DAY2
        plt.scatter(xs, vals, alpha=0.6, color=color, s=SCATTER_SIZE)

    # Connecting lines (thin black with small white markers)
    plt.plot(
        range(1, 5), connecting_day1,
        color="black", marker="o", markerfacecolor="white",
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE
    )
    plt.plot(
        range(5, 9), connecting_day2,
        color="black", marker="o", markerfacecolor="white",
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE
    )

    # Axes & labels
    plt.xticks(range(1, 9), X_LABELS, rotation=30, ha="right", fontsize=16)
    plt.yticks(y_ticks, [f"{y:.2f}" for y in y_ticks], fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Eccentricity", fontsize=16)
    plt.title(f"Eccentricity Changes Across Epochs (Cluster {cluster_id})", fontsize=18)
    plt.grid(alpha=0.3)

    # Frame color per cluster
    ax = plt.gca()
    frame_color = FRAME_COLOR1 if cluster_id == 1 else FRAME_COLOR2
    for spine in ax.spines.values():
        spine.set_color(frame_color)
        spine.set_linewidth(SPINE_WIDTH)

    # Save
    png_path = os.path.join(out_dir, f"epoch_chart_group{cluster_id}.png")
    svg_path = os.path.join(out_dir, f"epoch_chart_group{cluster_id}.svg")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[cluster {cluster_id}] saved:\n  {png_path}\n  {svg_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Cluster 1
    g1_suffix = args.group1_suffix  # e.g., "group1.tsv"
    subj_means_all, conn_d1, conn_d2 = compute_connecting_values(args.base_path, g1_suffix)
    plot_cluster(1, subj_means_all, conn_d1, conn_d2, args.out_dir)

    # Cluster 2
    g2_suffix = args.group2_suffix  # e.g., "group2.tsv"
    subj_means_all, conn_d1, conn_d2 = compute_connecting_values(args.base_path, g2_suffix)
    plot_cluster(2, subj_means_all, conn_d1, conn_d2, args.out_dir)


if __name__ == "__main__":
    main()
