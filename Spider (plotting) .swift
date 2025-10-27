#!/usr/bin/env python3
"""
Spider (radar) plot of average T-values across networks
for a chosen contrast, computed from per-network TSVs.

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Spider plot of average T-values per network for a chosen contrast.")
    p.add_argument(
        "--networks-dir",
        required=True,
        help="Directory containing per-network TSVs (e.g., Visual.tsv, Somatomotor.tsv, DorsalAttention.tsv, etc.).",
    )
    p.add_argument(
        "--contrast",
        required=True,
        help='Contrast to extract (e.g., "Washout > Late"). Matching is case-insensitive and space/char tolerant.',
    )
    p.add_argument(
        "--seed",
        default="7Networks_LH_SomMot_XX",
        help='Seed label to include in the output filename (e.g., "7Networks_LH_SomMot_XX").',
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write the spider plot (PNG and SVG).",
    )
    return p.parse_args()


# ---------------------------
# Data & utils
# ---------------------------

# Canonical display order (x-axis of the radar)
DISPLAY_LABELS = [
    "Visual",
    "Somatomotor",
    "Dorsal Attn",
    "AMN",               # Salience/Ventral Attention
    "Limbic",
    "Control",
    "Default Mode",
]

# Map display name -> expected TSV filename in --networks-dir
DISPLAY_TO_FILE = {
    "Visual":        "Visual.tsv",
    "Somatomotor":   "Somatomotor.tsv",
    "Dorsal Attn":   "DorsalAttention.tsv",
    "AMN":           "SalienceVentralAttention.tsv",
    "Limbic":        "Limbic.tsv",
    "Control":       "Control.tsv",
    "Default Mode":  "Default.tsv",
}

def _norm(s: str) -> str:
    """Loosened normalization to match contrast names robustly."""
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "")
        .replace(">", "")
        .replace("-", "")
        .replace("_", "")
    )

def mean_t_for_contrast(tsv_path: str, contrast: str) -> float | None:
    """Return mean T Value for rows matching `contrast` in the given TSV."""
    if not os.path.exists(tsv_path):
        print(f"[warn] Missing file: {tsv_path}")
        return None

    df = pd.read_csv(tsv_path, sep="\t")
    if "Epochs" not in df.columns or "T Value" not in df.columns:
        print(f"[warn] {tsv_path} missing required columns 'Epochs' and/or 'T Value'.")
        return None

    # robust match (case/space/symbol-insensitive)
    want = _norm(contrast)
    mask = df["Epochs"].map(lambda x: _norm(x) == want)

    sel = df.loc[mask, "T Value"]
    if sel.empty:
        print(f"[warn] No rows for contrast '{contrast}' in {tsv_path}.")
        return None

    return float(sel.mean())


# ---------------------------
# Plotting (radar/spider)
# ---------------------------

def spider_plot(labels: List[str], values: List[float], title: str, out_png: str, out_svg: str):
    """
    Create a radar/spider plot with styling similar to the provided example.
    - Rings at [-4, -2, 0, 2, 4] with bold black ring at 0
    - White radial lines
    - Network labels outside the circle
    """
    # Close the loop
    vals = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("lightgray")

    # scale
    min_val, max_val = -4, 4
    ax.set_ylim(min_val, max_val)

    # remove default grid
    ax.grid(False)

    # rings & zero ring
    visible_yticks = [-4, -2, 0, 2, 4]
    for y in visible_yticks:
        ax.plot(np.linspace(0, 2 * np.pi, 500), [y] * 500, color="white", linewidth=3.5, zorder=1)
    ax.plot(np.linspace(0, 2 * np.pi, 500), [0] * 500, color="black", linewidth=2.5, zorder=2)

    # radial lines
    for angle in angles[:-1]:
        ax.plot([angle, angle], [min_val, max_val], color="white", linewidth=1.5, zorder=1)

    # polar spine
    ax.spines["polar"].set_color("white")
    ax.spines["polar"].set_linewidth(3.5)

    # y tick labels hidden; we draw our own small labels
    ax.set_yticks(visible_yticks)
    ax.set_yticklabels([])
    label_angle_rad = np.deg2rad(15)
    for y in visible_yticks:
        ax.text(label_angle_rad, y, f"{y:.0f}", ha="left", va="center", fontsize=18, color="black", zorder=5)

    # x (network) labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    label_radius = max_val + 0.4
    for ang, lab in zip(angles[:-1], labels):
        ang_deg = np.degrees(ang)
        rot = ang_deg + 90 if 90 < ang_deg < 270 else ang_deg - 90
        # flip a few labels for nicer orientation
        if lab in ["AMN", "Dorsal Attn", "Default Mode"]:
            rot = (rot + 180) % 360
        ax.text(ang, label_radius, lab, size=20, rotation=rot,
                rotation_mode="anchor", ha="center", va="center",
                color="black", zorder=6)

    # polygon (steelblue)
    ax.fill(angles, vals, color="#4682B4", alpha=0.25, zorder=11)
    ax.plot(angles, vals, color="#4682B4", linewidth=2.5, zorder=12)

    # title
    ax.set_title(title, pad=20, fontsize=18)

    # save
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect mean T per network for the chosen contrast
    values_by_display: Dict[str, float] = {}
    for disp_label in DISPLAY_LABELS:
        fname = DISPLAY_TO_FILE[disp_label]
        fpath = os.path.join(args.networks_dir, fname)
        m = mean_t_for_contrast(fpath, args.contrast)
        if m is None:
            print(f"[warn] Using NaN for {disp_label}.")
            values_by_display[disp_label] = np.nan
        else:
            values_by_display[disp_label] = m

    # Assemble in display order
    values = [values_by_display[l] for l in DISPLAY_LABELS]

    # Build plot
    base = f"network_{args.seed}_{args.contrast.replace(' ', '').replace('>', '_gt_').replace('/', '-')}"
    out_png = os.path.join(args.out_dir, f"{base}.png")
    out_svg = os.path.join(args.out_dir, f"{base}.svg")
    title = f"Average T (contrast: {args.contrast})"
    spider_plot(DISPLAY_LABELS, values, title, out_png, out_svg)

    print("✅ Saved spider plot:")
    print("  ", out_png)
    print("  ", out_svg)


if __name__ == "__main__":
    sys.exit(main())
