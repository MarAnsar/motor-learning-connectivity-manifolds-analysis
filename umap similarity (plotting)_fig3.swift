#!/usr/bin/env python3
"""
UMAP similarity plot from a group-average correlation/similarity matrix.

"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP


# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="UMAP similarity plot from a labeled square TSV matrix.")
    ap.add_argument("--matrix", required=True,
                    help="Path to group-average correlation/similarity matrix (TSV, with index column).")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write figures (created if missing).")
    ap.add_argument("--basename", default="group_UMAP_embedding",
                    help="Base filename for outputs (default: group_UMAP_embedding).")

    # UMAP params (kept from your original)
    ap.add_argument("--neighbors", type=int, default=5, help="UMAP n_neighbors (default: 5).")
    ap.add_argument("--min-dist", type=float, default=0.3, help="UMAP min_dist (default: 0.3).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # Plot styling
    ap.add_argument("--label-fontsize", type=int, default=15, help="Inline text label font size (default: 15).")
    ap.add_argument("--point-size", type=float, default=120, help="Point size (default: 120).")
    ap.add_argument("--figsize", type=float, nargs=2, default=(10, 6), help="Figure size W H (default: 10 6).")
    ap.add_argument("--title", default="UMAP Projection: AMN–SMN Eccentricity",
                    help="Plot title (default: 'UMAP Projection: AMN–SMN Eccentricity').")
    ap.add_argument("--no-text-labels", action="store_true",
                    help="Disable drawing text labels next to points.")
    return ap.parse_args()


# -----------------------
# Style maps
# -----------------------

DISPLAY_ORDER = [
    "Base Day1", "Early Day1", "Late Day1", "Washout Day1",
    "Base Day2", "Early Day2", "Late Day2", "Washout Day2",
]

STYLE = {
    ("ses-01", "base"):    ("forestgreen",  "o"),
    ("ses-01", "early"):   ("gold",         "o"),
    ("ses-01", "late"):    ("darkorange",   "o"),
    ("ses-01", "washout"): ("firebrick",    "o"),
    ("ses-02", "base"):    ("grey",         "s"),
    ("ses-02", "early"):   ("black",        "s"),
    ("ses-02", "late"):    ("lightpink",    "s"),
    ("ses-02", "washout"): ("mediumpurple", "s"),
}

DISPLAY_NAME = {
    ("ses-01", "base"):    "Base Day1",
    ("ses-01", "early"):   "Early Day1",
    ("ses-01", "late"):    "Late Day1",
    ("ses-01", "washout"): "Washout Day1",
    ("ses-02", "base"):    "Base Day2",
    ("ses-02", "early"):   "Early Day2",
    ("ses-02", "late"):    "Late Day2",
    ("ses-02", "washout"): "Washout Day2",
}


# -----------------------
# Label parsing
# -----------------------

def parse_label(raw: str):
    """
    Accepts: 'Base_Day1', 'Baseline Day1', 'Early_Day2', 'Late_Day1',
             'Washout_Day2', 'D1_Baseline', 'Day2_Early', etc.
    Returns ('ses-01'|'ses-02', 'base'|'early'|'late'|'washout') or None.
    """
    s = raw.strip().lower().replace(" ", "")
    parts = re.split(r"[_\-]+", s)

    def norm_epoch(tok):
        if tok in ("base", "baseline", "bas"): return "base"
        if tok in ("early", "ear"):            return "early"
        if tok == "late":                      return "late"
        if tok in ("washout", "wash", "wo"):   return "washout"
        return None

    def norm_sess(tok):
        if tok in ("d1","day1","ses01","ses-01","ses_01","session1","ses1"):
            return "ses-01"
        if tok in ("d2","day2","ses02","ses-02","ses_02","session2","ses2"):
            return "ses-02"
        return None

    epoch = None
    sess  = None

    if len(parts) == 2:
        a, b = parts
        e1, s1 = norm_epoch(a), norm_sess(b)
        e2, s2 = norm_epoch(b), norm_sess(a)
        if e1 and s1:
            epoch, sess = e1, s1
        elif e2 and s2:
            epoch, sess = e2, s2

    if epoch is None or sess is None:
        if any(t in s for t in ("d1","day1","ses01","ses-01","ses_01","session1","ses1")): sess = "ses-01"
        if any(t in s for t in ("d2","day2","ses02","ses-02","ses_02","session2","ses2")): sess = "ses-02"
        for tok in ("baseline","base","early","late","washout","wash","wo"):
            if tok in s:
                epoch = norm_epoch(tok)
                break

    if epoch and sess:
        return (sess, epoch)
    return None


# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_base = out_dir / args.basename

    # Load matrix
    df = pd.read_csv(args.matrix, sep="\t", index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Input matrix must be square. Got {df.shape}.")

    labels = df.index.tolist()

    # UMAP
    reducer = UMAP(
        n_components=2,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        random_state=args.seed
    )
    embedding = reducer.fit_transform(df.values)

    # Plot
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    seen = set()
    for i, raw_label in enumerate(labels):
        key = parse_label(raw_label)
        if key in STYLE:
            color, marker = STYLE[key]
            disp = DISPLAY_NAME[key]
        else:
            color, marker = ("gray", "o")
            disp = raw_label.replace("_", " ")

        ax.scatter(
            embedding[i, 0], embedding[i, 1],
            s=args.point_size, marker=marker,
            color=color,
            label=disp if disp not in seen else None
        )
        if not args.no_text_labels:
            ax.text(embedding[i, 0] + 0.05, embedding[i, 1], disp, fontsize=args.label_fontsize)
        seen.add(disp)

    # Labels, title, legend
    ax.set_xlabel("Dimension 1", fontsize=16)
    ax.set_ylabel("Dimension 2", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(args.title, fontsize=18, pad=20)

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ordered_handles = [by_label[l] for l in DISPLAY_ORDER if l in by_label]
    ordered_labels  = [l for l in DISPLAY_ORDER if l in by_label]
    if ordered_handles:
        ax.legend(
            ordered_handles, ordered_labels,
            loc="upper left", bbox_to_anchor=(1.25, 1.05),
            fontsize=12, frameon=False
        )

    sns.despine()
    plt.tight_layout()

    # Save
    png_path = f"{output_base}.png"
    svg_path = f"{output_base}.svg"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("✅ UMAP plot saved:")
    print("  ", png_path)
    print("  ", svg_path)


if __name__ == "__main__":
    main()
