#!/usr/bin/env python3
"""
3D scatter of reference gradients (PC1–PC3) with network-based coloring.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Needed in some Matplotlib versions to enable 3D projection import side effects
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -----------------------------
# Helpers
# -----------------------------

def network_color(label: str, alpha: float = 0.65) -> tuple[float, float, float, float]:
    """Map an ROI label to an RGBA color (by substring rules)."""
    if "SomMot" in label:
        return (70/255, 130/255, 180/255, alpha)     # steelblue
    if "SalVentAttn" in label:
        return (218/255, 112/255, 214/255, alpha)    # orchid
    return (0.5, 0.5, 0.5, alpha)                    # gray


def load_gradients(path: Path) -> pd.DataFrame:
    """Read gradients TSV; ensure required columns exist."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    required = {"g1", "g2", "g3"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Gradient file must contain columns {sorted(required)}. "
            f"Found: {list(df.columns)}"
        )
    if df.index.name is None:
        # not strictly required, but labels are used for coloring; warn if missing
        print("[warn] Input TSV has no index name; assuming row index are ROI labels.")
    return df


def plot_3d(
    df: pd.DataFrame,
    out_dir: Path,
    alpha: float = 0.65,
    svg_name: str = "manifold_3Dplot.svg",
    png_name: str = "manifold_3Dplot.png",
    dpi: int = 300,
) -> None:
    """Render and save the 3D scatter (SVG + PNG)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Colors per ROI label
    colors = [network_color(str(lbl), alpha=alpha) for lbl in df.index]

    # Figure
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter
    ax.scatter(df["g1"], df["g2"], df["g3"], c=colors, s=60)

    # Axis labels with equal padding
    equal_labelpad = 15
    ax.set_xlabel("PC1", fontsize=20, labelpad=equal_labelpad)
    ax.set_ylabel("PC2", fontsize=20, labelpad=equal_labelpad)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("PC3", fontsize=20, labelpad=equal_labelpad)

    # Denser integer-ish ticks (~10 per axis)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="z", labelsize=18)

    # View angle
    ax.view_init(elev=20, azim=-120)

    # Margins so 3D labels aren’t cropped (avoid tight_layout for 3D)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.98)

    # Save
    svg_path = out_dir / svg_name
    png_path = out_dir / png_name
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"✅ Saved:\n  {svg_path}\n  {png_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="3D PC1–PC3 gradient plot with network-based coloring."
    )
    ap.add_argument("--grad-path", required=True,
                    help="Path to reference_gradient.tsv (columns: g1, g2, g3).")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to save the output figures.")
    ap.add_argument("--alpha", type=float, default=0.65,
                    help="Point alpha (transparency), default 0.65.")
    ap.add_argument("--dpi", type=int, default=300,
                    help="PNG DPI, default 300.")
    ap.add_argument("--svg-name", default="manifold_3Dplot.svg",
                    help="Output SVG filename (default: manifold_3Dplot.svg).")
    ap.add_argument("--png-name", default="manifold_3Dplot.png",
                    help="Output PNG filename (default: manifold_3Dplot.png).")
    return ap.parse_args()


def main():
    args = parse_args()
    grad_path = Path(args.grad_path)
    out_dir = Path(args.out_dir)

    df = load_gradients(grad_path)
    plot_3d(
        df=df,
        out_dir=out_dir,
        alpha=args.alpha,
        svg_name=args.svg_name,
        png_name=args.png_name,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
