#!/usr/bin/env python3
"""
Plot ANOVA F-statistics on cortical surfaces (Schaefer-400).

"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")           # headless matplotlib
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")  # headless pyvista
os.environ.setdefault("PYVISTA_ENABLE_TRAME", "false")
os.environ.setdefault("PYVISTA_USE_IPYVTK", "false")
os.environ.setdefault("SURFPLOT_ENGINE", "pyvista")  # surfplot engine hint

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from surfplot import Plot
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


def parse_args():
    ap = argparse.ArgumentParser(
        description="Map per-parcel F-values (Schaefer-400) onto cortical surfaces and save figure + colorbar."
    )
    ap.add_argument("--input", required=True,
                    help="TSV with columns 'Region' (1..400) and 'F'.")
    ap.add_argument("--atlas", required=True,
                    help="CIFTI .dlabel.nii with parcel IDs per vertex (Schaefer-400).")
    ap.add_argument("--surf-lh", required=True,
                    help="Left hemisphere surface (e.g., *.surf.gii).")
    ap.add_argument("--surf-rh", required=True,
                    help="Right hemisphere surface (e.g., *.surf.gii).")
    ap.add_argument("--out-dir", default="outputs/surface_maps",
                    help="Directory to write images.")
    ap.add_argument("--prefix", default="anova_fstats",
                    help="Filename prefix for outputs.")
    ap.add_argument("--vmin", type=float, default=None,
                    help="Fixed lower bound for color range (default: min F).")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Fixed upper bound for color range (default: max F).")
    return ap.parse_args()


def weights_to_vertices(parc_vals: np.ndarray, dlabel_path: str, n_parcels: int = 400) -> np.ndarray:
    """
    Map parcel-wise values (length n_parcels; index i -> parcel ID i+1) to vertices
    using a CIFTI .dlabel file that encodes parcel IDs per vertex.

    Returns an array (n_vertices,) of floats with NaN for unlabeled vertices.
    """
    lab = nib.load(dlabel_path).get_fdata().ravel()
    # Some dlabels encode IDs as float; cast safely.
    lab_ids = lab.astype(int)

    # Initialize with NaNs (rendered as background)
    mapped = np.full(lab_ids.shape, np.nan, dtype=float)

    # Valid parcels: 1..n_parcels
    valid = (lab_ids >= 1) & (lab_ids <= n_parcels)
    idx = lab_ids[valid] - 1  # 0-based indexing into parcel array
    mapped[valid] = parc_vals[idx]
    return mapped


def build_cmap():
    """High-contrast warm colormap (low=red → high=bright yellow)."""
    high_contrast_colors = [
        "#ff0000", "#ff2a00", "#ff5500", "#ff8000", "#ffab00",
        "#ffd000", "#ffee00", "#ffff33", "#ffff66", "#ffff99",
    ]
    return LinearSegmentedColormap.from_list("highlighter_contrast", high_contrast_colors, N=256)


def plot_surfaces(mapped_vals: np.ndarray, lh_path: str, rh_path: str,
                  vmin: float, vmax: float, out_png: str):
    """
    Render a brain surface with surfplot (no colorbar embedded).
    """
    cmap = build_cmap()
    clipped = np.clip(mapped_vals, vmin, vmax)

    p = Plot(lh_path, rh_path)
    p.add_layer(clipped, cmap=cmap, color_range=(vmin, vmax), cbar=False)
    fig = p.build(colorbar=False)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_colorbar(vmin: float, vmax: float, out_png: str):
    """
    Save a standalone horizontal colorbar that matches the surface map.
    """
    cmap = build_cmap()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(7.5, 1.2))
    ax = fig.add_axes([0.08, 0.4, 0.84, 0.35])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax, orientation="horizontal")
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    cbar.set_label("F-value", fontsize=12)
    cbar.outline.set_visible(False)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load TSV with Region/F ---
    df = pd.read_csv(args.input, sep="\t")
    if "Region" not in df.columns or "F" not in df.columns:
        raise ValueError("Input TSV must contain columns 'Region' and 'F'.")

    # Build parcel array length 400 (NaN for missing parcels)
    parc_vals = np.full(400, np.nan, dtype=float)
    for rid, fval in zip(df["Region"], df["F"]):
        if pd.notna(rid):
            idx = int(rid) - 1
            if 0 <= idx < 400:
                parc_vals[idx] = float(fval)

    # Determine color range
    finite_vals = parc_vals[np.isfinite(parc_vals)]
    if finite_vals.size == 0:
        raise ValueError("No finite F-values found in input.")
    vmin = float(np.min(finite_vals)) if args.vmin is None else args.vmin
    vmax = float(np.max(finite_vals)) if args.vmax is None else args.vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError(f"Invalid color range vmin={vmin}, vmax={vmax}.")

    # Map parcels → vertices
    mapped = weights_to_vertices(parc_vals, args.atlas, n_parcels=400)

    # Save surface figure (no colorbar) and a separate colorbar
    out_fig = os.path.join(args.out_dir, f"{args.prefix}_fmap.png")
    out_cbar = os.path.join(args.out_dir, f"{args.prefix}_cbar.png")

    plot_surfaces(mapped, args.surf_lh, args.surf_rh, vmin, vmax, out_fig)
    plot_colorbar(vmin, vmax, out_cbar)

    print("✅ Saved:")
    print("  Surface :", out_fig)
    print("  Colorbar:", out_cbar)


if __name__ == "__main__":
    main()
