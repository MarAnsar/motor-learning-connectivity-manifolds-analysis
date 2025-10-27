#!/usr/bin/env python3
"""
Brain-surface maps of seed-based pairwise t-statistics.
"""

from __future__ import annotations

import argparse
import os
from decimal import Decimal
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from surfplot import Plot
from brainspace.utils.parcellation import map_to_labels


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot seed-based pairwise t-stat maps on brain surface (no >7 highlight).")
    p.add_argument("--infile", required=True, help="Input TSV with columns: Epochs, Region (1..400), T Value.")
    p.add_argument("--atlas", required=True, help="Path to Schaefer-400 dlabel (CIFTI dense label).")
    p.add_argument("--surf-lh", required=True, help="Left hemisphere surface GIFTI.")
    p.add_argument("--surf-rh", required=True, help="Right hemisphere surface GIFTI.")
    p.add_argument("--outdir", required=True, help="Directory to save figures.")
    p.add_argument("--tmin", type=float, default=-4.0, help="Lower color limit for t-values.")
    p.add_argument("--tmax", type=float, default=4.0, help="Upper color limit for t-values.")
    p.add_argument(
        "--no-outlines",
        action="store_true",
        help="Disable AMN/SMN outline accents (default: outlines on)."
    )
    return p.parse_args()


# ---------------------------
# Styling / helpers
# ---------------------------

def make_diverging_cmap() -> LinearSegmentedColormap:
    """Blue–white–red diverging colormap."""
    stops = [
        (0.00, "#0000FF"),
        (0.125, "#3A7DFF"),
        (0.25, "#5E9CFF"),
        (0.375, "#D0E8FF"),
        (0.49, "#E8F5FF"),
        (0.50, "#FFFFFF"),
        (0.51, "#FFECE8"),
        (0.625, "#FFD6D1"),
        (0.75, "#FFAAAA"),
        (0.875, "#FF5555"),
        (1.00, "#FF0000"),
    ]
    return LinearSegmentedColormap.from_list("diverging", stops, N=256)


MAIN_CMAP = make_diverging_cmap()
BLACK_CMAP = LinearSegmentedColormap.from_list("black", ["#000000", "#000000"], N=256)

# Canonical view presets
VIEWS: Dict[str, Dict[str, object]] = {
    "lateral":   {"size": (800, 200), "zoom": 1.2},
    "medial":    {"size": (800, 200), "zoom": 1.2},
    "dorsal":    {"size": (150, 200), "zoom": 3.3},
    "ventral":   {"size": (800, 200), "zoom": 1.2},
    "posterior": {"size": (150, 200), "zoom": 3.3},
    "anterior":  {"size": (800, 200), "zoom": 1.2},
}

# Example AMN/SMN outline sets (parcel IDs, 1-based) — tweak/remove to taste.
PINK_IDS = list(range(92, 114)) + list(range(294, 319))   # e.g., AMN (Sal/VentAttn) set
BLUE_IDS = list(range(32, 69)) + list(range(231, 271))    # e.g., SMN set


def add_outline_layers(p: Plot, atlas_vec: np.ndarray, thickness: int = 50):
    """Add thick outlines for AMN/SMN example sets."""
    pink_mask = np.isin(atlas_vec, PINK_IDS).astype(float)
    blue_mask = np.isin(atlas_vec, BLUE_IDS).astype(float)
    for _ in range(thickness):
        p.add_layer(data=pink_mask, cmap=BLACK_CMAP, as_outline=True, cbar=False)
        p.add_layer(data=blue_mask, cmap=BLACK_CMAP, as_outline=True, cbar=False)


def save_all_formats(fig: plt.Figure, outpath_base: str):
    fig.savefig(outpath_base + ".png", dpi=1200, bbox_inches="tight", facecolor="white")
    fig.savefig(outpath_base + ".svg", bbox_inches="tight", facecolor="white")
    fig.savefig(outpath_base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_view(
    mapped: np.ndarray,
    label: str,
    atlas_vec: np.ndarray,
    surf_lh: str,
    surf_rh: str,
    tmin: float,
    tmax: float,
    outdir: str,
    with_outlines: bool = True,
):
    """Render each view separately and compose with a colorbar."""
    # cap to range
    data = mapped.copy()
    data[data < tmin] = tmin
    data[data > tmax] = tmax

    for view, opts in VIEWS.items():
        # draw surface (no colorbar directly)
        p = Plot(surf_lh, surf_rh, views=view, layout="row", mirror_views=True,
                 size=opts["size"], zoom=opts["zoom"])
        p.add_layer(data=data, cmap=MAIN_CMAP, color_range=(tmin, tmax), cbar=False)
        if with_outlines:
            add_outline_layers(p, atlas_vec, thickness=50)

        fig = p.build()
        tmp_png = os.path.join(outdir, f"{label}_{view}_tmp.png")
        fig.savefig(tmp_png, dpi=1200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # compose with colorbar
        fig2, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(plt.imread(tmp_png))
        ax.axis("off")

        # single colorbar
        cax = fig2.add_axes([0.15, 0.05, 0.3, 0.03])
        cb = plt.colorbar(
            plt.cm.ScalarMappable(cmap=MAIN_CMAP, norm=plt.Normalize(tmin, tmax)),
            cax=cax, orientation="horizontal"
        )
        cb.set_label("t-statistic")
        cb.set_ticks([tmin, 0, tmax])

        out_base = os.path.join(outdir, f"{label}_{view}")
        save_all_formats(fig2, out_base)
        os.remove(tmp_png)
        print(f"✔ Saved: {out_base}.png / .svg / .pdf")


def plot_4panel(
    mapped: np.ndarray,
    label: str,
    atlas_vec: np.ndarray,
    surf_lh: str,
    surf_rh: str,
    tmin: float,
    tmax: float,
    outdir: str,
    with_outlines: bool = True,
):
    """Render the default 4-view figure."""
    data = mapped.copy()
    data[data < tmin] = tmin
    data[data > tmax] = tmax

    p = Plot(surf_lh, surf_rh)
    p.add_layer(data=data, cmap=MAIN_CMAP, color_range=(tmin, tmax), cbar=False)
    if with_outlines:
        add_outline_layers(p, atlas_vec, thickness=50)

    fig = p.build()
    out_base = os.path.join(outdir, f"{label}_4views")
    save_all_formats(fig, out_base)
    print(f"✔ Saved: {out_base}.png / .svg / .pdf")


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load data
    if not os.path.isfile(args.infile):
        raise FileNotFoundError(f"--infile not found: {args.infile}")

    df = pd.read_csv(args.infile, sep="\t")
    for col in ("Epochs", "Region", "T Value"):
        if col not in df.columns:
            raise ValueError(f"Input TSV must contain column: {col}")

    # load atlas to vector (parcel IDs per vertex)
    atlas_vec = nib.load(args.atlas).get_fdata().ravel()

    # process each contrast/epoch label
    for epoch_label in df["Epochs"].unique():
        sub = df[df["Epochs"] == epoch_label].copy()

        # create a 400-length array (NaN default)
        tvals = np.full(400, np.nan)
        for rid, tv in zip(sub["Region"], sub["T Value"]):
            try:
                r = int(rid)
                if 1 <= r <= 400:
                    tvals[r - 1] = float(Decimal(str(tv)))
            except Exception:
                # skip malformed rows
                continue

        # map to vertices
        mapped = map_to_labels(tvals, atlas_vec, mask=atlas_vec > 0)

        # plots
        plot_per_view(
            mapped=mapped,
            label=epoch_label,
            atlas_vec=atlas_vec,
            surf_lh=args.surf_lh,
            surf_rh=args.surf_rh,
            tmin=args.tmin,
            tmax=args.tmax,
            outdir=args.outdir,
            with_outlines=(not args.no_outlines),
        )
        plot_4panel(
            mapped=mapped,
            label=epoch_label,
            atlas_vec=atlas_vec,
            surf_lh=args.surf_lh,
            surf_rh=args.surf_rh,
            tmin=args.tmin,
            tmax=args.tmax,
            outdir=args.outdir,
            with_outlines=(not args.no_outlines),
        )

    print("🎉 All contrasts exported (PNG/SVG/PDF).")


if __name__ == "__main__":
    main()
