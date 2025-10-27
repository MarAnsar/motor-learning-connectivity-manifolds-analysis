#!/usr/bin/env python3
"""
Render eccentricity maps on cortical surfaces (Schaefer 400, 7-network order).

"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")           # headless-safe
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")  # avoid GUI
os.environ.setdefault("PYVISTA_ENABLE_TRAME", "false")
os.environ.setdefault("PYVISTA_USE_IPYVTK", "false")
os.environ.setdefault("SURFPLOT_ENGINE", "pyvista")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from surfplot import Plot

# Uses plotting.weights_to_vertices
from adaptman.analyses import plotting


# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot eccentricity (BaseAvg) on cortical surfaces (Schaefer 400)."
    )
    ap.add_argument("--atlas", required=True,
                    help="Path to Schaefer2018_400Parcels_7Networks_order.dlabel.nii")
    ap.add_argument("--surf-lh", required=True,
                    help="Path to LH GIFTI surface (e.g., S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii)")
    ap.add_argument("--surf-rh", required=True,
                    help="Path to RH GIFTI surface (e.g., S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii)")
    ap.add_argument("--input-tsv", required=True,
                    help="TSV with columns ['Region','BaseAvg'] (Region in 1..400).")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to save figures (created if missing).")

    ap.add_argument("--cmap", default="auto",
                    help="Matplotlib colormap name, or 'auto' for a 4-color gradient.")
    ap.add_argument("--vmin", type=float, default=None,
                    help="Colorbar lower bound (default: min of data).")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Colorbar upper bound (default: max of data).")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Output DPI (default: 300).")
    return ap.parse_args()


# -----------------------
# Core
# -----------------------

VIEWS = {
    "lateral":   {"size": (800, 200), "zoom": 1.2},
    "medial":    {"size": (800, 200), "zoom": 1.2},
    "dorsal":    {"size": (150, 200), "zoom": 3.3},
    "posterior": {"size": (150, 200), "zoom": 3.3},
}

def build_cmap(name: str):
    """Return a colormap. 'auto' = the original 4-color gradient."""
    if name != "auto":
        return plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        "ecc_cmap",
        ["#440154", "#31688e", "#35b779", "#fde725"],
        N=256
    )

def load_eccentricity_table(path: Path) -> np.ndarray:
    """Load TSV with columns ['Region','BaseAvg'] into a length-400 array (NaN where missing)."""
    df = pd.read_csv(path, sep="\t")
    if not {"Region", "BaseAvg"}.issubset(df.columns):
        raise ValueError("Input TSV must contain columns ['Region','BaseAvg'].")

    vals = np.full(400, np.nan, dtype=float)
    for region, value in zip(df["Region"].to_numpy(), df["BaseAvg"].to_numpy()):
        idx = int(region) - 1
        if idx < 0 or idx >= 400:
            continue
        vals[idx] = float(value)
    return vals

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    tvals = load_eccentricity_table(Path(args.input_tsv))

    # 2) Determine color range
    data_vmin = np.nanmin(tvals)
    data_vmax = np.nanmax(tvals)
    vmin = data_vmin if args.vmin is None else args.vmin
    vmax = data_vmax if args.vmax is None else args.vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("vmin/vmax could not be determined from data. Check input TSV.")

    # 3) Project parcels -> vertices
    vertices = plotting.weights_to_vertices(tvals, args.atlas)

    # 4) Colormap
    cmap = build_cmap(args.cmap)

    # 5) Plot each view
    for view, params in VIEWS.items():
        p = Plot(
            args.surf_lh, args.surf_rh,
            views=view, layout="row", mirror_views=True,
            size=params["size"], zoom=params["zoom"]
        )
        # add layer with colorbar (we’ll format ticks after building)
        p.add_layer(vertices, cmap=cmap, color_range=(vmin, vmax), cbar=True)
        fig = p.build()

        # format colorbar ticks/label (last axis is the colorbar in surfplot)
        cbar_ax = fig.axes[-1]
        cbar_ax.set_xticks([vmin, vmax])
        cbar_ax.set_xticklabels([f"{vmin:.2f}", f"{vmax:.2f}"], fontsize=12)
        cbar_ax.set_xlabel("Ecc.", fontsize=13, fontweight="bold", labelpad=10)

        # save
        out_path = out_dir / f"eccentricity_baseavg_map_{view}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"✅ Eccentricity brain maps saved to: {out_dir}")

if __name__ == "__main__":
    main()
