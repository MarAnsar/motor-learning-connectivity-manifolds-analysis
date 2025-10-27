#!/usr/bin/env python3
"""
Plot PCA/gradient component maps on cortical surfaces.

Overview
--------
- Locate a TSV with columns: Region, g1, g2, ... (use --tsv or search dirs).
- Clean & filter rows to a fixed set of valid region IDs.
- Map region values to the 400-parcel Schaefer atlas → surface vertices.
- Render each component as:
    (1) Standard 4-panel cortex figure
    (2) Dorsal-only export (mirrored LH/RH)
- Write a shared horizontal colorbar.

Examples
--------
# Search for the TSV across multiple folders
python plot_brain_gradients.py \
  --search-dir PATH/TO/results/day1 \
  --search-dir PATH/TO/results/day2 \
  --atlas PATH/TO/Schaefer2018_400Parcels_7Networks_order.dlabel.nii \
  --surf-lh PATH/TO/S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii \
  --surf-rh PATH/TO/S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii \
  --fig-dir outputs/brain_gradients

# Or pass the TSV directly
python plot_brain_gradients.py \
  --tsv PATH/TO/reference_gradient_with_ID.tsv \
  --atlas PATH/TO/Schaefer2018_400Parcels_7Networks_order.dlabel.nii \
  --surf-lh PATH/TO/S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii \
  --surf-rh PATH/TO/S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii \
  --fig-dir outputs/brain_gradients \
  --k 5
"""

# --- keep plotting headless & quiet for CI/servers ---
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("PYVISTA_ENABLE_TRAME", "false")
os.environ.setdefault("PYVISTA_USE_IPYVTK", "false")
os.environ.setdefault("SURFPLOT_ENGINE", "pyvista")

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmasher as cmr
from surfplot import Plot

# ---- Repo-local import shim (plotting -> utils.get_surfaces) ----
import adaptman.utils as _am_utils

def _get_surfaces(sp):
    if isinstance(sp, dict) and "lh" in sp and "rh" in sp:
        return sp["lh"], sp["rh"]
    if isinstance(sp, (list, tuple)) and len(sp) == 2:
        return sp[0], sp[1]
    raise ValueError("surf_path must be {'lh': <path>, 'rh': <path>} or a 2-sequence of paths.")

if not hasattr(_am_utils, "get_surfaces"):
    setattr(_am_utils, "get_surfaces", _get_surfaces)

from adaptman.analyses import plotting  # uses adaptman.utils.get_surfaces inside


# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot PCA/gradient component maps on cortical surfaces (TSV + atlas + GIFTI surfaces)."
    )
    # TSV locating
    ap.add_argument("--tsv", type=str, default=None,
                    help="Direct path to TSV with columns: Region, g1, g2, ...")
    ap.add_argument("--search-dir", action="append", default=None,
                    help="Folder to search for the TSV (can be repeated).")
    ap.add_argument("--env-override", type=str, default="PCA_RESULTS_PATH_OVERRIDE",
                    help="Env var name; if set, searched first for TSV (path or directory).")

    # Required anatomy
    ap.add_argument("--atlas", required=True,
                    help="Path to Schaefer 400 (7-net) CIFTI dlabel (.dlabel.nii).")
    ap.add_argument("--surf-lh", required=True,
                    help="Left hemisphere GIFTI surface (.surf.gii).")
    ap.add_argument("--surf-rh", required=True,
                    help="Right hemisphere GIFTI surface (.surf.gii).")

    # Outputs
    ap.add_argument("--fig-dir", required=True,
                    help="Output directory for images (will be created).")

    # Options
    ap.add_argument("--k", type=int, default=5,
                    help="Number of components to plot (g1..gK).")
    return ap.parse_args()


# =========================
# Constants / view presets
# =========================

VIEWS = {
    "dorsal": {"size": (150, 200), "zoom": 3.3, "layout": "row", "mirror": True},
}

# Valid region IDs (VAN SMN)
VALID_REGION_IDS = {
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 231, 232, 233, 234, 235, 236,
    237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 294, 295, 296, 297,
    298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316,
    317, 318
}


# =========================
# File discovery / loading
# =========================

def find_reference_tsv(tsv_arg: str | None, search_dirs: list[str] | None, env_override_name: str) -> Path:
    """
    Locate the gradients TSV from --tsv, or by searching directories, with an optional env var hint.
    """
    # 1) Direct path (highest priority)
    if tsv_arg:
        tsv = Path(tsv_arg)
        if not tsv.exists():
            raise FileNotFoundError(f"--tsv not found: {tsv}")
        print(f"[info] Using TSV (explicit): {tsv}")
        return tsv

    candidate_dirs: list[Path] = []

    # 2) Env var (if set) can be a file or directory
    env_value = os.environ.get(env_override_name, None)
    if env_value:
        env_path = Path(env_value)
        if env_path.is_file():
            print(f"[info] Using TSV (env override): {env_path}")
            return env_path
        candidate_dirs.append(env_path)

    # 3) User-provided search dirs
    if search_dirs:
        candidate_dirs.extend(Path(d) for d in search_dirs)

    if not candidate_dirs:
        raise ValueError("Provide --tsv OR at least one --search-dir (or set the env var).")

    # search patterns (first match wins)
    targets = ["reference_gradient_with_ID.tsv"]
    patterns = ["reference*_with_ID.tsv", "*gradient*ID*.tsv", "*reference*ID*.tsv"]

    checked = []
    for d in candidate_dirs:
        if not d.is_dir():
            continue
        for t in targets:
            f = d / t
            checked.append(str(f))
            if f.exists():
                print(f"[info] Using TSV: {f}")
                return f
        for pat in patterns:
            hits = sorted(d.glob(pat))
            if hits:
                print(f"[info] Using TSV (pattern match): {hits[0]}")
                return hits[0]

    msg = "Could not find a TSV with gradients. Checked:\n  " + "\n  ".join(checked)
    raise FileNotFoundError(msg)


def load_and_clean(tsv_path: Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Read TSV (Region, g1..gK..), coerce numeric, filter rows, drop NA in used cols.
    Returns:
        region_ids : (N,) int array (1..400 Schaefer indices)
        grads      : (N, k) float array
    """
    df = pd.read_csv(tsv_path, sep="\t", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"File {tsv_path} has <2 columns (Region + components).")
    n_comp = df.shape[1] - 1
    colnames = ["Region"] + [f"g{j+1}" for j in range(n_comp)]
    df.columns = colnames

    df["Region"] = pd.to_numeric(df["Region"], errors="coerce").astype("Int64")
    for c in colnames[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["Region"].isin(VALID_REGION_IDS)].copy()
    use_cols = colnames[1:1 + k]
    subset = ["Region"] + use_cols
    df = df.dropna(subset=subset)
    if df.empty:
        raise ValueError("After filtering/coercion, no valid rows remain. Check Region IDs and gradient values.")

    region_ids = df["Region"].astype(int).to_numpy()
    grads = df[use_cols].to_numpy(dtype=float)
    return region_ids, grads


# =========================
# Plotting
# =========================

def plot_ref_brain_gradients(tsv_path: Path, atlas: Path, surf_lh: Path, surf_rh: Path,
                             fig_dir: Path, k: int = 5) -> None:
    """
    Render g1..gK on cortical surfaces; save colorbar and per-component panels.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    plotting.set_plotting()

    region_ids, grads = load_and_clean(tsv_path, k=k)
    vmax = float(np.nanmax(np.abs(grads))) if grads.size else 1.0
    vmax = float(np.around(vmax, 1)) or 1.0
    prefix = str(fig_dir / "gradients_")
    cmap = cmr.get_sub_cmap("twilight_shifted", 0.05, 0.95)

    # --- shared colorbar (ticks preserved) ---
    fig, ax = plt.subplots(figsize=(7.5, 1))
    norm = plt.Normalize(-vmax, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_ticks([-2.3, 0, 2.3])
    cbar.set_ticklabels(["-2.3", "0", "2.3"])
    plt.xticks(fontsize=50)
    cbar.outline.set_visible(False)
    ax.spines[:].set_visible(False)
    fig.savefig(prefix + "cbar.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- per-component (standard + dorsal) ---
    n_comp = grads.shape[1]
    for i in range(n_comp):
        # map region values -> 400 parcels -> vertices
        mapped_values = np.zeros(400, dtype=float)
        idx = (region_ids - 1).astype(int)
        valid_mask = (idx >= 0) & (idx < 400)
        mapped_values[idx[valid_mask]] = grads[:, i][valid_mask]
        vertices = plotting.weights_to_vertices(mapped_values, str(atlas))

        # 4-panel export
        p = Plot(str(surf_lh), str(surf_rh))
        p.add_layer(vertices, cmap=cmap, color_range=(-vmax, vmax), cbar=False)
        fig_std = p.build(colorbar=False)
        fig_std.savefig(prefix + f"PC{i + 1}_brain.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_std)

        # dorsal-only export
        vopts = VIEWS["dorsal"]
        p_d = Plot(
            str(surf_lh), str(surf_rh),
            views="dorsal",
            layout=vopts["layout"],
            mirror_views=vopts["mirror"],
            size=vopts["size"],
            zoom=vopts["zoom"],
        )
        p_d.add_layer(vertices, cmap=cmap, color_range=(-vmax, vmax), cbar=False)
        fig_d = p_d.build(colorbar=False)
        fig_d.savefig(prefix + f"PC{i + 1}_dorsal.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_d)


# =========================
# Main
# =========================

def main():
    args = parse_args()

    tsv_path = find_reference_tsv(
        tsv_arg=args.tsv,
        search_dirs=args.search_dir,
        env_override_name=args.env_override
    )

    plot_ref_brain_gradients(
        tsv_path=tsv_path,
        atlas=Path(args.atlas),
        surf_lh=Path(args.surf_lh),
        surf_rh=Path(args.surf_rh),
        fig_dir=Path(args.fig_dir),
        k=args.k,
    )
    print("✅ Saved figures to:", args.fig_dir)


if __name__ == "__main__":
    main()
