#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _read_table(path: Path) -> pd.DataFrame:
    """
    Read a TSV/CSV by extension. Falls back to pandas default if unknown.
    """
    ext = path.suffix.lower()
    if ext in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if ext == ".csv":
        return pd.read_csv(path)
    # be forgiving (user might have no/odd extension)
    return pd.read_csv(path, sep=None, engine="python")


def _pairwise_dists_squared(x: np.ndarray) -> np.ndarray:
    """
    Fast pairwise squared distances using (x - x)^2 expansion:
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    """
    # x: (n, d)
    norms = np.sum(x * x, axis=1, keepdims=True)  # (n, 1)
    d2 = norms + norms.T - 2.0 * (x @ x.T)
    # numerical noise can produce tiny negative values; clamp:
    np.maximum(d2, 0.0, out=d2)
    return d2


def _pick_four_spread_points(coords: np.ndarray) -> list[int]:
    """
    Heuristic selection of 4 points that are maximally spread:
    - pick the farthest pair
    - then greedily add points that maximize min-distance to the set
    """
    d2 = _pairwise_dists_squared(coords)
    # farthest pair
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    chosen = [int(i), int(j)]

    # add until 4
    while len(chosen) < 4:
        mask = np.ones(len(coords), dtype=bool)
        mask[chosen] = False
        candidates = np.where(mask)[0]
        # for each candidate, compute distance to closest chosen point; pick the max
        min_d2_to_chosen = np.min(d2[np.ix_(candidates, chosen)], axis=1)
        next_idx = int(candidates[np.argmax(min_d2_to_chosen)])
        chosen.append(next_idx)

    return chosen


def plot_eccentricity_3d(
    df_grad: pd.DataFrame,
    df_ecc: pd.DataFrame,
    ecc_col: str = "BaseAvg",
    view_elev: float = 20,
    view_azim: float = -120,
    figsize: tuple[float, float] = (8.5, 8.5),
) -> plt.Figure:
    """
    Create the 3D eccentricity plot and return the Matplotlib Figure.
    """
    _require_columns(df_grad, ["g1", "g2", "g3"], "Gradient table")
    _require_columns(df_ecc, [ecc_col], "Eccentricity table")

    # Align by row order (common case in pipelines). If your data needs joining
    # by an index/label, adapt here.
    df = df_grad.copy().reset_index(drop=True)
    df[ecc_col] = df_ecc[ecc_col].values

    coords = df[["g1", "g2", "g3"]].to_numpy(dtype=float)
    ecc = df[ecc_col].to_numpy(dtype=float)
    vmin, vmax = float(np.min(ecc)), float(np.max(ecc))

    chosen = _pick_four_spread_points(coords)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # scatter
    sc = ax.scatter(df["g1"], df["g2"], df["g3"], c=ecc, cmap="viridis", s=60, alpha=0.65)

    # axis labels
    equal_labelpad = 15
    ax.set_xlabel("PC1", fontsize=20, labelpad=equal_labelpad)
    ax.set_ylabel("PC2", fontsize=20, labelpad=equal_labelpad)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("PC3", fontsize=20, labelpad=equal_labelpad)

    # integer-ish ticks with ~10 per axis
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="z", labelsize=18)

    # colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.15, orientation="horizontal")
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_title("Ecc", fontsize=25, pad=8)

    # view
    ax.view_init(elev=view_elev, azim=view_azim)

    # central dot + connectors to the 4 selected points
    ax.scatter(0, 0, 0, c="k", marker="s", s=60, zorder=11)
    for idx in chosen:
        x, y, z = coords[idx]
        ax.scatter(x, y, z, facecolors="none", edgecolors="k", s=120, linewidths=1.4, zorder=12)
        ax.plot([0, x], [0, y], [0, z], "k--", lw=1.5)

    # avoid tight_layout for 3D; adjust margins manually
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.98)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot 3D gradient coordinates colored by eccentricity."
    )
    p.add_argument("--grad", type=Path, required=True,
                   help="Path to reference gradient table (TSV/CSV) with columns g1,g2,g3.")
    p.add_argument("--ecc", type=Path, required=True,
                   help="Path to eccentricity table (TSV/CSV).")
    p.add_argument("--ecc-col", type=str, default="BaseAvg",
                   help="Column name in the eccentricity table to color by. Default: BaseAvg")
    p.add_argument("--out", type=Path, default=Path("outputs"),
                   help="Output directory. Default: ./outputs")
    p.add_argument("--svg", type=str, default="eccentricity_3D.svg",
                   help="SVG filename. Default: eccentricity_3D.svg")
    p.add_argument("--png", type=str, default="eccentricity_3D.png",
                   help="PNG filename. Default: eccentricity_3D.png")
    p.add_argument("--dpi", type=int, default=300,
                   help="PNG DPI. Default: 300")
    p.add_argument("--view-elev", type=float, default=20,
                   help="3D elevation angle. Default: 20")
    p.add_argument("--view-azim", type=float, default=-120,
                   help="3D azimuth angle. Default: -120")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.grad.exists():
        raise FileNotFoundError(f"--grad not found: {args.grad}")
    if not args.ecc.exists():
        raise FileNotFoundError(f"--ecc not found: {args.ecc}")

    df_grad = _read_table(args.grad)
    df_ecc = _read_table(args.ecc)

    fig = plot_eccentricity_3d(
        df_grad=df_grad,
        df_ecc=df_ecc,
        ecc_col=args.ecc_col,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    svg_path = args.out / args.svg
    png_path = args.out / args.png

    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved:\n  - {svg_path}\n  - {png_path}")


if __name__ == "__main__":
    main()
