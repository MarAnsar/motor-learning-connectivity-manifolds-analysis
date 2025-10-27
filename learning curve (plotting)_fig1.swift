#!/usr/bin/env python3
"""
Learning curve plot (Day 1 vs Day 2) for visuomotor adaptation.

You MUST provide your own paths via CLI flags (no hardcoded defaults).

"""

import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def bin_series(series: pd.Series, trials_per_block: int) -> np.ndarray:
    """Bin a per-trial 1D series into blocks and return mean per block."""
    s = pd.Series(series).squeeze()
    return s.groupby(s.index // trials_per_block).mean().to_numpy()

def read_vector(path: Path) -> pd.Series:
    """Read a single-column TSV into a Series (no header)."""
    return pd.read_csv(path, header=None, sep="\t").squeeze()

def load_binned_group(path: Path, trials_per_block: int) -> np.ndarray:
    """Read per-trial group series and return binned means."""
    return bin_series(read_vector(path), trials_per_block)

def clamp_len(arr: np.ndarray, n: int) -> np.ndarray:
    """Trim array to length n (no pad)."""
    return arr[:n]

def build_axes(blocks: int, offset: int):
    """Return x-coordinates for Day 1 and Day 2 (offset) and epoch ranges."""
    x_day1 = np.arange(1, blocks + 1)
    x_day2 = np.arange(1, blocks + 1) + offset
    x_pre = np.arange(1, 16)
    x_post = np.arange(16, 56)
    x_washout = np.arange(56, 71)
    return x_day1, x_day2, x_pre, x_post, x_washout


# -----------------------------
# Plot
# -----------------------------

def plot_learning_curve(
    base_dir: Path,
    out_dir: Path,
    trials_per_block: int,
    blocks: int,
    offset: int,
    sem_value: float,
    lw_subject: float,
    lw_mean: float,
    lw_aux: float,
):
    """
    Create learning-curve plot:
    - individual subject traces (thin, transparent)
    - group means with SEM shading for pre/post/washout
    - Day 2 offset on x-axis to show both days on one timeline
    """
    day1_dir = base_dir / "day1"
    day2_dir = base_dir / "day2"

    # Validate required files exist (basic checks)
    required = [
        day1_dir / "day1_pre.tsv", day1_dir / "day1_post.tsv", day1_dir / "day1_washout.tsv",
        day2_dir / "day2_pre.tsv", day2_dir / "day2_post.tsv", day2_dir / "day2_washout.tsv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n  - " + "\n  - ".join(missing)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_svg = out_dir / "learning_curve_final_clean.svg"

    # Colors
    color_day1 = "#145A32"  # forest green
    color_day2 = "#FF9900"  # orange

    # X axes
    x_day1, x_day2, x_pre, x_post, x_washout = build_axes(blocks, offset)

    # -------------------------
    # Group means (binned)
    # -------------------------
    mean_pre1  = load_binned_group(day1_dir / "day1_pre.tsv",  trials_per_block)
    mean_post1 = load_binned_group(day1_dir / "day1_post.tsv", trials_per_block)
    mean_wo1   = load_binned_group(day1_dir / "day1_washout.tsv", trials_per_block)

    mean_pre2  = load_binned_group(day2_dir / "day2_pre.tsv",  trials_per_block)
    mean_post2 = load_binned_group(day2_dir / "day2_post.tsv", trials_per_block)
    mean_wo2   = load_binned_group(day2_dir / "day2_washout.tsv", trials_per_block)

    # Clamp epoch lengths to expected ranges (15 / 40 / 15 blocks)
    mean_pre1  = clamp_len(mean_pre1,  len(x_pre))
    mean_post1 = clamp_len(mean_post1, len(x_post))
    mean_wo1   = clamp_len(mean_wo1,   len(x_washout))
    mean_pre2  = clamp_len(mean_pre2,  len(x_pre))
    mean_post2 = clamp_len(mean_post2, len(x_post))
    mean_wo2   = clamp_len(mean_wo2,   len(x_washout))

    # SEMs (constant placeholder unless you provide real SEMs)
    sem_pre1  = np.full_like(mean_pre1,  sem_value, dtype=float)
    sem_post1 = np.full_like(mean_post1, sem_value, dtype=float)
    sem_wo1   = np.full_like(mean_wo1,   sem_value, dtype=float)
    sem_pre2  = np.full_like(mean_pre2,  sem_value, dtype=float)
    sem_post2 = np.full_like(mean_post2, sem_value, dtype=float)
    sem_wo2   = np.full_like(mean_wo2,   sem_value, dtype=float)

    # -------------------------
    # Subject spaghetti lines
    # -------------------------
    sub_files_day1 = sorted(glob.glob(str(day1_dir / "subject_*_day1.tsv")))
    sub_files_day2 = sorted(glob.glob(str(day2_dir / "subject_*_day2.tsv")))
    n1, n2 = len(sub_files_day1), len(sub_files_day2)
    pair_count = min(n1, n2)
    if n1 != n2:
        print(f"⚠️ Subject file count mismatch: Day1={n1}, Day2={n2}. Plotting pairs up to min={pair_count}.")

    fig, ax = plt.subplots(figsize=(44, 14))

    # Day 1 traces
    for f in sub_files_day1[:pair_count]:
        trials = read_vector(Path(f))
        binned = clamp_len(bin_series(trials, trials_per_block), blocks)
        ax.plot(x_day1, binned, color=color_day1, lw=lw_subject, alpha=0.22)

    # Day 2 traces (offset)
    for f in sub_files_day2[:pair_count]:
        trials = read_vector(Path(f))
        binned = clamp_len(bin_series(trials, trials_per_block), blocks)
        ax.plot(x_day2, binned, color=color_day2, lw=lw_subject, alpha=0.22)

    # -------------------------
    # Group means + SEM shading
    # -------------------------
    # Day 1
    ax.fill_between(x_pre,     mean_pre1 - sem_pre1,   mean_pre1 + sem_pre1,   color=color_day1, alpha=0.20)
    ax.fill_between(x_post,    mean_post1 - sem_post1, mean_post1 + sem_post1, color=color_day1, alpha=0.20)
    ax.fill_between(x_washout, mean_wo1 - sem_wo1,     mean_wo1 + sem_wo1,     color=color_day1, alpha=0.20)
    ax.plot(x_pre,     mean_pre1,  color=color_day1, lw=lw_mean, label="Day 1")
    ax.plot(x_post,    mean_post1, color=color_day1, lw=lw_mean)
    ax.plot(x_washout, mean_wo1,   color=color_day1, lw=lw_mean)

    # Day 2 (offset)
    ax.fill_between(x_pre + offset,     mean_pre2 - sem_pre2,   mean_pre2 + sem_pre2,   color=color_day2, alpha=0.20)
    ax.fill_between(x_post + offset,    mean_post2 - sem_post2, mean_post2 + sem_post2, color=color_day2, alpha=0.20)
    ax.fill_between(x_washout + offset, mean_wo2 - sem_wo2,     mean_wo2 + sem_wo2,     color=color_day2, alpha=0.20)
    ax.plot(x_pre + offset,     mean_pre2,  color=color_day2, lw=lw_mean, label="Day 2")
    ax.plot(x_post + offset,    mean_post2, color=color_day2, lw=lw_mean)
    ax.plot(x_washout + offset, mean_wo2,   color=color_day2, lw=lw_mean)

    # -------------------------
    # Axes / labels / style
    # -------------------------
    xticks_all = np.concatenate([x_day1, x_day2])
    xticks_show = [x for x in xticks_all if x % 5 == 0]
    xtick_labels = [str(x if x <= blocks else x - offset) for x in xticks_show]

    ax.set_xticks(xticks_show)
    ax.set_xticklabels(xtick_labels, fontsize=30)

    ax.set_xlim(0.5, x_day2[-1] + 5)
    ax.set_ylim(-55, 78)

    yticks = np.arange(-45, 90, 15)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y)}" for y in yticks], fontsize=30)

    ax.grid(False)
    ax.set_axisbelow(True)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=lw_aux)
    ax.axvline(x=blocks + 2.5, color="black", linestyle="-", lw=lw_aux)

    ax.text(x_day1[0], 80, "Day 1", ha="left", fontsize=34, color="gray")
    ax.text(x_day2[0], 80, "Day 2", ha="left", fontsize=34, color="gray")

    ax.set_xlabel("Trial block",   fontsize=34, labelpad=18)
    ax.set_ylabel("Adaptation (°)", fontsize=34, labelpad=18)
    ax.tick_params(labelsize=28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save
    plt.tight_layout()
    fig.savefig(out_svg, dpi=600, bbox_inches="tight", format="svg")
    plt.close(fig)

    print(f"✅ Learning curve plot saved (SVG): {out_svg}")
    print(f"Subjects plotted per day: {pair_count}  "
          f"(subject lw={lw_subject}, mean lw={lw_mean}, aux lw={lw_aux})")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot learning curves (Day 1 vs Day 2) with binned blocks.\n"
                    "Provide your own paths for data and outputs."
    )
    ap.add_argument(
        "--base-dir",
        required=True,
        help="PATH/TO/.../all_together directory that contains day1/ and day2/ subfolders."
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="PATH/TO/output directory where the SVG figure will be saved."
    )
    ap.add_argument("--trials-per-block", type=int, default=8, help="Trials per block.")
    ap.add_argument("--blocks", type=int, default=70, help="Number of plotted blocks per day.")
    ap.add_argument("--offset", type=int, default=75, help="X-axis offset for Day 2 blocks.")
    ap.add_argument("--sem", type=float, default=6.0, help="Placeholder SEM value if real SEMs are not provided.")
    ap.add_argument("--lw-subject", type=float, default=0.15, help="Line width for subject traces.")
    ap.add_argument("--lw-mean", type=float, default=1.5, help="Line width for group means.")
    ap.add_argument("--lw-aux", type=float, default=0.25, help="Line width for zero/divider lines.")
    return ap.parse_args()


def main():
    args = parse_args()
    plot_learning_curve(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        trials_per_block=args.trials_per_block,
        blocks=args.blocks,
        offset=args.offset,
        sem_value=args.sem,
        lw_subject=args.lw_subject,
        lw_mean=args.lw_mean,
        lw_aux=args.lw_aux,
    )


if __name__ == "__main__":
    main()
