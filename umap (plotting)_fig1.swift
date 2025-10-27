#!/usr/bin/env python3
"""
UMAP plots for centered and uncentered VAN/SM connectivity matrices.

You MUST pass your paths via CLI flags (no hardcoded defaults).

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation


# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="UMAP plots for centered and uncentered VAN/SM connectivity matrices. "
                    "All paths must be provided explicitly."
    )
    # centered inputs (required)
    ap.add_argument("--centered-day1", required=True,
                    help="Folder with centered Day 1 matrices (TSV).")
    ap.add_argument("--centered-day2", required=True,
                    help="Folder with centered Day 2 matrices (TSV).")

    # uncentered inputs (required)
    ap.add_argument("--uncentered-day1", required=True,
                    help="Folder with uncentered Day 1 matrices (TSV).")
    ap.add_argument("--uncentered-day2", required=True,
                    help="Folder with uncentered Day 2 matrices (TSV).")

    # outputs (required)
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write figures and legends.")

    # UMAP params
    ap.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors.")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for UMAP.")
    return ap.parse_args()


# =========================
# Style / constants
# =========================

sns.set_style("white")
sns.set_context("notebook", font_scale=1.1)

FIGSIZE      = (8, 6)
TITLE_FSIZE  = 25
LABEL_FSIZE  = 30
TICK_FSIZE   = 30
POINT_SIZE   = 50
ALPHA        = 0.85
EDGEWIDTH    = 0.5
TICK_PAD     = 8
EXTRA_OFFSET = 4  # only used for uncentered tick offset

SUBJECT_LIST = [f"sub-{i:02d}" for i in range(1, 33)]
_cmap = plt.get_cmap("tab20")
SUBJECT_COLOR_MAP = {subj: _cmap(idx % 20) for idx, subj in enumerate(SUBJECT_LIST)}

SESSION_EPOCH_STYLE = {
    ("ses-01", "base"):    ("forestgreen",  "o", "D1 Baseline"),
    ("ses-01", "early"):   ("gold",         "o", "D1 Early"),
    ("ses-01", "late"):    ("darkorange",   "o", "D1 Late"),
    ("ses-01", "washout"): ("firebrick",    "o", "D1 Washout"),
    ("ses-02", "base"):    ("grey",         "s", "D2 Baseline"),
    ("ses-02", "early"):   ("black",        "s", "D2 Early"),
    ("ses-02", "late"):    ("lightpink",    "s", "D2 Late"),
    ("ses-02", "washout"): ("mediumpurple", "s", "D2 Washout"),
}

def make_subject_legend_handles(subject_list=None, color_map=None):
    subject_list = SUBJECT_LIST if subject_list is None else subject_list
    color_map = SUBJECT_COLOR_MAP if color_map is None else color_map
    return [Patch(color=color_map[s], label=s) for s in subject_list if s in color_map]

def make_session_epoch_legend_handles(style_map=None):
    style_map = SESSION_EPOCH_STYLE if style_map is None else style_map
    return [
        Line2D([0],[0], marker=mk, color='w',
               markerfacecolor=col, markeredgecolor=col,
               linestyle='', markersize=10, label=lbl)
        for (_, _), (col, mk, lbl) in style_map.items()
    ]


# =========================
# Data readers
# =========================

def read_and_flatten_centered(dirs):
    """
    Read centered .tsv matrices and flatten them row-wise.
    Expect filenames like: sub-XX_ses-YY_*_{base|early|late|washout}.tsv
    Returns:
        vectors: (N, F) array, subjects: list, sessions: list, epochs: list
    """
    records = []
    for day_dir in dirs:
        day_dir = Path(day_dir)
        if not day_dir.exists():
            continue
        for subdir in sorted(day_dir.glob("sub-*")):
            subj_id = subdir.name
            for fpath in sorted(subdir.glob("*.tsv")):
                parts = fpath.stem.split("_")
                if len(parts) < 3:
                    continue
                sess = parts[1]          # e.g., 'ses-01'
                epoch = parts[-1]        # e.g., 'base'
                mat = pd.read_table(fpath, index_col=0).values
                vec = mat.flatten()
                records.append((vec, subj_id, sess, epoch))

    if not records:
        raise RuntimeError("No centered .tsv files found in provided directories.")

    vectors = np.stack([r[0] for r in records], axis=0)
    subjects = [r[1] for r in records]
    sessions = [r[2] for r in records]
    epochs   = [r[3] for r in records]
    return vectors, subjects, sessions, epochs


def read_and_flatten_uncentered(dirs):
    """
    Read uncentered .tsv matrices and return a matrix whose columns are flattened samples.
    This mirrors the original uncentered script (columns = samples).
    Returns:
        flat_df: DataFrame shape (F, N) with columns named 'sub_ses_epoch'
    """
    files = []
    for d in dirs:
        d = Path(d)
        if d.exists():
            files.extend(list(d.rglob("*.tsv")))

    cols = []
    data = []
    for fp in sorted(files):
        parts = fp.stem.split("_")
        if len(parts) < 3:
            continue
        subj, sess, epoch = parts[0], parts[1], parts[-1]
        mat = pd.read_table(fp, index_col=0).values
        data.append(mat.flatten())
        cols.append(f"{subj}_{sess}_{epoch}")

    if not data:
        raise RuntimeError("No uncentered .tsv files read.")

    flat_df = pd.DataFrame(np.array(data).T, columns=cols)
    return flat_df


# =========================
# Axis styling
# =========================

def style_ax_centered(ax, title, xlim, ylim):
    ax.set_title(title, fontsize=TITLE_FSIZE, pad=15)
    ax.set_xlabel("Dimension 1", fontsize=LABEL_FSIZE, labelpad=10)
    ax.set_ylabel("Dimension 2", fontsize=LABEL_FSIZE, labelpad=10)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='x', which='major', direction='out',
                   length=6, width=1, bottom=True, top=False,
                   pad=TICK_PAD, labelsize=TICK_FSIZE)
    ax.tick_params(axis='y', which='major', direction='out',
                   length=6, width=1, left=True, right=False,
                   pad=TICK_PAD, labelsize=TICK_FSIZE)
    sns.despine(ax=ax, trim=False)

def offset_first_tick_label(ax, axis='x'):
    fig = ax.figure
    if axis == 'x':
        labs = ax.get_xticklabels()
        trans = ScaledTranslation(0, -EXTRA_OFFSET/72, fig.dpi_scale_trans)
    else:
        labs = ax.get_yticklabels()
        trans = ScaledTranslation(-EXTRA_OFFSET/72, 0, fig.dpi_scale_trans)
    if labs:
        labs[0].set_transform(labs[0].get_transform() + trans)

def style_ax_uncentered(ax, title, xlim, ylim):
    ax.set_title(title, fontsize=TITLE_FSIZE, pad=15)
    ax.set_xlabel("Dimension 1", fontsize=LABEL_FSIZE, labelpad=10)
    ax.set_ylabel("Dimension 2", fontsize=LABEL_FSIZE, labelpad=10)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='x', which='major', direction='out',
                   length=6, width=1, bottom=True, top=False,
                   pad=TICK_PAD, labelsize=TICK_FSIZE)
    ax.tick_params(axis='y', which='major', direction='out',
                   length=6, width=1, left=True, right=False,
                   pad=TICK_PAD, labelsize=TICK_FSIZE)
    # extra offset of first tick label away from axes spines
    offset_first_tick_label(ax, 'x')
    offset_first_tick_label(ax, 'y')
    sns.despine(ax=ax, trim=False)


# =========================
# Plotters
# =========================

def plot_centered(args, out_dir: Path):
    # load data
    vectors, subjects, sessions, epochs = read_and_flatten_centered(
        [args.centered_day1, args.centered_day2]
    )

    reducer = UMAP(random_state=args.seed, n_neighbors=args.neighbors, min_dist=args.min_dist)
    emb = reducer.fit_transform(vectors)

    # margins
    x_min, x_max = emb[:, 0].min(), emb[:, 0].max()
    y_min, y_max = emb[:, 1].min(), emb[:, 1].max()
    margin_x = 0.05 * (x_max - x_min) or 1.0
    margin_y = 0.05 * (y_max - y_min) or 1.0
    xlim = (np.floor(x_min - margin_x), np.ceil(x_max + margin_x))
    ylim = (np.floor(y_min - margin_y), np.ceil(y_max + margin_y))

    # A) by subject
    fig, ax = plt.subplots(figsize=FIGSIZE)
    subs_arr = np.array(subjects)
    for subj in SUBJECT_LIST:
        mask = subs_arr == subj
        if not mask.any():
            continue
        col = SUBJECT_COLOR_MAP.get(subj, "gray")
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            color=col, edgecolor=col,
            s=POINT_SIZE, alpha=ALPHA, linewidth=EDGEWIDTH, label=subj
        )
    style_ax_centered(ax, "Subjects (Centered Connectivity)", xlim, ylim)
    plt.tight_layout()
    fig.savefig(out_dir / "umap_centered_by_subject_fmt.png", dpi=300)
    fig.savefig(out_dir / "umap_centered_by_subject_fmt.svg")
    plt.close(fig)

    # B) by session-epoch
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sessions_arr = np.array(sessions)
    epochs_arr   = np.array(epochs)
    for (sess_key, epoch_key), (col, mk, lbl) in SESSION_EPOCH_STYLE.items():
        mask = (sessions_arr == sess_key) & (epochs_arr == epoch_key)
        if not mask.any():
            continue
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            color=col, marker=mk,
            edgecolor=col, linewidth=EDGEWIDTH,
            s=POINT_SIZE, alpha=ALPHA, label=lbl
        )
    style_ax_centered(ax, "Epochs (Centered Connectivity)", xlim, ylim)
    plt.tight_layout()
    fig.savefig(out_dir / "umap_centered_by_task_session_fmt.png", dpi=300)
    fig.savefig(out_dir / "umap_centered_by_task_session_fmt.svg")
    plt.close(fig)

    # Legends as separate images
    subj_handles = make_subject_legend_handles()
    fig, ax = plt.subplots(figsize=(3, max(4, 0.3 * len(subj_handles))))
    ax.legend(handles=subj_handles, loc="center")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "legend_subjects.png", dpi=300)
    fig.savefig(out_dir / "legend_subjects.svg")
    plt.close(fig)

    sess_handles = make_session_epoch_legend_handles()
    fig, ax = plt.subplots(figsize=(3, max(4, 0.5 * len(sess_handles))))
    ax.legend(handles=sess_handles, loc="center")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "legend_sessions.png", dpi=300)
    fig.savefig(out_dir / "legend_sessions.svg")
    plt.close(fig)


def plot_uncentered(args, out_dir: Path):
    # load data
    flat_df = read_and_flatten_uncentered([args.uncentered_day1, args.uncentered_day2])

    # parse labels from column names
    cols = flat_df.columns.str.split("_", expand=True)
    subjects = cols.get_level_values(0).to_numpy()
    sessions = cols.get_level_values(1).to_numpy()
    epochs   = cols.get_level_values(2).to_numpy()

    # samples = columns → transpose
    X = flat_df.T.values
    reducer = UMAP(random_state=args.seed, n_neighbors=args.neighbors, min_dist=args.min_dist)
    emb = reducer.fit_transform(X)

    # margins
    x0, x1 = emb[:, 0].min(), emb[:, 0].max()
    y0, y1 = emb[:, 1].min(), emb[:, 1].max()
    mx = 0.05 * (x1 - x0) or 1.0
    my = 0.05 * (y1 - y0) or 1.0
    xlim = (np.floor(x0 - mx), np.ceil(x1 + mx))
    ylim = (np.floor(y0 - my), np.ceil(y1 + my))

    # A) by subject (no inline legend)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for subj in SUBJECT_LIST:
        mask = (subjects == subj)
        if not mask.any():
            continue
        col = SUBJECT_COLOR_MAP[subj]
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            color=col, edgecolor=col,
            s=POINT_SIZE, alpha=ALPHA, linewidth=EDGEWIDTH
        )
    style_ax_uncentered(ax, "Subjects (Uncentered Connectivity)", xlim, ylim)
    plt.tight_layout()
    fig.savefig(out_dir / "umap_uncentered_by_subject_fmt.png", dpi=300)
    fig.savefig(out_dir / "umap_uncentered_by_subject_fmt.svg")
    plt.close(fig)

    # B) by session–epoch (no inline legend)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for (sess, epoch), (col, mk, lbl) in SESSION_EPOCH_STYLE.items():
        mask = (sessions == sess) & (epochs == epoch)
        if not mask.any():
            continue
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            color=col, marker=mk,
            edgecolor=col, linewidth=EDGEWIDTH,
            s=POINT_SIZE, alpha=ALPHA
        )
    style_ax_uncentered(ax, "Epochs (Uncentered Connectivity)", xlim, ylim)
    plt.tight_layout()
    fig.savefig(out_dir / "umap_uncentered_by_task_fmt.png", dpi=300)
    fig.savefig(out_dir / "umap_uncentered_by_task_fmt.svg")
    plt.close(fig)


# =========================
# Main
# =========================

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Centered plots (with separate legends)
    plot_centered(args, out_dir)

    # Uncentered plots (no inline legends)
    plot_uncentered(args, out_dir)

    print("✅ Done! Centered & Uncentered UMAP plots saved to:", out_dir)


if __name__ == "__main__":
    main()
