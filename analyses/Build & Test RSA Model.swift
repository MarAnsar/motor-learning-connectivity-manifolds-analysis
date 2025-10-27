#!/usr/bin/env python3
"""
Build & Test RSA Model RSMs (Time, Epoch, Day) Against Subject RSMs

A) Builds three 0/1 model RSMs over 8 conditions:
   ["Base Day1", "Early Day1", "Late Day1", "Washout Day1",
    "Base Day2", "Early Day2", "Late Day2", "Washout Day2"]

   - TIME  : 1 for adjacent timepoints within a day (Base↔Early, Early↔Late, Late↔Washout)
   - EPOCH : 1 for same epoch type across days (Base D1↔Base D2, Early D1↔Early D2, etc.)
   - DAY   : 1 for within-day pairs (any two epochs of Day1; any two of Day2)

B) Tests each model RSM vs each subject's RSM using a chosen similarity metric:
   - cosine  (default)
 
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ----------------------
# Defaults / conventions
# ----------------------

DEFAULT_LABELS: List[str] = [
    "Base Day1", "Early Day1", "Late Day1", "Washout Day1",
    "Base Day2", "Early Day2", "Late Day2", "Washout Day2",
]

BIN_CMAP = ListedColormap(["#000000", "#f2e9e4"])  # 0 -> black, 1 -> cream


# ----------------------
# Model constructors
# ----------------------

def build_time_model(labels: List[str]) -> pd.DataFrame:
    """Adjacent timepoints within each day = 1, else 0."""
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    adjacent_pairs = [
        ("Base Day1", "Early Day1"),
        ("Early Day1", "Late Day1"),
        ("Late Day1", "Washout Day1"),
        ("Base Day2", "Early Day2"),
        ("Early Day2", "Late Day2"),
        ("Late Day2", "Washout Day2"),
    ]
    idx = {lab: i for i, lab in enumerate(labels)}
    for a, b in adjacent_pairs:
        i, j = idx[a], idx[b]
        mat[i, j] = mat[j, i] = 1
    return pd.DataFrame(mat, index=labels, columns=labels)


def build_epoch_model(labels: List[str]) -> pd.DataFrame:
    """Same epoch type across days = 1, else 0."""
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    type_groups: Dict[str, List[str]] = {
        "Base":    ["Base Day1", "Base Day2"],
        "Early":   ["Early Day1", "Early Day2"],
        "Late":    ["Late Day1", "Late Day2"],
        "Washout": ["Washout Day1", "Washout Day2"],
    }
    for group in type_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ai, bj = labels.index(a), labels.index(b)
                mat[ai, bj] = mat[bj, ai] = 1
    return pd.DataFrame(mat, index=labels, columns=labels)


def build_day_model(labels: List[str]) -> pd.DataFrame:
    """Within-day pairs (any two epochs in the same day) = 1."""
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    day1_idx = [i for i, lab in enumerate(labels) if "Day1" in lab]
    day2_idx = [i for i, lab in enumerate(labels) if "Day2" in lab]
    for group in (day1_idx, day2_idx):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                mat[a, b] = mat[b, a] = 1
    return pd.DataFrame(mat, index=labels, columns=labels)


# ----------------------
# Plotting helpers
# ----------------------

def _annotation_matrix(df: pd.DataFrame) -> np.ndarray:
    """String annotations with blank diagonal."""
    arr = df.astype(int).astype(str).values
    np.fill_diagonal(arr, "")
    return arr


def plot_model(df: pd.DataFrame, title: str, out_base: Path, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df,
        cmap=BIN_CMAP,
        cbar=False,
        linewidths=0.5,
        square=True,
        annot=_annotation_matrix(df),
        fmt="",
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis="x", labelsize=15, rotation=90)
    ax.tick_params(axis="y", labelsize=15, rotation=0)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------
# Subject RSM loading / testing
# ----------------------

def _underscore_version(labels: Iterable[str]) -> List[str]:
    """Return labels with spaces replaced by underscores."""
    return [lab.replace(" ", "_") for lab in labels]


def read_square_matrix(path: Path) -> pd.DataFrame:
    """Read TSV/CSV with index column; keep as DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path, sep="\t", index_col=0)


def reorder_to_labels(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    """
    Reindex df to 'labels'. If df uses underscores or spaces differently,
    try to reconcile automatically.
    """
    # Exact attempt
    if set(df.index) == set(df.columns) and set(labels).issubset(df.index):
        return df.loc[labels, labels]

    # Try underscore variant of labels
    alt = _underscore_version(labels)
    if set(alt).issubset(df.index):
        return df.loc[alt, alt].set_axis(labels, axis=0).set_axis(labels, axis=1)

    # Convert df labels back to spaces
    df_space = df.copy()
    df_space.index = [i.replace("_", " ") for i in df_space.index]
    df_space.columns = [c.replace("_", " ") for c in df_space.columns]
    if set(labels).issubset(df_space.index):
        return df_space.loc[labels, labels]

    raise ValueError("Could not align subject RSM to provided labels. Check naming consistency.")


def vec_upper_tri(df: pd.DataFrame) -> np.ndarray:
    """Vectorize the upper triangle (excluding diagonal) as 1D array."""
    a = df.values
    iu = np.triu_indices_from(a, k=1)
    return a[iu]


# ---- metrics ----

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _compare_vecs(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    return _cosine(a, b) if metric == "cosine" else _pearson(a, b)


def test_model_against_subjects(
    model: pd.DataFrame,
    subject_rsms: List[pd.DataFrame],
    metric: str = "cosine",
) -> List[float]:
    """Return similarity for each subject between model and subject RSM (upper-tri vectors)."""
    m = vec_upper_tri(model)
    return [_compare_vecs(m, vec_upper_tri(r), metric) for r in subject_rsms]


# ----------------------
# CLI
# ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Build & test Time/Epoch/Day model RSMs.")
    # Figure outputs
    ap.add_argument("--out-dir", required=True, help="Directory to save model figures and results.")
    ap.add_argument("--basename", default="models", help="Base filename (default: models).")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300).")

    # Subject RSM inputs
    ap.add_argument("--subjects-dir", required=True, help="Directory containing subject RSM files.")
    ap.add_argument("--pattern", default="*_corr_matrix.tsv",
                    help="Glob for subject RSM files (default: '*_corr_matrix.tsv').")

    # Labels
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                    help="Condition labels (order matters).")

    # Which models to run
    ap.add_argument("--models", nargs="+", choices=["time", "epoch", "day"],
                    default=["time", "epoch", "day"],
                    help="Which models to build/test (default: all).")

    # Metric
    ap.add_argument("--metric", choices=["pearson", "cosine"], default="cosine",
                    help="Similarity metric for testing (default: cosine).")
    return ap.parse_args()


# ----------------------
# Main
# ----------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(args.labels)

    # 1) Build & plot models
    built = {}
    if "time" in args.models:
        df_time = build_time_model(labels)
        plot_model(df_time, "Time", out_dir / f"{args.basename}_time_model_rsm", dpi=args.dpi)
        built["time"] = df_time

    if "epoch" in args.models:
        df_epoch = build_epoch_model(labels)
        plot_model(df_epoch, "Epoch", out_dir / f"{args.basename}_epoch_model_rsm", dpi=args.dpi)
        built["epoch"] = df_epoch

    if "day" in args.models:
        df_day = build_day_model(labels)
        plot_model(df_day, "Day", out_dir / f"{args.basename}_day_model_rsm", dpi=args.dpi)
        built["day"] = df_day

    # 2) Load subject RSMs
    subj_dir = Path(args.subjects_dir)
    files = sorted(subj_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No subject RSM files matched pattern '{args.pattern}' in {subj_dir}")

    subject_ids = []
    subject_rsms = []
    for f in files:
        df = read_square_matrix(f)
        df = reorder_to_labels(df, labels)
        subject_rsms.append(df)
        # Subject ID heuristic: token before first underscore
        subject_ids.append(f.stem.split("_")[0])

    # 3) Test each model vs each subject
    col_suffix = args.metric  # "cosine" or "pearson"
    results = {"subject": subject_ids}
    for name, model in built.items():
        vals = test_model_against_subjects(model, subject_rsms, metric=args.metric)
        results[f"{name}_{col_suffix}"] = vals

    results_df = pd.DataFrame(results)
    results_csv = out_dir / f"{args.basename}_model_test_results.csv"
    results_df.to_csv(results_csv, index=False)

    print("✅ Saved model figures and test results to:", out_dir)
    print("   Results CSV:", results_csv)


if __name__ == "__main__":
    main()
