#!/usr/bin/env python3
"""
Extract functional connectivity (FC) rows for chosen seed parcels and keep only
targets that belong to the 7-network Schaefer system (all networks included).

Outputs are saved per seed, separately for Day 1 and Day 2.

Expected filenames (per subject, per epoch):
  Day 1 (ses-01):
    sub-XX_ses-01_task-rotation_cmat_base.tsv
    sub-XX_ses-01_task-rotation_cmat_early.tsv
    sub-XX_ses-01_task-rotation_cmat_late.tsv
    sub-XX_ses-01_task-washout_cmat_washout.tsv
  Day 2 (ses-02):
    sub-XX_ses-02_task-rotation_cmat_base.tsv
    sub-XX_ses-02_task-rotation_cmat_early.tsv
    sub-XX_ses-02_task-rotation_cmat_late.tsv
    sub-XX_ses-02_task-washout_cmat_washout.tsv

"""

import argparse
import os
import sys
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 7-NETWORK LABEL PREFIXES  (Schaefer 2018, 400p, 7-networks)
# ──────────────────────────────────────────────────────────────────────────────
VALID_PREFIXES = (
    # Visual
    "7Networks_LH_Vis_", "7Networks_RH_Vis_",
    # Dorsal Attention
    "7Networks_LH_DorsAttn_Post_", "7Networks_RH_DorsAttn_Post_",
    "7Networks_LH_DorsAttn_FEF_",  "7Networks_RH_DorsAttn_FEF_",
    "7Networks_LH_DorsAttn_PrCv_", "7Networks_RH_DorsAttn_PrCv_",
    # Limbic
    "7Networks_LH_Limbic_", "7Networks_RH_Limbic_",
    # Control
    "7Networks_LH_Cont_", "7Networks_RH_Cont_",
    # Default
    "7Networks_LH_Default_", "7Networks_RH_Default_",
    # Salience / Ventral Attention
    "7Networks_LH_SalVentAttn_", "7Networks_RH_SalVentAttn_",
    # Somatomotor
    "7Networks_LH_SomMot_", "7Networks_RH_SomMot_",
)

EPOCHS = ("base", "early", "late", "washout")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract seed-based FC to 7 Yeo networks for Day 1 and Day 2."
    )
    p.add_argument(
        "--day1-dir",
        default="PATH/TO/Day1/connectivity",
        help="Directory with per-subject Day 1 connectivity matrices (ses-01).",
    )
    p.add_argument(
        "--day2-dir",
        default="PATH/TO/Day2/connectivity",
        help="Directory with per-subject Day 2 connectivity matrices (ses-02).",
    )
    p.add_argument(
        "--out-dir",
        default="PATH/TO/output/FC_extraction_7networks",
        help="Output directory to write per-seed TSV files.",
    )
    p.add_argument(
        "--seeds",
        required=True,
        help=(
            "Comma-separated seed labels. "
            'Example: "7Networks_LH_SalVentAttn_Med_2,7Networks_RH_SalVentAttn_Med_3,7Networks_LH_SomMot_26". '
            "Use your own labels; examples are illustrative only."
        ),
    )
    p.add_argument(
        "--n-subjects",
        type=int,
        default=32,
        help="Number of subjects to scan (expects sub-01..sub-XX).",
    )
    return p.parse_args()


def seed_row_to_table(df: pd.DataFrame, seed: str) -> pd.DataFrame:
    """
    Convert a seed's FC row (Series) to a two-column table [Region, Value],
    filtering to targets that start with any VALID_PREFIXES.
    """
    if seed not in df.index:
        raise KeyError(f"Seed not found in matrix index: {seed}")

    # Series (targets) -> DataFrame with columns [Region, Value]
    row = df.loc[seed].reset_index()
    row.columns = ["Region", "Value"]
    # Keep only targets in the 7-network set
    row = row[row["Region"].str.startswith(VALID_PREFIXES)]
    return row


def epoch_filename(sub_id: str, ses: str, epoch: str) -> str:
    """
    Build file name based on session and epoch.
      rotation: sub-XX_ses-0Y_task-rotation_cmat_<epoch>.tsv
      washout : sub-XX_ses-0Y_task-washout_cmat_washout.tsv
    """
    if epoch == "washout":
        return f"{sub_id}_{ses}_task-washout_cmat_washout.tsv"
    else:
        return f"{sub_id}_{ses}_task-rotation_cmat_{epoch}.tsv"


def process_day(day_dir: str, ses_tag: str, day_label: str, seeds: list[str], n_subjects: int) -> dict[str, list[pd.DataFrame]]:
    """
    Iterate subjects × epochs for one day (Day 1 or Day 2).
    Return dict: seed -> list of DataFrames to be concatenated.
    """
    store: dict[str, list[pd.DataFrame]] = {s: [] for s in seeds}

    for i in range(1, n_subjects + 1):
        sub_id = f"sub-{i:02d}"
        sub_dir = os.path.join(day_dir, sub_id)
        if not os.path.isdir(sub_dir):
            print(f"[warn] Missing subject folder: {sub_dir}")
            continue

        for epoch in EPOCHS:
            fname = epoch_filename(sub_id, ses_tag, epoch)
            fpath = os.path.join(sub_dir, fname)
            if not os.path.exists(fpath):
                print(f"[miss] {fpath}")
                continue

            try:
                # Load symmetric connectivity matrix (labels in index & columns)
                cmat = pd.read_csv(fpath, sep="\t", index_col=0)
                # Sanity: ensure columns are strings (parcel labels)
                cmat.columns = cmat.columns.astype(str)
            except Exception as e:
                print(f"[error] Failed to read {fpath}: {e}")
                continue

            for seed in seeds:
                try:
                    tbl = seed_row_to_table(cmat, seed)
                except KeyError:
                    print(f"[warn] Seed not found ({seed}) in {fpath}")
                    continue

                # Add metadata
                tbl.insert(0, "Epoch", epoch.capitalize())
                tbl.insert(0, "Subject", sub_id)
                store[seed].append(tbl)

    # Report basic counts
    for seed, parts in store.items():
        print(f"[{day_label}] {seed}: {sum(len(t) for t in parts)} rows across {len(parts)} tables")

    return store


def save_outputs(per_seed_tables: dict[str, list[pd.DataFrame]], out_dir: str, day_label: str):
    """
    Concatenate and save per-seed outputs for a given day.
    """
    os.makedirs(out_dir, exist_ok=True)
    for seed, parts in per_seed_tables.items():
        if not parts:
            continue
        df_out = pd.concat(parts, ignore_index=True)
        safe_seed = seed.replace("/", "_")
        out_file = os.path.join(out_dir, f"FC_{safe_seed}_{day_label}.tsv")
        df_out.to_csv(out_file, sep="\t", index=False)
        print(f"[save] {out_file}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("No seeds provided. Use --seeds 'seedA,seedB,...'", file=sys.stderr)
        sys.exit(1)

    # Day 1 (ses-01)
    print("\n=== Day 1 (ses-01) ===")
    day1_tables = process_day(args.day1_dir, "ses-01", "Day1", seeds, args.n_subjects)
    save_outputs(day1_tables, args.out_dir, "Day1")

    # Day 2 (ses-02)
    print("\n=== Day 2 (ses-02) ===")
    day2_tables = process_day(args.day2_dir, "ses-02", "Day2", seeds, args.n_subjects)
    save_outputs(day2_tables, args.out_dir, "Day2")

    print("\n✅ Done extracting seed FC to all 7 networks for Day 1 and Day 2.")


if __name__ == "__main__":
    main()
