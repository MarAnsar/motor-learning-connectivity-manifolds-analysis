#!/usr/bin/env python3
"""
import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.connectome import ConnectivityMeasure
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import sqrtm, logm, expm, invsqrtm


def get_files(pattern: str):
    """Glob pattern -> sorted list of paths."""
    return sorted(glob(pattern))

def display(msg: str):
    """Simple progress logger."""
    print(msg, flush=True)

pjoin = os.path.join


# -----------------------
# Connectivity computation
# -----------------------

def _baseline_window_length():
    return 96

def _drop_nontask_samps(x: np.ndarray) -> np.ndarray:
    # Remove 4 samples from the start and 8 from the end
    return x[4:-8, :]

def _split_by_learning(x: np.ndarray):
    baseline_length = _baseline_window_length()
    early_start = 240
    early_length = 96
    baseline = x[:baseline_length]
    early = x[early_start:early_start + early_length]
    late = x[-baseline_length:]
    return [baseline, early, late]

def _split_by_washout(x: np.ndarray):
    washout_length = 96
    return x[:washout_length]

def compute_connectivity(timeseries: str, output_dir: str, float_fmt: str = '%1.8f'):
    fname = os.path.split(timeseries)[1]
    display(f"Computing connectivity: {fname}")

    data = pd.read_table(timeseries)  # columns = ROIs, rows = time points
    tseries = data.values
    regions = data.columns

    # Window selection by filename
    if 'rotation' in fname:
        tseries = _drop_nontask_samps(tseries)
        dataset = _split_by_learning(tseries)
    elif 'washout' in fname:
        tseries = _drop_nontask_samps(tseries)
        dataset = [_split_by_washout(tseries)]
    else:
        dataset = [tseries]

    # Covariance connectomes
    conn = ConnectivityMeasure(kind='covariance')
    connectivities = conn.fit_transform(dataset)

    # Subject folder (first 6 chars)
    subj_out = pjoin(output_dir, fname[:6])
    os.makedirs(subj_out, exist_ok=True)

    # File naming
    cmat_name = fname.split('_464data')[0] + '_cmat'
    if len(connectivities) == 3:
        suffixes = ['_base', '_early', '_late']
    elif 'washout' in fname:
        suffixes = ['_washout']
    else:
        suffixes = ['']

    for sfx, cmat in zip(suffixes, connectivities):
        out_path = pjoin(subj_out, cmat_name + sfx + '.tsv')
        pd.DataFrame(cmat, index=regions, columns=regions).to_csv(
            out_path, sep='\t', float_format=float_fmt
        )

def connectivity_analysis(input_data_dir: str, out_dir: str, njobs: int = 8, ses: str = None):
    os.makedirs(out_dir, exist_ok=True)
    if ses is None:
        timeseries = get_files(pjoin(input_data_dir, '*.tsv'))
    else:
        timeseries = get_files(pjoin(input_data_dir, f'*{ses}*.tsv'))
    # remove "rest" runs
    timeseries = [p for p in timeseries if 'rest' not in os.path.basename(p)]

    if not timeseries:
        display("No input .tsv files found (or all contained 'rest').")
        return

    Parallel(n_jobs=njobs)(
        delayed(compute_connectivity)(ts, out_dir) for ts in timeseries
    )


# -----------------------
# VAN & SM extraction
# -----------------------

TARGET_REGIONS = [
    # LH SomMot
    "7Networks_LH_SomMot_1", "7Networks_LH_SomMot_2", "7Networks_LH_SomMot_3", "7Networks_LH_SomMot_4", "7Networks_LH_SomMot_5",
    "7Networks_LH_SomMot_6", "7Networks_LH_SomMot_7", "7Networks_LH_SomMot_8", "7Networks_LH_SomMot_9", "7Networks_LH_SomMot_10",
    "7Networks_LH_SomMot_11", "7Networks_LH_SomMot_12", "7Networks_LH_SomMot_13", "7Networks_LH_SomMot_14", "7Networks_LH_SomMot_15",
    "7Networks_LH_SomMot_16", "7Networks_LH_SomMot_17", "7Networks_LH_SomMot_18", "7Networks_LH_SomMot_19", "7Networks_LH_SomMot_20",
    "7Networks_LH_SomMot_21", "7Networks_LH_SomMot_22", "7Networks_LH_SomMot_23", "7Networks_LH_SomMot_24", "7Networks_LH_SomMot_25",
    "7Networks_LH_SomMot_26", "7Networks_LH_SomMot_27", "7Networks_LH_SomMot_28", "7Networks_LH_SomMot_29", "7Networks_LH_SomMot_30",
    "7Networks_LH_SomMot_31", "7Networks_LH_SomMot_32", "7Networks_LH_SomMot_33", "7Networks_LH_SomMot_34", "7Networks_LH_SomMot_35",
    "7Networks_LH_SomMot_36", "7Networks_LH_SomMot_37",

    # LH SalVentAttn
    "7Networks_LH_SalVentAttn_ParOper_1", "7Networks_LH_SalVentAttn_ParOper_2", "7Networks_LH_SalVentAttn_ParOper_3", "7Networks_LH_SalVentAttn_ParOper_4",
    "7Networks_LH_SalVentAttn_TempOcc_1", "7Networks_LH_SalVentAttn_FrOperIns_1", "7Networks_LH_SalVentAttn_FrOperIns_2", "7Networks_LH_SalVentAttn_FrOperIns_3",
    "7Networks_LH_SalVentAttn_FrOperIns_4", "7Networks_LH_SalVentAttn_FrOperIns_5", "7Networks_LH_SalVentAttn_FrOperIns_6", "7Networks_LH_SalVentAttn_FrOperIns_7",
    "7Networks_LH_SalVentAttn_FrOperIns_8", "7Networks_LH_SalVentAttn_FrOperIns_9", "7Networks_LH_SalVentAttn_PFCl_1", "7Networks_LH_SalVentAttn_Med_1",
    "7Networks_LH_SalVentAttn_Med_2", "7Networks_LH_SalVentAttn_Med_3", "7Networks_LH_SalVentAttn_Med_4", "7Networks_LH_SalVentAttn_Med_5",
    "7Networks_LH_SalVentAttn_Med_6", "7Networks_LH_SalVentAttn_Med_7",

    # RH SomMot
    "7Networks_RH_SomMot_1", "7Networks_RH_SomMot_2", "7Networks_RH_SomMot_3", "7Networks_RH_SomMot_4", "7Networks_RH_SomMot_5",
    "7Networks_RH_SomMot_6", "7Networks_RH_SomMot_7", "7Networks_RH_SomMot_8", "7Networks_RH_SomMot_9", "7Networks_RH_SomMot_10",
    "7Networks_RH_SomMot_11", "7Networks_RH_SomMot_12", "7Networks_RH_SomMot_13", "7Networks_RH_SomMot_14", "7Networks_RH_SomMot_15",
    "7Networks_RH_SomMot_16", "7Networks_RH_SomMot_17", "7Networks_RH_SomMot_18", "7Networks_RH_SomMot_19", "7Networks_RH_SomMot_20",
    "7Networks_RH_SomMot_21", "7Networks_RH_SomMot_22", "7Networks_RH_SomMot_23", "7Networks_RH_SomMot_24", "7Networks_RH_SomMot_25",
    "7Networks_RH_SomMot_26", "7Networks_RH_SomMot_27", "7Networks_RH_SomMot_28", "7Networks_RH_SomMot_29", "7Networks_RH_SomMot_30",
    "7Networks_RH_SomMot_31", "7Networks_RH_SomMot_32", "7Networks_RH_SomMot_33", "7Networks_RH_SomMot_34", "7Networks_RH_SomMot_35",
    "7Networks_RH_SomMot_36", "7Networks_RH_SomMot_37", "7Networks_RH_SomMot_38", "7Networks_RH_SomMot_39", "7Networks_RH_SomMot_40",

    # RH SalVentAttn
    "7Networks_RH_SalVentAttn_TempOccPar_1", "7Networks_RH_SalVentAttn_TempOccPar_2", "7Networks_RH_SalVentAttn_TempOccPar_3", "7Networks_RH_SalVentAttn_TempOccPar_4",
    "7Networks_RH_SalVentAttn_TempOccPar_5", "7Networks_RH_SalVentAttn_TempOccPar_6", "7Networks_RH_SalVentAttn_TempOccPar_7", "7Networks_RH_SalVentAttn_PrC_1",
    "7Networks_RH_SalVentAttn_FrOperIns_1", "7Networks_RH_SalVentAttn_FrOperIns_2", "7Networks_RH_SalVentAttn_FrOperIns_3", "7Networks_RH_SalVentAttn_FrOperIns_4",
    "7Networks_RH_SalVentAttn_FrOperIns_5", "7Networks_RH_SalVentAttn_FrOperIns_6", "7Networks_RH_SalVentAttn_FrOperIns_7", "7Networks_RH_SalVentAttn_FrOperIns_8",
    "7Networks_RH_SalVentAttn_PFCl_1", "7Networks_RH_SalVentAttn_Med_1", "7Networks_RH_SalVentAttn_Med_2", "7Networks_RH_SalVentAttn_Med_3",
    "7Networks_RH_SalVentAttn_Med_4", "7Networks_RH_SalVentAttn_Med_5", "7Networks_RH_SalVentAttn_Med_6", "7Networks_RH_SalVentAttn_Med_7",
    "7Networks_RH_SalVentAttn_Med_8"
]

def extract_van_sm(base_path: str, output_path: str, float_fmt: str = "%1.8f"):
    os.makedirs(output_path, exist_ok=True)
    subjects = [d for d in sorted(os.listdir(base_path)) if os.path.isdir(pjoin(base_path, d))]
    if not subjects:
        display("No subject folders found for extraction.")
        return

    for subj in subjects:
        in_dir = pjoin(base_path, subj)
        out_dir = pjoin(output_path, subj)
        os.makedirs(out_dir, exist_ok=True)

        for fname in sorted(os.listdir(in_dir)):
            if not fname.endswith(".tsv"):
                continue
            in_file = pjoin(in_dir, fname)
            df = pd.read_csv(in_file, sep="\t", index_col=0)
            # exact ordering subset (will raise KeyError if labels are missing)
            df_van_sm = df.loc[TARGET_REGIONS, TARGET_REGIONS]
            df_van_sm.to_csv(pjoin(out_dir, fname), sep="\t", float_format=float_fmt)

    print("Extraction complete with correct region order.")


# -----------------------
# Riemannian centering
# -----------------------

def _to_tangent(s: np.ndarray, mean: np.ndarray) -> np.ndarray:
    p = sqrtm(mean)
    p_inv = invsqrtm(mean)
    return p @ logm(p_inv @ s @ p_inv) @ p

def _gl_transport(t: np.ndarray, sub_mean: np.ndarray, grand_mean: np.ndarray) -> np.ndarray:
    g = sqrtm(grand_mean) @ invsqrtm(sub_mean)
    return g @ t @ g.T

def _from_tangent(t: np.ndarray, grand_mean: np.ndarray) -> np.ndarray:
    p = sqrtm(grand_mean)
    p_inv = invsqrtm(grand_mean)
    return p @ expm(p_inv @ t @ p_inv) @ p

def center_cmat(c: np.ndarray, sub_mean: np.ndarray, grand_mean: np.ndarray) -> np.ndarray:
    """Center a single SPD covariance matrix via tangent mapping and GL transport."""
    t = _to_tangent(c, sub_mean)
    tc = _gl_transport(t, sub_mean, grand_mean)
    return _from_tangent(tc, grand_mean)

def center_subject(sub_cmats: np.ndarray, grand_mean: np.ndarray) -> np.ndarray:
    """Center all matrices from one subject with respect to the grand mean."""
    sub_mean = mean_riemann(sub_cmats)
    return np.array([center_cmat(c, sub_mean, grand_mean) for c in sub_cmats])

def _read_and_stack_cmats(file_list):
    """Load connectivity matrices into array [N, R, R] and return with ROI labels."""
    arr = np.array([pd.read_table(f, index_col=0).values for f in file_list])
    labels = pd.read_table(file_list[0], index_col=0).columns
    return arr, labels

def center_matrices(dataset_dir: str, ses: str = None, float_fmt: str = '%1.8f'):
    """
    Center all covariance matrices in dataset_dir and write to dataset_dir+'-centered'.
    Saves grand_mean.tsv at the root of the centered output directory.
    """
    if ses is None:
        cmats = get_files(pjoin(dataset_dir, '*/*.tsv'))
    else:
        cmats = [p for p in get_files(pjoin(dataset_dir, '*/*.tsv')) if ses in os.path.basename(p)]

    if not cmats:
        display("No connectivity matrices found for centering.")
        return

    out_dir = dataset_dir + '-centered'
    os.makedirs(out_dir, exist_ok=True)

    # Grand Riemannian mean
    all_cmats, roi_labels = _read_and_stack_cmats(cmats)
    display('Computing grand mean')
    grand_mean = mean_riemann(all_cmats)

    # Save grand mean
    pd.DataFrame(grand_mean, index=roi_labels, columns=roi_labels).to_csv(
        pjoin(out_dir, 'grand_mean.tsv'), sep='\t', float_format=float_fmt
    )

    # Center per subject (subject id = first 6 chars of filename)
    subs = np.unique([os.path.basename(x)[:6] for x in cmats])
    for s in subs:
        display(s)
        sub_files = [i for i in cmats if os.path.basename(i).startswith(s)]
        sub_cmats, _ = _read_and_stack_cmats(sub_files)
        centered_cmats = center_subject(sub_cmats, grand_mean)

        for i, fname in enumerate(sub_files):
            df = pd.DataFrame(centered_cmats[i], index=roi_labels, columns=roi_labels)
            out_name = fname.replace(dataset_dir, out_dir)
            os.makedirs(os.path.split(out_name)[0], exist_ok=True)
            df.to_csv(out_name, sep='\t', float_format=float_fmt)


# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Connectivity → VAN/SM extract → Riemannian centering (on extracted).")
    ap.add_argument("--input-data", default="data/464_32data",
                    help="Directory containing ROI time-series .tsv files.")
    ap.add_argument("--connectivity-out", default="outputs/connectivity",
                    help="Directory to write connectivity matrices.")
    ap.add_argument("--extract-out", default="outputs/VAN_SM/connectivity_extract",
                    help="Directory to write VAN/SM submatrices.")
    ap.add_argument("--njobs", type=int, default=8,
                    help="Parallel jobs for connectivity computation.")
    ap.add_argument("--ses", default=None,
                    help="Optional session substring filter (e.g., 'ses-01').")
    ap.add_argument("--skip-connectivity", action="store_true",
                    help="Skip connectivity computation.")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip VAN/SM extraction.")
    ap.add_argument("--skip-centering", action="store_true",
                    help="Skip centering step (on extracted matrices).")
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) Connectivity
    if not args.skip_connectivity:
        connectivity_analysis(args.input_data, args.connectivity_out, njobs=args.njobs, ses=args.ses)

    # 2) VAN/SM extraction
    if not args.skip_extract:
        extract_van_sm(args.connectivity_out, args.extract_out)

    # 3) Riemannian centering on the EXTRACTED VAN/SM matrices
    if not args.skip_centering:
        center_matrices(args.extract_out, ses=args.ses)  # writes to args.extract_out + '-centered'

if __name__ == "__main__":
    sys.exit(main())
