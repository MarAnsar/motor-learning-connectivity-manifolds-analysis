#!/usr/bin/env python3

"""

import argparse
import glob
import os
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from pyriemann.utils.mean import mean_riemann

# --- Safe import of optional project helpers; provide fallbacks if absent ---
try:
    from adaptman.utils import get_files as _get_files, display as _display  # type: ignore
except Exception:
    _get_files, _display = None, None


def display(msg: str):
    """Fallback display if project logger not available."""
    if _display is not None:
        return _display(msg)
    print(msg)


def get_files(args):
    """
    Fallback get_files if project helper not available.
    Usage: get_files([root_dir, pattern, pattern2, ...]) -> sorted list of paths
    """
    if _get_files is not None:
        return _get_files(args)
    if not isinstance(args, (list, tuple)) or len(args) < 2:
        raise ValueError("get_files expects [root_dir, pattern, ...]")
    root = args[0]
    patterns = args[1:]
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(root, pat)))
    return sorted(out)


# --- Gradient helpers ---
def _pca_gradients(x: np.ndarray, ref: np.ndarray | None = None, n_components: int = 5) -> GradientMaps:
    """PCA model (cosine kernel) with optional Procrustes alignment to a reference."""
    alignment = None if ref is None else "procrustes"
    gm = GradientMaps(n_components=n_components, approach="pca", kernel="cosine", alignment=alignment)
    gm.fit(x, reference=ref)
    return gm


def save_gradients(gradients: np.ndarray,
                   lambdas: np.ndarray,
                   regions: pd.Index,
                   out_prefix: str,
                   float_fmt: str | None = None,
                   save_threshold: float | None = 0.01):
    """Save gradients and eigenvalues; optionally drop low-variance components."""
    n_grad = gradients.shape[1]
    labels = [f"g{i}" for i in range(1, n_grad + 1)]

    grad_df = pd.DataFrame(gradients, index=regions, columns=labels)
    denom = float(lambdas.sum()) if float(lambdas.sum()) != 0.0 else 1.0
    eig_df = pd.DataFrame({"eigenvalues": lambdas, "proportion": lambdas / denom}, index=labels)

    if save_threshold is not None:
        keep = eig_df.index[eig_df["proportion"] > save_threshold]
        grad_df = grad_df[keep]

    grad_df.to_csv(out_prefix + "_gradient.tsv", sep="\t", float_format=float_fmt)
    eig_df.to_csv(out_prefix + "_eigenvalues.tsv", sep="\t", float_format=float_fmt)


def epoch_gradients(cmat_file: str, reference_grad: np.ndarray, out_dir: str, n_components: int = 5):
    """Compute aligned PCA gradients for a single connectivity matrix."""
    fname = os.path.split(cmat_file)[1]
    sub_id = fname[:6]  # matches upstream convention
    display(f"Processing {fname}")

    cmat = pd.read_table(cmat_file, index_col=0)
    regions = cmat.columns

    gm = _pca_gradients(cmat.values, ref=reference_grad, n_components=n_components)

    save_dir = os.path.join(out_dir, sub_id)
    os.makedirs(save_dir, exist_ok=True)

    # Drop '_cmat' and '.tsv' from the base name
    base = fname.replace("_cmat", "")
    if base.endswith(".tsv"):
        base = base[:-4]
    out_prefix = os.path.join(save_dir, base)

    # gm.aligned_ contains aligned gradients when a reference is provided
    save_gradients(gm.aligned_, gm.lambdas_, regions, out_prefix)


def dataset_gradient(cmats: list[str], out_dir: str, reference_cmat: str, n_jobs: int = 16, n_components: int = 5):
    """Compute and save reference gradients, then aligned gradients for each matrix."""
    os.makedirs(out_dir, exist_ok=True)

    # Reference gradients
    ref = pd.read_table(reference_cmat, index_col=0)
    gm_ref = _pca_gradients(ref.values, ref=None, n_components=n_components)
    ref_prefix = os.path.join(out_dir, "reference")
    save_gradients(gm_ref.gradients_, gm_ref.lambdas_, ref.columns, ref_prefix)

    # Aligned gradients for each matrix
    Parallel(n_jobs=n_jobs)(
        delayed(epoch_gradients)(cmat_file, gm_ref.gradients_, out_dir, n_components)
        for cmat_file in cmats
    )


def create_reference(dataset_dir: str,
                     ses: str = "ses-01",
                     specifier: str = "base",
                     mean: str = "geometric") -> str:
    """
    Create a reference connectivity matrix from a subset of files in dataset_dir.

    Parameters
    ----------
    dataset_dir : str
        Directory of centered, extracted VAN/SM matrices (e.g., outputs/VAN_SM/connectivity_extract-centered).
    ses : str
        Session substring (e.g., 'ses-01').
    specifier : str
        Substring in filenames ('base', 'early', 'late', 'washout', etc.).
    mean : {'geometric', 'arithmetic'}
        Averaging method for the reference connectivity matrix.

    Returns
    -------
    str : path to saved reference matrix (TSV).
    """
    pattern = f"*/*{ses}*{specifier}*.tsv"
    files = get_files([dataset_dir, pattern])
    if not files:
        raise ValueError(f"No connectivity matrices found in {dataset_dir} matching pattern: {pattern}")

    mats = np.array([pd.read_table(f, index_col=0).values for f in files])
    labels = pd.read_table(files[0], index_col=0).columns

    if mean == "arithmetic":
        ref_mat = np.mean(mats, axis=0)
    else:
        ref_mat = mean_riemann(mats)

    out_path = os.path.join(dataset_dir, "reference_cmat.tsv")
    pd.DataFrame(ref_mat, index=labels, columns=labels).to_csv(out_path, sep="\t")
    return out_path


# --- CLI ---
def parse_args():
    ap = argparse.ArgumentParser(description="PCA gradients on centered, extracted VAN/SM connectivity matrices.")
    ap.add_argument("--dataset-dir", default="outputs/VAN_SM/connectivity_extract-centered",
                    help="Directory with centered, extracted VAN/SM matrices (TSVs).")
    ap.add_argument("--pattern", default="*/*.tsv",
                    help="Glob pattern relative to dataset-dir selecting matrices (e.g., '*/*ses-01*.tsv').")
    ap.add_argument("--ref-ses", default="ses-01",
                    help="Session substring for building the reference (e.g., 'ses-01').")
    ap.add_argument("--ref-specifier", default="base",
                    help="Specifier substring for the reference (e.g., 'base', 'early', 'late', 'washout').")
    ap.add_argument("--ref-mean", choices=["geometric", "arithmetic"], default="geometric",
                    help="Averaging method to build the reference connectivity matrix.")
    ap.add_argument("--out-dir", default="outputs/gradients/pca_vansm",
                    help="Output directory for gradients and eigenvalues.")
    ap.add_argument("--n-components", type=int, default=5,
                    help="Number of PCA components.")
    ap.add_argument("--n-jobs", type=int, default=16,
                    help="Parallel jobs for subject/epoch gradients.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Enumerate matrices to process
    cmats = get_files([args.dataset_dir, args.pattern])
    if not cmats:
        display("No connectivity matrices matched the pattern.")
        return

    # Build reference matrix and gradients
    ref_cmat = create_reference(
        dataset_dir=args.dataset_dir,
        ses=args.ref_ses,
        specifier=args.ref_specifier,
        mean=args.ref_mean
    )

    # Compute dataset gradients aligned to the reference
    dataset_gradient(
        cmats=cmats,
        out_dir=args.out_dir,
        reference_cmat=ref_cmat,
        n_jobs=args.n_jobs,
        n_components=args.n_components
    )


if __name__ == "__main__":
    main()
