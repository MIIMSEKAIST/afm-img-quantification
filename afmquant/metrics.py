"""Reconstruction fidelity metrics.

All metrics are computed over the valid (confidently matched) pixels and, where
applicable, normalized by the ground-truth dynamic range so that they are
comparable across scans and materials.

Source: extracted verbatim (logic-preserving) from scripts/gold_validation.py.
The primary metric `nMAE_pct` is the range-normalized MAE reported in the paper
(formerly named `relative_MAE_pct` in the original script).
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance


def evaluate(gt, recon, valid_mask):
    """Compare a reconstructed map against ground truth.

    Parameters
    ----------
    gt : ndarray (H, W), float
        Ground-truth matrix.
    recon : ndarray (H, W), float
        Reconstructed matrix from the quantification engine.
    valid_mask : ndarray (H, W), bool
        Valid-pixel mask from the engine.

    Returns
    -------
    dict
        nMAE_pct        : range-normalized mean absolute error (%).
        mean_error_pct  : relative deviation of the reconstructed mean (%).
        std_error_pct   : relative deviation of the reconstructed std (%).
        normalized_EMD_pct : range-normalized earth mover's distance (%).
        SSIM            : structural similarity index on range-normalized maps.
    """
    g = gt[valid_mask]
    r = recon[valid_mask]
    rng = gt.max() - gt.min() + 1e-12

    rel_mae = float(np.abs(g - r).mean() / rng)          # range-normalized MAE
    mean_err = float(abs(g.mean() - r.mean()) / (abs(g.mean()) + 1e-12))
    std_err = float(abs(g.std() - r.std()) / (g.std() + 1e-12))
    emd = float(wasserstein_distance(g.flatten(), r.flatten()) / rng)

    # SSIM on range-normalized maps
    gn = (gt - gt.min()) / rng
    rn = (recon - gt.min()) / rng
    s = float(ssim(gn, rn, data_range=1.0))

    return {
        "nMAE_pct": rel_mae * 100,
        "mean_error_pct": mean_err * 100,
        "std_error_pct": std_err * 100,
        "normalized_EMD_pct": emd * 100,
        "SSIM": s,
    }
