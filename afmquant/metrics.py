"""Reconstruction fidelity metrics.

The five original metrics (nMAE_pct, mean_error_pct, std_error_pct,
normalized_EMD_pct, SSIM) are unchanged from the manuscript benchmark. The
additional keys were added in response to review:

* nMAE_IQR_pct / nMAE_SD_pct : MAE normalized by the scan's interquartile range /
  standard deviation instead of its full dynamic range (outlier-robust view of
  the error relative to the physically relevant variation).
* mean_error_signed_pct / std_error_signed_pct : signed versions of the mean and
  standard-deviation deviations (negative = reconstruction too low / too smooth).
* nMAE_allpix_pct : range-normalized MAE over ALL pixels, i.e. without excluding
  pixels rejected by the validity mask.
* IQR_over_range, levels_in_IQR : how much of the 256-level color encoding falls
  inside the scan's IQR (the information actually available in the figure).

Source: extracted (logic-preserving) from scripts/gold_validation.py.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance

N_LEVELS = 256  # color-bar / LUT quantization levels


def evaluate(gt, recon, valid_mask):
    """Compare a reconstructed map against ground truth.

    Parameters
    ----------
    gt : ndarray (H, W), float
        Ground-truth matrix.
    recon : ndarray (H, W), float
        Reconstructed matrix from the quantification engine.
    valid_mask : ndarray (H, W), bool
        Valid-pixel mask from the engine. Pixel-wise statistics are computed over
        valid pixels (as in the manuscript); ``nMAE_allpix_pct`` is computed over
        every pixel.

    Returns
    -------
    dict  (see module docstring)
    """
    gt = np.asarray(gt, dtype=float)
    recon = np.asarray(recon, dtype=float)
    g = gt[valid_mask]
    r = recon[valid_mask]

    rng = gt.max() - gt.min() + 1e-12
    q25, q75 = np.percentile(gt, [25, 75])
    iqr = (q75 - q25) + 1e-12
    sd = gt.std() + 1e-12

    mae = float(np.abs(g - r).mean())
    mae_all = float(np.abs(gt - recon).mean())
    rel_mae = mae / rng                                   # range-normalized MAE
    mean_err_signed = float((r.mean() - g.mean()) / (abs(g.mean()) + 1e-12))
    std_err_signed = float((r.std() - g.std()) / (g.std() + 1e-12))
    emd = float(wasserstein_distance(g.flatten(), r.flatten()) / rng)

    # SSIM on range-normalized maps (full maps, as in the manuscript)
    gn = (gt - gt.min()) / rng
    rn = (recon - gt.min()) / rng
    s = float(ssim(gn, rn, data_range=1.0))

    return {
        # --- original manuscript metrics (unchanged definitions) ---
        "nMAE_pct": rel_mae * 100,
        "mean_error_pct": abs(mean_err_signed) * 100,
        "std_error_pct": abs(std_err_signed) * 100,
        "normalized_EMD_pct": emd * 100,
        "SSIM": s,
        # --- added in revision ---
        "nMAE_IQR_pct": mae / iqr * 100,
        "nMAE_SD_pct": mae / sd * 100,
        "mean_error_signed_pct": mean_err_signed * 100,
        "std_error_signed_pct": std_err_signed * 100,
        "nMAE_allpix_pct": mae_all / rng * 100,
        "IQR_over_range": float(iqr / rng),
        "levels_in_IQR": float(iqr / (rng / (N_LEVELS - 1))),
        "gt_range": float(rng),
        "gt_IQR": float(iqr),
        "gt_SD": float(sd),
    }
