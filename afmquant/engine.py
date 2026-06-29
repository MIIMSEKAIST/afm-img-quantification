"""Color-bar-guided quantification engine (the validated core of the pipeline).

Given a color-rendered map image, the color bar sampled along its principal
axis, and the scale endpoints (vmin, vmax), the engine assigns a physical value
to every map pixel by nearest-color matching in CIELAB perceptual color space.

This module is intentionally dependency-light and free of any I/O so that it can
be imported and unit-tested in isolation. It is the function validated in the
manuscript against 104 ground-truth benchmark cases (nMAE 2.09 +/- 0.76 %).

Source: extracted verbatim (logic-preserving) from scripts/gold_validation.py.
"""
from __future__ import annotations

import numpy as np
from colorspacious import cspace_convert
from sklearn.neighbors import KDTree


def quantify_map(map_rgb, colorbar_strip_rgb, vmin, vmax, distance_threshold=20.0):
    """Reconstruct numerical values from a color-rendered map.

    Parameters
    ----------
    map_rgb : ndarray (H, W, 3), uint8
        The rendered map image (sRGB).
    colorbar_strip_rgb : ndarray (N, 3), uint8
        Colors sampled along the color-bar principal axis, ordered from the
        vmax end to the vmin end.
    vmin, vmax : float
        Scale endpoints read from the color-bar labels (e.g. OCR output).
    distance_threshold : float, default 20.0
        Maximum CIELAB (Delta-E) nearest-neighbour distance for a pixel to be
        considered valid. Pixels above this (annotations, scale bars, text
        overlays) are flagged invalid and excluded from the reconstruction.

    Returns
    -------
    value_map : ndarray (H, W), float
        Reconstructed physical values.
    valid_mask : ndarray (H, W), bool
        True where the pixel was confidently matched to the color bar.
    qc : dict
        Quality-control summary (valid pixel ratio, mean CIELAB distance).
    """
    # 1. color-bar colors -> CIELAB
    cb_lab = cspace_convert(colorbar_strip_rgb.astype(float), "sRGB255", "CIELab")

    # 2. position -> value (top = vmax, bottom = vmin; linear)
    n = len(colorbar_strip_rgb)
    values = np.linspace(vmax, vmin, n)

    # 3. KD-tree over calibration colors
    tree = KDTree(cb_lab)

    # 4. map pixels -> CIELAB -> nearest calibration color
    H, W = map_rgb.shape[:2]
    map_lab = cspace_convert(map_rgb.reshape(-1, 3).astype(float), "sRGB255", "CIELab")
    dist, idx = tree.query(map_lab, k=1)
    value_map = values[idx.flatten()].reshape(H, W)
    dist_map = dist.flatten().reshape(H, W)

    # 5. validity mask from perceptual distance
    valid_mask = dist_map < distance_threshold

    qc = {
        "valid_pixel_ratio": float(valid_mask.mean()),
        "mean_cielab_distance": float(dist_map[valid_mask].mean()) if valid_mask.any() else None,
    }
    return value_map, valid_mask, qc
