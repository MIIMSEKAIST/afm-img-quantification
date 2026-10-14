"""Color-bar-guided quantification engine (the validated core of the pipeline).

Given a color-rendered map image, the color bar sampled along its principal
axis, and the scale endpoints (vmin, vmax), the engine assigns a physical value
to every map pixel by nearest-color matching in a chosen color space (CIELAB by
default; sRGB is available as a baseline).

This module is intentionally dependency-light and free of any I/O so that it can
be imported and unit-tested in isolation. The default configuration
(``color_space="cielab"``, ``interpolate=False``, ``distance_threshold=20``) is
the function validated in the manuscript against 104 ground-truth benchmark
cases (nMAE 2.09 +/- 0.76 %).

Source: extracted (logic-preserving) from scripts/gold_validation.py; the
``color_space`` and ``interpolate`` options were added for the baseline
comparison requested in review (they do not change the default behaviour).
"""
from __future__ import annotations

import numpy as np
from colorspacious import cspace_convert
from sklearn.neighbors import KDTree

COLOR_SPACES = ("cielab", "rgb")


def _to_space(rgb_flat, color_space):
    """sRGB (0-255) -> coordinates of the matching color space."""
    x = np.asarray(rgb_flat, dtype=float).reshape(-1, 3)
    if color_space == "cielab":
        return cspace_convert(x, "sRGB255", "CIELab")
    if color_space == "rgb":
        return x
    raise ValueError(f"color_space must be one of {COLOR_SPACES}, got {color_space!r}")


def _project_onto_polyline(pix, strip, values, idx):
    """Project each pixel onto the two strip segments adjacent to its nearest
    sample and interpolate the value along the chosen segment.

    Returns (values, distances) with the same length as ``pix``.
    """
    n = len(strip)
    best_val = values[idx].astype(float)
    best_d = np.linalg.norm(pix - strip[idx], axis=1)
    for a_idx, b_idx in ((idx - 1, idx), (idx, idx + 1)):
        ok = (a_idx >= 0) & (b_idx < n)
        if not ok.any():
            continue
        a = strip[np.clip(a_idx, 0, n - 1)]
        b = strip[np.clip(b_idx, 0, n - 1)]
        ab = b - a
        seg = (ab * ab).sum(axis=1)
        t = np.where(seg > 1e-12, ((pix - a) * ab).sum(axis=1) / np.maximum(seg, 1e-12), 0.0)
        t = np.clip(t, 0.0, 1.0)
        foot = a + t[:, None] * ab
        d = np.linalg.norm(pix - foot, axis=1)
        va = values[np.clip(a_idx, 0, n - 1)]
        vb = values[np.clip(b_idx, 0, n - 1)]
        v = va + t * (vb - va)
        better = ok & (d < best_d)
        best_d = np.where(better, d, best_d)
        best_val = np.where(better, v, best_val)
    return best_val, best_d


def quantify_map(map_rgb, colorbar_strip_rgb, vmin, vmax, distance_threshold=20.0,
                 color_space="cielab", interpolate=False):
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
    distance_threshold : float or None, default 20.0
        Maximum nearest-neighbour distance (CIELAB Delta-E for the default
        color space) for a pixel to be considered valid. Pixels above this
        (annotations, scale bars, text overlays) are flagged invalid. ``None``
        disables rejection (every pixel is valid); used for baseline comparisons
        in color spaces where the Delta-E threshold has no meaning.
    color_space : {"cielab", "rgb"}, default "cielab"
        Space in which nearest-color matching is performed. "rgb" is the
        Euclidean-sRGB baseline requested in review.
    interpolate : bool, default False
        If True, project each pixel onto the strip polyline between neighbouring
        samples and interpolate the value (continuous output instead of the
        256-level nearest-sample lookup).

    Returns
    -------
    value_map : ndarray (H, W), float
        Reconstructed physical values.
    valid_mask : ndarray (H, W), bool
        True where the pixel was confidently matched to the color bar.
    qc : dict
        Quality-control summary (valid pixel ratio, mean matching distance).
    """
    strip = _to_space(colorbar_strip_rgb, color_space)
    n = len(strip)
    values = np.linspace(vmax, vmin, n)          # top = vmax, bottom = vmin (linear)

    tree = KDTree(strip)
    H, W = map_rgb.shape[:2]
    pix = _to_space(map_rgb.reshape(-1, 3), color_space)
    dist, idx = tree.query(pix, k=1)
    idx = idx.flatten()
    dist = dist.flatten()

    if interpolate:
        vals, dist = _project_onto_polyline(pix, strip, values, idx)
    else:
        vals = values[idx]

    value_map = vals.reshape(H, W)
    dist_map = dist.reshape(H, W)

    if distance_threshold is None:
        valid_mask = np.ones((H, W), dtype=bool)
    else:
        valid_mask = dist_map < distance_threshold

    qc = {
        "valid_pixel_ratio": float(valid_mask.mean()),
        "mean_cielab_distance": float(dist_map[valid_mask].mean()) if valid_mask.any() else None,
        "color_space": color_space,
        "interpolate": bool(interpolate),
    }
    return value_map, valid_mask, qc
