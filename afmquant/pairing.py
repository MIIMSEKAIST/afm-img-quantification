"""Pair detected maps with detected colour bars.

Rule (Referee 1, comment 3): a map is paired with the colour bar whose colour
gamut it is compatible with - the fraction of (sub-sampled) map pixels lying
within Delta-E < 20 of the bar's calibration strip - and, when several bars are
compatible, with the nearest one. A single bar may therefore be shared by
several maps. Maps without a compatible bar (compatibility < ``min_compat``)
are returned unpaired and are not quantified.
"""
from __future__ import annotations

import numpy as np
from colorspacious import cspace_convert
from sklearn.neighbors import KDTree


def gamut_compatibility(map_rgb, strip_rgb, de_threshold=20.0, n_samples=4000, seed=0):
    """Fraction of map pixels within ``de_threshold`` of the strip in CIELAB."""
    pix = map_rgb.reshape(-1, 3)
    if len(pix) > n_samples:
        rng = np.random.default_rng(seed)
        pix = pix[rng.choice(len(pix), n_samples, replace=False)]
    lab_p = cspace_convert(pix.astype(float), "sRGB255", "CIELab")
    lab_s = cspace_convert(strip_rgb.astype(float), "sRGB255", "CIELab")
    d, _ = KDTree(lab_s).query(lab_p, k=1)
    return float((d.flatten() < de_threshold).mean())


def box_distance(a, b):
    """Edge-to-edge distance between two (x0, y0, x1, y1) boxes (0 if overlapping)."""
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return float(np.hypot(dx, dy))


def pair_maps_to_bars(image_rgb, map_boxes, bars, min_compat=0.5, tie_margin=0.10):
    """
    Parameters
    ----------
    image_rgb : ndarray (H, W, 3)
    map_boxes : list of (x0, y0, x1, y1)
    bars : list of dicts with keys ``box`` (x0, y0, x1, y1) and ``strip`` (N x 3)

    Returns
    -------
    list of dicts (one per map): map_idx, bar_idx (or None), compat, distance,
    candidates (compat per bar).
    """
    H, W = image_rgb.shape[:2]
    diag = float(np.hypot(W, H))
    out = []
    for mi, mb in enumerate(map_boxes):
        x0, y0, x1, y1 = [int(round(v)) for v in mb]
        crop = image_rgb[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        cands = []
        for bi, bar in enumerate(bars):
            if bar.get("strip") is None or crop.size == 0:
                continue
            c = gamut_compatibility(crop, bar["strip"])
            cands.append((bi, c, box_distance(mb, bar["box"]) / diag))
        best = None
        if cands:
            cmax = max(c for _, c, _ in cands)
            tied = [t for t in cands if t[1] >= cmax - tie_margin]
            best = min(tied, key=lambda t: t[2])           # nearest among compatible
            if best[1] < min_compat:
                best = None
        out.append({"map_idx": mi,
                    "bar_idx": None if best is None else best[0],
                    "compat": None if best is None else best[1],
                    "distance": None if best is None else best[2],
                    "candidates": [{"bar_idx": b, "compat": c, "distance": d} for b, c, d in cands]})
    return out
