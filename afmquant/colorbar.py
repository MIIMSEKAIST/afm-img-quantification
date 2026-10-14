"""Locate the colour gradient inside a detected colour-bar crop and sample it
along its principal axis.

This is the step between detection and quantification that review found
missing from the released code: a detected colour-bar region (which usually
contains the gradient strip, its outline, tick marks and tick labels) is turned
into a clean calibration strip (N x 3, uint8) ordered from one end of the bar to
the other. The numeric scale is attached separately by :mod:`afmquant.ocr`.

Method (deterministic, no learned components)
---------------------------------------------
1. Background colour = median of the crop border.
2. Foreground mask = pixels differing from the background; thin strokes (text,
   tick marks, outline) are removed by a morphological opening, leaving the
   solid gradient as the largest connected component.
3. Orientation = long axis of that component.
4. The strip is the per-position median colour over the central band of the
   bar (middle 50 % of its width), resampled to ``n`` positions along the axis.
5. Quality flags: gradient length, path smoothness and number of distinct
   colours, used downstream to reject non-gradient regions.

Known limitation: a colormap end that coincides with the background colour
(e.g. the white end of 'hot' on a white page) is not separable from the page
and is truncated; the value scale is still anchored by the OCR tick positions,
so the effect is confined to the extreme values.
"""
from __future__ import annotations

import numpy as np
import cv2
from colorspacious import cspace_convert


def find_gradient_region(crop_rgb, fg_threshold=40, open_frac=0.04):
    """Return ``((x0, y0, x1, y1), orientation)`` of the gradient strip inside a
    colour-bar crop, or ``None`` if no solid region is found."""
    h, w = crop_rgb.shape[:2]
    if h < 4 or w < 4:
        return None
    border = np.concatenate([crop_rgb[0], crop_rgb[-1], crop_rgb[:, 0], crop_rgb[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.abs(crop_rgb.astype(int) - bg.astype(int)).max(axis=2)
    fg = (diff > fg_threshold).astype(np.uint8)

    k = max(3, int(round(open_frac * min(h, w))))
    fg_open = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg_open, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
    bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
    orientation = "vertical" if bh >= bw else "horizontal"
    return (x, y, x + bw, y + bh), orientation


def sample_strip(crop_rgb, bbox, orientation, n=256, band=0.5):
    """Median colour along the bar axis over the central ``band`` of its width,
    resampled to ``n`` samples. Order: top->bottom (vertical) or left->right
    (horizontal)."""
    x0, y0, x1, y1 = bbox
    if orientation == "vertical":
        cw = x1 - x0
        a = x0 + int(cw * (1 - band) / 2)
        b = max(a + 1, x1 - int(cw * (1 - band) / 2))
        line = np.median(crop_rgb[y0:y1, a:b].astype(float), axis=1)   # (L, 3)
    else:
        ch = y1 - y0
        a = y0 + int(ch * (1 - band) / 2)
        b = max(a + 1, y1 - int(ch * (1 - band) / 2))
        line = np.median(crop_rgb[a:b, x0:x1].astype(float), axis=0)   # (L, 3)
    L = len(line)
    if L < 2:
        return None
    pos = np.linspace(0, L - 1, n)
    strip = np.stack([np.interp(pos, np.arange(L), line[:, c]) for c in range(3)], axis=1)
    return np.clip(np.round(strip), 0, 255).astype(np.uint8)


def strip_quality(strip_rgb):
    """Quality descriptors of a calibration strip."""
    lab = cspace_convert(strip_rgb.astype(float), "sRGB255", "CIELab")
    steps = np.linalg.norm(np.diff(lab, axis=0), axis=1)
    path = float(steps.sum())
    direct = float(np.linalg.norm(lab[-1] - lab[0]))
    return {
        "path_dE": path,
        "end_to_end_dE": direct,
        "tortuosity": path / (direct + 1e-9),
        "n_distinct": int(len(np.unique(strip_rgb, axis=0))),
        "max_step_dE": float(steps.max()) if len(steps) else 0.0,
    }


def extract_strip(crop_rgb, n=256):
    """Full colour-bar strip extraction from a crop.

    Returns a dict with keys ``strip`` (n x 3 uint8, ordered top->bottom or
    left->right), ``bbox``, ``orientation``, ``quality`` and ``ok``; on failure
    ``ok`` is False and ``reason`` explains why.
    """
    found = find_gradient_region(crop_rgb)
    if found is None:
        return {"ok": False, "reason": "no solid gradient region"}
    bbox, orientation = found
    x0, y0, x1, y1 = bbox
    long_px = (y1 - y0) if orientation == "vertical" else (x1 - x0)
    if long_px < 20:
        return {"ok": False, "reason": f"gradient region too short ({long_px} px)",
                "bbox": bbox, "orientation": orientation}
    strip = sample_strip(crop_rgb, bbox, orientation, n=n)
    if strip is None:
        return {"ok": False, "reason": "strip sampling failed", "bbox": bbox,
                "orientation": orientation}
    q = strip_quality(strip)
    ok = q["path_dE"] > 20 and q["n_distinct"] >= 8
    return {"ok": ok, "reason": None if ok else "region is not a colour gradient",
            "strip": strip, "bbox": bbox, "orientation": orientation,
            "length_px": int(long_px), "quality": q}
