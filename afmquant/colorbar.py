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
2. Foreground = pixels differing from the background; thin strokes (text, tick
   marks) are removed by a morphological opening. The gradient is the largest
   remaining component plus any further components aligned with it on the same
   axis (diverging colormaps whose white centre coincides with the page split
   the bar in two).
3. Orientation = long axis of the merged region.
4. The region is extended along its axis while the bar outline (the one- to
   three-pixel columns/rows just outside the gradient) remains non-background:
   this recovers bar ends whose colour equals the page (white end of 'hot',
   'gray', ...), which step 2 cannot see.
5. The strip is the per-position median colour over the central band of the
   bar (middle 50 % of its width), resampled to ``n`` positions.
6. Quality flags: gradient length, path smoothness and number of distinct
   colours, used downstream to reject non-gradient regions.

The OCR stage can further widen the axis extent to the outermost tick labels
(ticks always lie inside the bar), see :func:`afmquant.ocr.read_colorbar`.
"""
from __future__ import annotations

import numpy as np
import cv2
from colorspacious import cspace_convert


def _background_and_diff(crop_rgb, min_share=0.35):
    """Page background = most common (coarsely quantized) colour of the crop
    border; a mixed border (e.g. a neighbouring map inside a padded crop) falls
    back to white, the usual page colour."""
    border = np.concatenate([crop_rgb[0], crop_rgb[-1], crop_rgb[:, 0], crop_rgb[:, -1]]).astype(int)
    keys = (border[:, 0] // 32) * 64 + (border[:, 1] // 32) * 8 + (border[:, 2] // 32)
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[counts.argmax()]
    if counts.max() / len(keys) >= min_share:
        bg = np.median(border[keys == top], axis=0)
    else:
        bg = np.array([255.0, 255.0, 255.0])
    diff = np.abs(crop_rgb.astype(int) - bg.astype(int)).max(axis=2)
    return bg, diff


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _extend_along_axis(diff, bbox, orientation, thr, max_gap=2):
    """Grow the bbox along its long axis while the outline just outside the
    gradient band stays non-background on BOTH sides of the bar (text above or
    below the bar, e.g. an exponent label, marks only one side)."""
    H, W = diff.shape
    x0, y0, x1, y1 = bbox
    if orientation == "vertical":
        # the outline may lie just inside or just outside the component bbox
        left = list(range(max(0, x0 - 3), min(W, x0 + 3)))
        right = list(range(max(0, x1 - 3), min(W, x1 + 3)))
        if left and right:
            score = np.minimum(diff[:, left].max(axis=1), diff[:, right].max(axis=1))
        else:
            score = np.zeros(H)
        lo, hi = y0, y1
        g = 0
        for y in range(y0 - 1, -1, -1):
            if score[y] >= thr:
                lo, g = y, 0
            else:
                g += 1
                if g > max_gap:
                    break
        g = 0
        for y in range(y1, H):
            if score[y] >= thr:
                hi, g = y + 1, 0
            else:
                g += 1
                if g > max_gap:
                    break
        return (x0, lo, x1, hi)
    top = list(range(max(0, y0 - 3), min(H, y0 + 3)))
    bottom = list(range(max(0, y1 - 3), min(H, y1 + 3)))
    if top and bottom:
        score = np.minimum(diff[top, :].max(axis=0), diff[bottom, :].max(axis=0))
    else:
        score = np.zeros(W)
    lo, hi = x0, x1
    g = 0
    for x in range(x0 - 1, -1, -1):
        if score[x] >= thr:
            lo, g = x, 0
        else:
            g += 1
            if g > max_gap:
                break
    g = 0
    for x in range(x1, W):
        if score[x] >= thr:
            hi, g = x + 1, 0
        else:
            g += 1
            if g > max_gap:
                break
    return (lo, y0, hi, y1)


def find_gradient_region(crop_rgb, fg_threshold=40, open_frac=0.04, roi=None, roi_margin=0.15):
    """Return ``((x0, y0, x1, y1), orientation)`` of the gradient strip inside a
    colour-bar crop, or ``None`` if no solid region is found.

    ``roi`` (x0, y0, x1, y1), when given (e.g. the detector's colour-bar box in
    crop coordinates), restricts the search for the gradient to that box
    (plus a margin) so that a neighbouring map inside the padded crop is never
    mistaken for the bar; the outline-based extension may still grow beyond it.
    """
    h, w = crop_rgb.shape[:2]
    if h < 4 or w < 4:
        return None
    _, diff = _background_and_diff(crop_rgb)
    fg = (diff > fg_threshold).astype(np.uint8)
    if roi is not None:
        rx0, ry0, rx1, ry1 = roi
        mx, my = roi_margin * (rx1 - rx0), roi_margin * (ry1 - ry0)
        keep = np.zeros_like(fg)
        keep[int(max(0, ry0 - my)):int(min(h, ry1 + my)), int(max(0, rx0 - mx)):int(min(w, rx1 + mx))] = 1
        fg = fg * keep
    k = max(3, int(round(open_frac * min(h, w))))
    fg_open = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg_open, connectivity=8)
    if n <= 1:
        return None
    comps = [(int(s[cv2.CC_STAT_AREA]), int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]),
              int(s[cv2.CC_STAT_WIDTH]), int(s[cv2.CC_STAT_HEIGHT])) for s in stats[1:]]
    area, x, y, bw, bh = max(comps)
    vertical = bh >= bw
    x0, y0, x1, y1 = x, y, x + bw, y + bh
    # merge components aligned with the largest one on the short axis
    for a, cx, cy, cw, ch in comps:
        if a < 0.05 * area or (cx, cy, cw, ch) == (x, y, bw, bh):
            continue
        if vertical:
            ov = _overlap(x, x + bw, cx, cx + cw) / max(1, min(bw, cw))
            similar = 0.5 <= cw / max(1, bw) <= 2.0
        else:
            ov = _overlap(y, y + bh, cy, cy + ch) / max(1, min(bh, ch))
            similar = 0.5 <= ch / max(1, bh) <= 2.0
        if ov >= 0.6 and similar:
            x0, y0 = min(x0, cx), min(y0, cy)
            x1, y1 = max(x1, cx + cw), max(y1, cy + ch)
    orientation = "vertical" if (y1 - y0) >= (x1 - x0) else "horizontal"
    if roi is not None:
        # the region must not be the edge of a larger image (e.g. the map next
        # to a falsely detected bar): compare with the unrestricted components
        fg_all = cv2.morphologyEx((diff > fg_threshold).astype(np.uint8), cv2.MORPH_OPEN,
                                  np.ones((k, k), np.uint8))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(fg_all, connectivity=8)
        inside = lab2[y0:y1, x0:x1]
        inside = inside[inside > 0]
        lbl = int(np.bincount(inside).argmax()) if inside.size else 0
        if lbl > 0:
            fw, fh = int(st2[lbl, cv2.CC_STAT_WIDTH]), int(st2[lbl, cv2.CC_STAT_HEIGHT])
            if fw > 1.5 * (x1 - x0) + 6 and fh > 0.8 * (y1 - y0) if orientation == "vertical" else \
               fh > 1.5 * (y1 - y0) + 6 and fw > 0.8 * (x1 - x0):
                return None
    bbox = _extend_along_axis(diff, (x0, y0, x1, y1), orientation, thr=fg_threshold * 0.5)
    return bbox, orientation


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


def extract_strip(crop_rgb, n=256, axis_extent=None, roi=None):
    """Full colour-bar strip extraction from a crop.

    Parameters
    ----------
    crop_rgb : ndarray (H, W, 3) uint8
    n : number of strip samples
    axis_extent : (lo, hi) or None
        Override of the gradient extent along the bar axis (pixel coordinates
        in the crop), e.g. widened to the outermost tick labels by the OCR stage.
    roi : (x0, y0, x1, y1) or None
        Prior location of the bar inside the crop (see find_gradient_region).

    Returns a dict with ``strip`` (n x 3 uint8, ordered top->bottom or
    left->right), ``bbox``, ``orientation``, ``quality`` and ``ok``; on failure
    ``ok`` is False and ``reason`` explains why.
    """
    found = find_gradient_region(crop_rgb, roi=roi)
    if found is None:
        return {"ok": False, "reason": "no solid gradient region"}
    bbox, orientation = found
    x0, y0, x1, y1 = bbox
    H, W = crop_rgb.shape[:2]
    if axis_extent is not None:
        lo, hi = int(round(axis_extent[0])), int(round(axis_extent[1]))
        if orientation == "vertical":
            y0, y1 = max(0, min(y0, lo)), min(H, max(y1, hi))
        else:
            x0, x1 = max(0, min(x0, lo)), min(W, max(x1, hi))
        bbox = (x0, y0, x1, y1)
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
