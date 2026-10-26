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


_FINDER_CONFIGS = ((None, False), (2, False), (1, False), (None, 3), (2, 5))


def find_gradient_region(crop_rgb, fg_threshold=40, open_frac=0.04, roi=None, roi_margin=0.15, extend=True,
                         candidates=False):
    """Gradient strip of a colour-bar crop (see ``_find_gradient_region_once``). The search runs with the standard
    opening kernel, with smaller kernels (thin strips of 2-4 px are erased by the standard one) and with a closing
    step (white text written across the bar fragments the strip); every configuration that finds a region
    contributes a candidate, refined across the bar axis to the lines that behave like a gradient
    (``refine_across``). ``candidates=True`` returns the distinct candidates in search order; otherwise the
    candidate chosen by ``choose_candidate`` (the longest along the bar axis among those whose orientation agrees
    with an elongated ``roi``)."""
    found = []
    for k_override, close in _FINDER_CONFIGS:
        r = _find_gradient_region_once(crop_rgb, fg_threshold, open_frac, roi, roi_margin, extend, k_override, close)
        if r is not None and r not in found:
            found.append(r)
    if candidates:
        return found
    return choose_candidate(found, roi, roi_margin)


def choose_candidate(found, roi=None, roi_margin=0.15, longer=1.5):
    """The longest candidate along the bar axis (extent clipped to the roi plus its margin) among those whose
    orientation agrees with the roi when the roi is clearly elongated and whose width is comparable to the
    narrowest candidate of comparable length - but only when it is at least 1.5x longer than the candidate of
    the standard configuration (the first), which is kept otherwise."""
    if not found:
        return None
    ok = list(found)
    if roi is not None:
        rx0, ry0, rx1, ry1 = roi
        rw, rh = max(1, rx1 - rx0), max(1, ry1 - ry0)
        if max(rw, rh) >= 1.5 * min(rw, rh):
            exp = "vertical" if rh >= rw else "horizontal"
            ok = [c for c in ok if c[1] == exp]
            if not ok:
                return None

    def geom(c):
        (x0, y0, x1, y1), o = c
        if o == "vertical":
            a0, a1, w = y0, y1, x1 - x0
            if roi is not None:
                m = roi_margin * (roi[3] - roi[1])
                a0, a1 = max(a0, roi[1] - m), min(a1, roi[3] + m)
        else:
            a0, a1, w = x0, x1, y1 - y0
            if roi is not None:
                m = roi_margin * (roi[2] - roi[0])
                a0, a1 = max(a0, roi[0] - m), min(a1, roi[2] + m)
        return a1 - a0, w

    lengths = [geom(c)[0] for c in ok]
    longest = max(lengths)
    # a candidate much wider than the narrowest candidate of comparable length has swallowed the labels or a
    # neighbouring map (a closing step merges them with the strip): it is not the strip
    w_min = min(geom(c)[1] for c, l in zip(ok, lengths) if l >= 0.5 * longest)
    ok = [c for c in ok if geom(c)[1] <= max(1.5 * w_min, w_min + 4)]
    # the standard configuration (first) is kept unless a fallback candidate is much longer: a thin strip erased by
    # the standard opening or a strip fragmented by text comes back whole, a slightly longer candidate does not
    base = ok[0]
    best = max(ok, key=lambda c: geom(c)[0])
    return best if geom(best)[0] >= longer * geom(base)[0] else base


def refine_across(crop_rgb, bbox, orientation, diff=None, fg_threshold=40, min_width=3):
    """Across the bar axis, keep the longest run of lines (the columns of a vertical bar) that behave like a
    gradient strip. Each line is judged on its foreground rows (those differing from the page): the median
    absolute step between neighbouring rows must be within a few levels of the step a gradient of that contrast
    and length needs (3 + 5 * contrast / rows), the line must be foreground over at least 30% of the region and
    must not be a dark frame line. A map touching the bar's frame (its texture steps are larger), the frame
    itself and tick marks beside a thin strip are thereby removed from the region. Returns the input box when no
    such run of at least ``min_width`` lines exists."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    sub = crop_rgb[y0:y1, x0:x1].astype(float)
    if sub.shape[0] < 4 or sub.shape[1] < 2:
        return bbox
    if diff is None:
        _, diff = _background_and_diff(crop_rgb)
    d = diff[y0:y1, x0:x1]
    if orientation != "vertical":                                     # axis 0 = along the bar, axis 1 = across
        sub, d = sub.transpose(1, 0, 2), d.T
    L, n_lines = sub.shape[:2]
    if L < 4:
        return bbox
    lum = sub.mean(axis=2)
    fgm = d > fg_threshold
    steps = np.abs(np.diff(sub, axis=0)).max(axis=2)
    both = fgm[1:] & fgm[:-1]
    good = np.zeros(n_lines, bool)
    for x in range(n_lines):
        rows = lum[fgm[:, x], x]
        st = steps[both[:, x], x]
        if len(rows) < max(4, 0.3 * L) or len(st) < 3:
            continue
        rng = float(np.percentile(rows, 95) - np.percentile(rows, 5))
        thr = 3.0 + 5.0 * max(rng, 60.0) / float(len(rows))
        good[x] = float(np.median(st)) <= thr and float(np.percentile(rows, 75)) > 40.0
    best, start = None, None
    for i, g in enumerate(list(good) + [False]):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is None or best[1] - best[0] < min_width:
        return bbox
    lo, hi = best
    return (x0 + lo, y0, x0 + hi, y1) if orientation == "vertical" else (x0, y0 + lo, x1, y0 + hi)

def _find_gradient_region_once(crop_rgb, fg_threshold=40, open_frac=0.04, roi=None, roi_margin=0.15, extend=True,
                               k_override=None, close=False):
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
    if k_override is not None:
        k = int(k_override)
    if close:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((int(close), int(close)), np.uint8))
    fg_open = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) if k > 1 else fg
    if k_override is not None:
        fg_open = fg_open * fg                 # an even fallback kernel shifts the opened mask by a pixel: keep it inside fg
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg_open, connectivity=8)
    if n <= 1:
        return None
    comps = sorted([(int(s[cv2.CC_STAT_AREA]), int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]),
                     int(s[cv2.CC_STAT_WIDTH]), int(s[cv2.CC_STAT_HEIGHT])) for s in stats[1:]],
                   reverse=True)
    lab_all = st_all = None
    if roi is not None:
        fg_all = cv2.morphologyEx((diff > fg_threshold).astype(np.uint8), cv2.MORPH_OPEN,
                                  np.ones((k, k), np.uint8))
        _, lab_all, st_all, _ = cv2.connectedComponentsWithStats(fg_all, connectivity=8)

    # candidates in order of area; a candidate that is the clipped edge of a
    # larger image (e.g. the map next to the bar) is skipped, not fatal
    fallback = None
    for area, x, y, bw, bh in comps:
        if area < 0.02 * comps[0][0]:
            break
        vertical = bh >= bw
        from_clip = False
        x0, y0, x1, y1 = x, y, x + bw, y + bh
        # merge components aligned with this one on the short axis
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
            inside = lab_all[y0:y1, x0:x1]
            inside = inside[inside > 0]
            lbl = int(np.bincount(inside).argmax()) if inside.size else 0
            if lbl > 0:
                fw, fh = int(st_all[lbl, cv2.CC_STAT_WIDTH]), int(st_all[lbl, cv2.CC_STAT_HEIGHT])
                rx0, ry0, rx1, ry1 = roi
                mx, my = roi_margin * (rx1 - rx0), roi_margin * (ry1 - ry0)
                if orientation == "vertical":
                    clipped = x1 >= int(min(w, rx1 + mx)) - 1 or x0 <= int(max(0, rx0 - mx)) + 1
                    larger = fw > 1.5 * (x1 - x0) + 6 and fh > 0.5 * (y1 - y0)
                else:
                    clipped = y1 >= int(min(h, ry1 + my)) - 1 or y0 <= int(max(0, ry0 - my)) + 1
                    larger = fh > 1.5 * (y1 - y0) + 6 and fw > 0.5 * (x1 - x0)
                if clipped and larger:
                    # a map touching the bar merges with it into one component: keep the part that behaves like
                    # a gradient strip across the bar axis, if it separates from the clipping edge
                    ref = refine_across(crop_rgb, (x0, y0, x1, y1), orientation, diff, fg_threshold)
                    if ref == (x0, y0, x1, y1):
                        continue
                    if orientation == "vertical":
                        still = ref[2] >= int(min(w, rx1 + mx)) - 1 or ref[0] <= int(max(0, rx0 - mx)) + 1
                    else:
                        still = ref[3] >= int(min(h, ry1 + my)) - 1 or ref[1] <= int(max(0, ry0 - my)) + 1
                    if still:
                        continue
                    x0, y0, x1, y1 = ref
                    from_clip = True
        bbox = (x0, y0, x1, y1)
        # tick marks or a touching map attached to a thin strip widen its component: keep the gradient-like lines
        # when they are a clear minority of the component's width (a one- or two-pixel trim is not applied)
        ref = refine_across(crop_rgb, bbox, orientation, diff, fg_threshold)
        if ref != bbox:
            w_ref = (ref[2] - ref[0]) if orientation == "vertical" else (ref[3] - ref[1])
            w_box = (x1 - x0) if orientation == "vertical" else (y1 - y0)
            if w_ref <= 0.6 * w_box:
                bbox = ref
        if extend:
            bbox = _extend_along_axis(diff, bbox, orientation, thr=fg_threshold * 0.5)
        if from_clip and fallback is None:
            # the gradient-like part of a component merged with a map is taken only when no separate component of
            # comparable length follows: the smooth edge of a map beside the bar refines to a few gradient-like
            # lines as well, and the bar is then another, smaller component inside the box
            fallback = (bbox, orientation)
            continue
        if fallback is not None:
            fl = (fallback[0][3] - fallback[0][1]) if fallback[1] == "vertical" else (fallback[0][2] - fallback[0][0])
            L = (bbox[3] - bbox[1]) if orientation == "vertical" else (bbox[2] - bbox[0])
            if orientation != fallback[1] or L < 0.5 * fl:
                continue
        return bbox, orientation
    return fallback


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
