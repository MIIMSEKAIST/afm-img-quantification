"""Pair detected maps with detected colour bars.

Rule (Referee 1, comment 3): a map is paired with the colour bar whose colour
gamut it is compatible with - the fraction of (sub-sampled) map pixels lying
within Delta-E < 20 of the bar's calibration strip - and, when several bars are
compatible, with the nearest one. A single bar may therefore be shared by
several maps. Maps without a compatible bar (compatibility < ``min_compat``)
are returned unpaired and are not quantified.

``mode="evidence"`` (revision R1, human verification of the corpus sample):
gamut compatibility alone does not discriminate between the bars of a figure
(same colormap everywhere), so the nearest compatible bar is accepted only when
the figure supplies positive evidence that it belongs to the map: either the
bar is *laid out* as the map's own bar (aligned with the map's long side, at
most ``layout_gap`` map widths away) or it is *exclusive* (no other map chose
it). Maps without such evidence are returned unpaired with ``reason =
"no_exclusive_bar"`` instead of being assigned to a distant bar.
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


def layout_evidence(map_box, bar_box, orientation=None, min_overlap=0.8, layout_gap=0.3):
    """True when the bar is laid out as the map's own bar: it overlaps the map's extent along
    the bar axis by >= ``min_overlap`` and lies within ``layout_gap`` map widths (heights) of the map."""
    mx0, my0, mx1, my1 = map_box
    bx0, by0, bx1, by1 = bar_box
    mw, mh = max(1.0, mx1 - mx0), max(1.0, my1 - my0)
    if orientation is None:
        orientation = "vertical" if (by1 - by0) >= (bx1 - bx0) else "horizontal"
    if orientation == "vertical":
        overlap = max(0.0, min(my1, by1) - max(my0, by0)) / mh
        gap = max(0.0, max(mx0 - bx1, bx0 - mx1)) / mw
    else:
        overlap = max(0.0, min(mx1, bx1) - max(mx0, bx0)) / mw
        gap = max(0.0, max(my0 - by1, by0 - my1)) / mh
    return bool(overlap >= min_overlap and gap <= layout_gap)


def pair_maps_to_bars(image_rgb, map_boxes, bars, min_compat=0.5, tie_margin=0.10, mode="nearest",
                      layout_gap=0.3, min_overlap=0.8):
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
    if mode == "evidence":
        chosen = [o["bar_idx"] for o in out]
        # owners: maps for which a bar is laid out as their own bar
        owners = {}
        for mi, (o, mb) in enumerate(zip(out, map_boxes)):
            if o["bar_idx"] is not None:
                bar = bars[o["bar_idx"]]
                if layout_evidence(mb, bar["box"], bar.get("orientation"), min_overlap, layout_gap):
                    owners.setdefault(o["bar_idx"], []).append(mi)
        for mi, (o, mb) in enumerate(zip(out, map_boxes)):
            if o["bar_idx"] is None:
                continue
            bi = o["bar_idx"]
            laid_out = mi in owners.get(bi, [])
            exclusive = chosen.count(bi) == 1
            in_row = row_evidence(mi, bi, map_boxes, bars, chosen, min_overlap)
            # a shared bar that is owned by another map is not shared with this map unless the layout evidence
            # (row / stack) includes that owner, i.e. the bar serves the whole group
            owned_by_other = bool(owners.get(bi)) and not laid_out
            o["evidence"] = {"layout": laid_out, "exclusive": exclusive, "row": in_row, "owned_by_other": owned_by_other}
            if not (laid_out or exclusive or in_row):
                o.update({"bar_idx": None, "reason": "no_exclusive_bar"})
    return out


def _aligned_axis(a, b, orientation, min_overlap):
    """Overlap of two boxes along the bar axis (rows for a vertical bar), as a fraction of the smaller extent."""
    if orientation == "vertical":
        ov = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / max(1.0, min(a[3] - a[1], b[3] - b[1]))
    else:
        ov = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return ov >= min_overlap


def row_evidence(mi, bi, map_boxes, bars, chosen, min_overlap=0.8, layout_gap=0.3):
    """Evidence that a bar shared by several maps is laid out for the group that map ``mi`` belongs to.

    Two layouts are recognised. *Row*: every sharing map is aligned with the bar along the bar axis (a row of
    maps with one vertical bar), the bar lies beyond all of them (at the end of the row) and no other bar lies
    between map ``mi`` and it. *Stack*: the sharing maps are aligned with each other across the bar axis (a
    column of maps beside one vertical bar), the bar is adjacent to the stack's extent and inside it along the
    bar axis, and no other bar lies between map ``mi`` and the bar's position in the stack."""
    bar = bars[bi]
    bb = bar["box"]
    orientation = bar.get("orientation") or ("vertical" if (bb[3] - bb[1]) >= (bb[2] - bb[0]) else "horizontal")
    group = [j for j, c in enumerate(chosen) if c == bi]
    if len(group) < 2:
        return False
    mb = map_boxes[mi]
    v = orientation == "vertical"
    # axis indices: a = along the bar axis (y for a vertical bar), c = across it
    a0, a1, c0, c1 = (1, 3, 0, 2) if v else (0, 2, 1, 3)
    others = [b2["box"] for k, b2 in enumerate(bars) if k != bi]

    def centre(box, lo_i, hi_i):
        return (box[lo_i] + box[hi_i]) / 2

    # ---- row layout
    if all(_aligned_axis(map_boxes[j], bb, orientation, min_overlap) for j in group):
        size = lambda b: b[c1] - b[c0]
        beyond = all(bb[c0] >= map_boxes[j][c1] - 0.05 * size(map_boxes[j]) for j in group) or \
                 all(bb[c1] <= map_boxes[j][c0] + 0.05 * size(map_boxes[j]) for j in group)
        lo, hi = min(mb[c0], bb[c0]), max(mb[c1], bb[c1])
        between = [b2 for b2 in others if _aligned_axis(b2, bb, orientation, 0.5) and lo < centre(b2, c0, c1) < hi]
        if beyond and not between:
            return True
    # ---- stack layout
    perp = "horizontal" if v else "vertical"
    if not all(_aligned_axis(map_boxes[j], mb, perp, min_overlap) for j in group):
        return False
    xs = [map_boxes[j] for j in group]
    ext_c0, ext_c1 = min(b[c0] for b in xs), max(b[c1] for b in xs)       # stack extent across the bar axis
    ext_a0, ext_a1 = min(b[a0] for b in xs), max(b[a1] for b in xs)       # stack extent along the bar axis
    w = max(1.0, max(b[c1] - b[c0] for b in xs))
    adjacent = max(0.0, max(ext_c0 - bb[c1], bb[c0] - ext_c1)) <= layout_gap * w and bb[a0] < ext_a1 and bb[a1] > ext_a0
    if not adjacent:
        return False
    # the map nearest to the bar along the stack is the bar's anchor; nothing may lie between it and map mi
    anchor = min(group, key=lambda j: abs(centre(map_boxes[j], a0, a1) - centre(bb, a0, a1)))
    am = map_boxes[anchor]
    lo, hi = min(mb[a0], am[a0]), max(mb[a1], am[a1])
    # another bar in the same bar column, between the anchor map and map mi along the stack
    between = [b2 for b2 in others if lo < centre(b2, a0, a1) < hi and _aligned_axis(b2, bb, perp, 0.3)]
    return not between
