"""Pair detected maps with detected colour bars.

Rule (Referee 1, comment 3): a map is paired with the colour bar whose colour
gamut it is compatible with - the fraction of (sub-sampled) map pixels lying
within Delta-E < 20 of the bar's calibration strip - and, when several bars are
compatible, with the nearest one. A single bar may therefore be shared by
several maps. Maps without a compatible bar (compatibility < ``min_compat``)
are returned unpaired and are not quantified.

``mode="evidence"`` (revision R1, after the first corpus sample):
gamut compatibility alone does not discriminate between the bars of a figure
(same colormap everywhere), so a bar is accepted only with positive evidence
that it belongs to the map. A map whose figure lays out a bar as its own (the
bar aligned with the map's side within ``layout_gap`` map widths; a bar laid
out between two maps goes to the map on the figure's usual side of its bars,
or to the clearly nearer map) takes that bar - the most compatible readable
one when several are laid out - and is held (``reason = "own_bar_unreadable"``)
when its own bar cannot be read or does not match the map's colours. Any other
map takes the nearest compatible bar only when the bar is *exclusive* (chosen
by no other map near it, laid out for no map, and near) or laid out for the map's row
or column of maps (``row_evidence``); otherwise it is returned unpaired with
``reason = "no_exclusive_bar"`` instead of being assigned to a distant bar.
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


def _orientation(bar):
    bb = bar["box"]
    return bar.get("orientation") or ("vertical" if (bb[3] - bb[1]) >= (bb[2] - bb[0]) else "horizontal")


def layout_evidence(map_box, bar_box, orientation=None, min_overlap=0.8, layout_gap=0.3, min_len=0.25):
    """True when the bar is laid out as the map's own bar: it lies within ``layout_gap`` map widths (heights) of
    the map across the bar axis and, along the bar axis, overlaps the map by ``min_overlap`` of the shorter of the
    two extents; a bar shorter than the map must also be centred within the map's extent and at least
    ``min_len`` of the map long."""
    mx0, my0, mx1, my1 = map_box
    bx0, by0, bx1, by1 = bar_box
    mw, mh = max(1.0, mx1 - mx0), max(1.0, my1 - my0)
    if orientation is None:
        orientation = "vertical" if (by1 - by0) >= (bx1 - bx0) else "horizontal"
    if orientation == "vertical":
        m0, m1, b0, b1, mlen = my0, my1, by0, by1, mh
        gap = max(0.0, max(mx0 - bx1, bx0 - mx1)) / mw
    else:
        m0, m1, b0, b1, mlen = mx0, mx1, bx0, bx1, mw
        gap = max(0.0, max(my0 - by1, by0 - my1)) / mh
    blen = max(1.0, b1 - b0)
    ov = max(0.0, min(m1, b1) - max(m0, b0))
    if blen >= mlen:
        along = ov >= min_overlap * mlen
    else:
        centre = (b0 + b1) / 2.0
        along = ov >= min_overlap * blen and blen >= min_len * mlen and m0 - 0.1 * mlen <= centre <= m1 + 0.1 * mlen
    return bool(along and gap <= layout_gap)


def layout_side(map_box, bar_box, orientation=None):
    """Side of the map on which a bar lies: 'right' / 'left' for a vertical bar, 'below' / 'above' for a
    horizontal one (by box centres)."""
    mx0, my0, mx1, my1 = map_box
    bx0, by0, bx1, by1 = bar_box
    if orientation is None:
        orientation = "vertical" if (by1 - by0) >= (bx1 - bx0) else "horizontal"
    if orientation == "vertical":
        return "right" if (bx0 + bx1) / 2.0 >= (mx0 + mx1) / 2.0 else "left"
    return "below" if (by0 + by1) / 2.0 >= (my0 + my1) / 2.0 else "above"


def _across_gap(map_box, bar_box, orientation):
    mx0, my0, mx1, my1 = map_box
    bx0, by0, bx1, by1 = bar_box
    if orientation == "vertical":
        return max(0.0, max(mx0 - bx1, bx0 - mx1))
    return max(0.0, max(my0 - by1, by0 - my1))


def own_bars(map_boxes, bars, min_overlap=0.8, layout_gap=0.3):
    """For every map the bars laid out as its own (``layout_evidence``), after resolving bars laid out for maps on
    both of their sides (a bar between two maps): the figure's convention decides when every bar laid out for a
    single map lies on the same side of its map (bars left of their maps, say); otherwise the clearly nearer map
    keeps the bar; otherwise both keep it. Returns (own, owners): the bar indices per map and the map indices per
    bar."""
    laid = {}
    for bi, bar in enumerate(bars):
        o = _orientation(bar)
        for mi, mb in enumerate(map_boxes):
            if layout_evidence(mb, bar["box"], o, min_overlap, layout_gap):
                laid.setdefault(bi, []).append(mi)
    votes = {}
    for bi, ms in laid.items():
        if len(ms) == 1:
            s = layout_side(map_boxes[ms[0]], bars[bi]["box"], _orientation(bars[bi]))
            votes[s] = votes.get(s, 0) + 1

    def unanimous(keys):
        v = [k for k in keys if votes.get(k, 0) > 0]
        return v[0] if len(v) == 1 else None

    conv = {"vertical": unanimous(("left", "right")), "horizontal": unanimous(("above", "below"))}
    owners = {}
    for bi, ms in laid.items():
        o = _orientation(bars[bi])
        if len(ms) >= 2:
            by_side = {}
            for mi in ms:
                by_side.setdefault(layout_side(map_boxes[mi], bars[bi]["box"], o), []).append(mi)
            if len(by_side) == 2:
                c = conv[o]
                if c in by_side:
                    ms = by_side[c]
                else:
                    gaps = {mi: _across_gap(map_boxes[mi], bars[bi]["box"], o) for mi in ms}
                    near_side = min(by_side, key=lambda s: min(gaps[m] for m in by_side[s]))
                    far_side = next(s for s in by_side if s != near_side)
                    g_near = min(gaps[m] for m in by_side[near_side])
                    g_far = min(gaps[m] for m in by_side[far_side])
                    size = min(min(map_boxes[m][2] - map_boxes[m][0], map_boxes[m][3] - map_boxes[m][1]) for m in ms)
                    if g_far - g_near > max(4.0, 0.03 * size):
                        ms = by_side[near_side]
        owners[bi] = list(ms)
    own = [[bi for bi, ms in owners.items() if mi in ms] for mi in range(len(map_boxes))]
    return own, owners


def pair_maps_to_bars(image_rgb, map_boxes, bars, min_compat=0.5, tie_margin=0.10, mode="nearest",
                      layout_gap=0.3, min_overlap=0.8, own_min_compat=0.8, exclusive_reach=1.5):
    """
    Parameters
    ----------
    image_rgb : ndarray (H, W, 3)
    map_boxes : list of (x0, y0, x1, y1)
    bars : list of dicts with keys ``box`` (x0, y0, x1, y1) and ``strip`` (N x 3)

    Returns
    -------
    list of dicts (one per map): map_idx, bar_idx (or None), compat, distance,
    candidates (compat per bar); in evidence mode also evidence and, for unpaired
    maps, reason.
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
    if mode != "evidence":
        return out
    own, owners = own_bars(map_boxes, bars, min_overlap, layout_gap)
    # 1. a map with bars of its own takes the most compatible readable one; it is held when none can be read, when
    #    the readable one does not match the map's colours, or when an unreadable own bar competes with a readable
    #    one that matches only loosely
    for mi, (o, mb) in enumerate(zip(out, map_boxes)):
        mine = own[mi]
        if not mine:
            continue
        compat = {c["bar_idx"]: c["compat"] for c in o["candidates"]}
        readable = [bi for bi in mine if bars[bi].get("strip") is not None]
        unread = [bi for bi in mine if bars[bi].get("strip") is None]
        best = max(readable, key=lambda bi: compat.get(bi, -1.0)) if readable else None
        cb = compat.get(best, 0.0) if best is not None else 0.0
        if best is not None and cb >= min_compat and (not unread or cb >= own_min_compat):
            o.update({"bar_idx": best, "compat": cb, "distance": box_distance(mb, bars[best]["box"]) / diag,
                      "evidence": {"layout": True, "exclusive": len(owners.get(best, [])) == 1, "row": False,
                                   "owned_by_other": False}})
        else:
            o.update({"bar_idx": None, "reason": "own_bar_unreadable", "own_bar_idx": (unread or mine)[0],
                      "evidence": {"layout": False, "exclusive": False, "row": False, "own_unreadable": bool(unread),
                                   "own_incompatible": best is not None and cb < min_compat}})
    chosen = [o["bar_idx"] for o in out]
    # 2. the other maps: the nearest compatible bar is accepted when it is exclusive or laid out for the map's
    #    row or column of maps
    for mi, (o, mb) in enumerate(zip(out, map_boxes)):
        if own[mi] or o["bar_idx"] is None:
            continue
        bi = o["bar_idx"]
        owned_by_other = bool(owners.get(bi))
        reach = lambda j: exclusive_reach * max(map_boxes[j][2] - map_boxes[j][0], map_boxes[j][3] - map_boxes[j][1])
        near = box_distance(mb, bars[bi]["box"]) <= reach(mi)
        # exclusive: no other map within its own reach of the bar chose it (a distant map does not compete)
        competitors = [j for j, c in enumerate(chosen) if c == bi and j != mi and box_distance(map_boxes[j], bars[bi]["box"]) <= reach(j)]
        exclusive = near and not competitors and not owned_by_other
        in_row = row_evidence(mi, bi, map_boxes, bars, chosen, min_overlap, layout_gap)
        o["evidence"] = {"layout": False, "exclusive": exclusive, "row": in_row, "owned_by_other": owned_by_other}
        if not (exclusive or in_row):
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
    between map ``mi`` and it. *Stack*: the sharing maps aligned with map ``mi`` across the bar axis (its column
    of maps beside one vertical bar) number at least two, the bar is adjacent to that column's extent (and reaches
    map ``mi`` along the axis when it is laid out as another map's own bar), and no other bar lies between map
    ``mi`` and the bar's position in the column."""
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
    # ---- stack layout: the sharing maps aligned with map mi across the bar axis form its column
    perp = "horizontal" if v else "vertical"
    col = [j for j in group if j == mi or _aligned_axis(map_boxes[j], mb, perp, min_overlap)]
    if len(col) < 2:
        return False
    xs = [map_boxes[j] for j in col]
    ext_c0, ext_c1 = min(b[c0] for b in xs), max(b[c1] for b in xs)       # column extent across the bar axis
    ext_a0, ext_a1 = min(b[a0] for b in xs), max(b[a1] for b in xs)       # column extent along the bar axis
    w = max(1.0, max(b[c1] - b[c0] for b in xs))
    adjacent = max(0.0, max(ext_c0 - bb[c1], bb[c0] - ext_c1)) <= layout_gap * w and bb[a0] < ext_a1 and bb[a1] > ext_a0
    # a bar laid out as the own bar of another map of the column is shared with map mi only when it also reaches
    # map mi along the axis (a short bar beside one map of the column is that map's own)
    reach = max(0.0, min(mb[a1], bb[a1]) - max(mb[a0], bb[a0])) / max(1.0, mb[a1] - mb[a0])
    if reach < 0.3 and any(layout_evidence(map_boxes[j], bb, orientation, min_overlap, layout_gap) for j in col if j != mi):
        return False
    if not adjacent:
        return False
    # the map nearest to the bar along the column is the bar's anchor; nothing may lie between it and map mi
    anchor = min(col, key=lambda j: abs(centre(map_boxes[j], a0, a1) - centre(bb, a0, a1)))
    am = map_boxes[anchor]
    lo, hi = min(mb[a0], am[a0]), max(mb[a1], am[a1])
    # another bar in the same bar column, between the anchor map and map mi along the column
    between = [b2 for b2 in others if lo < centre(b2, a0, a1) < hi and _aligned_axis(b2, bb, perp, 0.3)]
    return not between
