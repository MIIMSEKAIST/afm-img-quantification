"""Label-column colour-bar reader (revision R1; ``FigurePipeline(reader="v2")``).

The submitted reader (:func:`afmquant.ocr.read_colorbar`) accepted every number inside a wide corridor beside the
gradient strip as a tick label. On real figures this corridor frequently contained numbers that are not tick labels
of the bar: the axis of a neighbouring plot, the labels of the colour bar of an adjacent panel, or scale-bar text
inside a map. It also grew the gradient region through a unit label written across the bar, which shifted every
tick position. This reader addresses each failure by a specific rule:

1. *Gradient extension is verified.* The core gradient region is the non-background connected component; its
   outline-based extension along the bar axis is accepted only where every added row is uniform across the bar
   (not text) and continuous in colour with the core.
2. *Light ends are recovered.* When the core ends in a bright colour and the detector box extends beyond it, the
   strip is extended to the detector box: the end of a colormap that fades into the page colour is invisible to
   the component analysis but visible to the detector. The extension is then snapped to the outermost tick label
   if that label lies in the extended part (labels are centred on the bar ends).
3. *Numbers belong to the nearest bar.* Numbers inside a detected map, or nearer to another detected colour bar
   than to this one, are not tick labels of this bar.
4. *Tick labels form one column.* Tick labels sit on one side of the bar at one offset from it (their near edges
   are aligned). The column nearest to the strip with at least two labels spread along it is taken; numbers in
   farther columns (a neighbouring axis) are ignored. Numbers inside the detector's bar box are tried first.
   Labels written inside the bar take precedence over any column.
5. *End labels.* A label that lies entirely beyond an end of the strip labels that end (the convention of AFM
   software, which writes the maximum above and the minimum below the bar); an inside label within one text
   height of an end labels that end. Labels beyond the strip that continue the linear sequence of the labels
   inside it extend the strip to them (the bar fades into the page beyond the last visible gradient row).
6. *Small text.* When the labels are small, the crop is re-read at a larger scale, and the label column alone is
   re-read at a still larger scale; the pass whose labels agree best with a linear scale is kept. The crop is
   widened when the strip touches its border (bar cut by the detector box). Digit/letter confusions glued to a
   unit ("Onm") are corrected.

The scale is fitted as before (:func:`afmquant.ocr.fit_scale`); two end labels alone are accepted when a third
label disagrees. The unit is taken from the labels of the chosen column or the unit text at the bar
(lexicon-constrained when ``unit_mode="lexicon"``).
"""
from __future__ import annotations

import re

import numpy as np
from colorspacious import cspace_convert

from .colorbar import find_gradient_region, _extend_along_axis, _background_and_diff, sample_strip, strip_quality
from .ocr import (parse_number_unit, parse_offset, parse_unit, fit_scale, canonical_unit, _unit_score,
                  split_merged_decimals)

_GLUE_O = re.compile(r"^(-?)[OoDQ](?=[.,]\d|\s*[A-Za-zµμ°%])")     # 'Onm' -> '0nm', 'O.5' -> '0.5'
_GLUE_1 = re.compile(r"^(-?)[lI|](?=[.,]\d|\s*[A-Za-zµμ°%])")      # 'lnm' -> '1nm'
_TICK_TAIL = re.compile(r"^(-?\d+(?:[.,]\d+)?)\s*[-–—_]+$")               # '4-' -> '4' (tick mark read as a dash)


def fix_glued(text: str) -> str:
    """Repair OCR confusions typical of tick labels glued to a unit or a tick mark."""
    t = text.strip()
    t = _TICK_TAIL.sub(r"\1", t)
    t = _GLUE_O.sub(r"\g<1>0", t)
    t = _GLUE_1.sub(r"\g<1>1", t)
    t = re.sub(r"(?<=\d)[Oo](?=\d|[.,]|\s*[A-Za-zµμ°%]|$)", "0", t)
    return t


def _lab(rgb):
    return cspace_convert(np.asarray(rgb, float).reshape(-1, 3), "sRGB255", "CIELab").reshape(np.shape(rgb))


# ----------------------------------------------------------------------------------------------- strip geometry
def _rows(crop, bbox, vertical, band=0.6):
    """Central band of the bar as an array (L, w, 3) ordered along the bar axis."""
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    if vertical:
        cw = x1 - x0; a = x0 + int(cw * (1 - band) / 2); b = max(a + 1, x1 - int(cw * (1 - band) / 2))
        return crop[y0:y1, a:b].astype(float)
    ch = y1 - y0; a = y0 + int(ch * (1 - band) / 2); b = max(a + 1, y1 - int(ch * (1 - band) / 2))
    return np.transpose(crop[a:b, x0:x1].astype(float), (1, 0, 2))


def _uniform_rows(rows, max_spread=45.0, ref=None):
    """True for rows whose colour is uniform across the bar width (gradient rows; text rows are not). ``ref`` gives
    reference rows of the same bar: rows are then text-like only when their spread is far above the typical spread
    of the reference (a thin bar whose central band includes its frame has a large spread in every row)."""
    if rows.shape[1] < 2:
        return np.ones(len(rows), bool)
    spread = (np.percentile(rows, 95, axis=1) - np.percentile(rows, 5, axis=1)).max(axis=1)
    if ref is not None and ref.shape[1] >= 2 and len(ref) >= 4:
        typ = float(np.median((np.percentile(ref, 95, axis=1) - np.percentile(ref, 5, axis=1)).max(axis=1)))
        return spread <= max(max_spread, 2.0 * typ + 10.0)
    return spread <= max_spread


def _continuous(rows, max_step=35.0):
    """Max channel difference between consecutive row medians <= max_step."""
    med = np.median(rows, axis=1)
    return np.abs(np.diff(med, axis=0)).max(axis=1) <= max_step


def trim_core_ends(crop, core, vertical, bg, bg_thr=12.0, max_frac=0.4):
    """Shorten the core region at both ends while its rows are text-like (not uniform across the bar) or pure page
    background (a light bar end that merely approaches the page colour is kept)."""
    x0, y0, x1, y1 = core
    rows = _rows(crop, core, vertical)
    if len(rows) < 6:
        return core
    uniform = _uniform_rows(rows, ref=rows)
    nonbg = np.abs(np.median(rows, axis=1) - np.asarray(bg, float)).max(axis=1) > bg_thr
    good = uniform & nonbg
    lo, hi = 0, len(rows)
    limit = int(max_frac * len(rows))
    while lo < limit and not good[lo]:
        lo += 1
    while hi > len(rows) - limit and not good[hi - 1]:
        hi -= 1
    if lo == 0 and hi == len(rows):
        return core
    return (x0, y0 + lo, x1, y0 + hi) if vertical else (x0 + lo, y0, x0 + hi, y1)


def verified_strip_bbox(crop, roi):
    """Core gradient region (text/background rows trimmed from its ends) + verified extension.
    Returns (bbox, orientation, core_bbox) or None."""
    found = find_gradient_region(crop, roi=roi, extend=False)
    if found is None:
        return None
    core, orientation = found
    vertical = orientation == "vertical"
    bg, diff = _background_and_diff(crop)
    core = trim_core_ends(crop, core, vertical, bg)
    ext = _extend_along_axis(diff, core, orientation, thr=20)
    a0c, a1c = (core[1], core[3]) if vertical else (core[0], core[2])
    a0e, a1e = (ext[1], ext[3]) if vertical else (ext[0], ext[2])
    lo, hi = a0c, a1c
    ref = _rows(crop, core, vertical)
    if a0e < a0c:
        seg = _rows(crop, (core[0], a0e, core[2], a0c + 1) if vertical else (a0e, core[1], a0c + 1, core[3]), vertical)
        if len(seg) > 1 and _uniform_rows(seg, ref=ref).all() and _continuous(seg).all():
            lo = a0e
    if a1e > a1c:
        seg = _rows(crop, (core[0], a1c - 1, core[2], a1e) if vertical else (a1c - 1, core[1], a1e, core[3]), vertical)
        if len(seg) > 1 and _uniform_rows(seg, ref=ref).all() and _continuous(seg).all():
            hi = a1e
    bbox = (core[0], lo, core[2], hi) if vertical else (lo, core[1], hi, core[3])
    return bbox, orientation, core


def recover_light_end(crop, bbox, vertical, det_ac, min_gap=0.04, max_step=60.0):
    """Extend a strip whose bright end fades into the page: from the core end outward, row by row, while the rows
    are uniform across the bar (no text), close in colour to the end row and not a frame line, up to the detector
    box end. Returns (bbox, extension at the low end in px, extension at the high end in px)."""
    x0, y0, x1, y1 = bbox
    a0, a1 = (y0, y1) if vertical else (x0, x1)
    L = max(1, a1 - a0)
    rows = _rows(crop, bbox, vertical)
    if len(rows) < 4:
        return bbox, 0, 0
    med_rows = np.median(rows, axis=1)
    # the end colour is taken a few rows inside the core: the outermost row is a blend with the page or a frame
    k = min(4, len(rows) // 2)
    end_cols = (np.median(med_rows[1:1 + k], axis=0), np.median(med_rows[-1 - k:-1], axis=0))
    lab_ends = _lab(np.stack(end_cols))
    d0, d1 = det_ac
    new0, new1 = a0, a1
    H, W = crop.shape[:2]
    typ = float(np.median((np.percentile(rows, 95, axis=1) - np.percentile(rows, 5, axis=1)).max(axis=1)))
    for end in (0, 1):
        lstar = lab_ends[end, 0]
        gap = (a0 - d0) if end == 0 else (d1 - a1)
        if gap <= min_gap * L or lstar < 80:
            continue
        end_row = end_cols[end]
        end_lum = float(np.mean(end_row))
        # a frame line just outside the core marks a real end
        fb = (x0, max(0, a0 - 4), x1, a0) if (end == 0 and vertical) else (x0, a1, x1, min(H, a1 + 4)) if vertical else \
             (max(0, a0 - 4), y0, a0, y1) if end == 0 else (a1, y0, min(W, a1 + 4), y1)
        fr = _rows(crop, fb, vertical, band=0.9)
        if len(fr) and ((np.mean(np.median(fr, axis=1), axis=1) < 160) & (end_lum - np.mean(np.median(fr, axis=1), axis=1) > 60)).any():
            continue
        limit = d0 if end == 0 else d1
        seg_box = ((x0, int(limit), x1, a0) if end == 0 else (x0, a1, x1, int(limit))) if vertical else \
                  ((int(limit), y0, a0, y1) if end == 0 else (a1, y0, int(limit), y1))
        seg = _rows(crop, seg_box, vertical)
        if len(seg) == 0:
            continue
        order = range(len(seg) - 1, -1, -1) if end == 0 else range(len(seg))     # outward from the core end
        spread = (np.percentile(seg, 95, axis=1) - np.percentile(seg, 5, axis=1)).max(axis=1)
        med = np.median(seg, axis=1)
        n_ok = 0
        for i in order:
            if spread[i] > max(60.0, 2.0 * typ + 10.0) or np.abs(med[i] - end_row).max() > max_step:
                break
            n_ok += 1
        if n_ok == 0:
            continue
        if end == 0:
            new0 = a0 - n_ok
        else:
            new1 = a1 + n_ok
    out = (x0, new0, x1, new1) if vertical else (new0, y0, new1, y1)
    return out, a0 - new0, new1 - a1


# ----------------------------------------------------------------------------------------------- label selection
def _item_bbox(it):
    pts = np.asarray(it["box"], float)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _ac(box, vertical):
    x0, y0, x1, y1 = box
    return (y0, y1, x0, x1) if vertical else (x0, x1, y0, y1)


def _edge_dist(pt, box):
    x, y = pt; x0, y0, x1, y1 = box
    return float(np.hypot(max(0.0, x0 - x, x - x1), max(0.0, y0 - y, y - y1)))


def numeric_items(items):
    out = []
    for it in items:
        txt = fix_glued(it["text"])
        if parse_offset(txt) is not None:
            continue
        v, u = parse_number_unit(txt)
        if v is None:
            continue
        out.append({**it, "num": v, "unit_tag": u, "bbox": _item_bbox(it)})
    return out


def own_items(nums, det_box, other_bars, map_boxes):
    """Numbers inside a map, or nearer to another bar's box than to this one (and not inside this box), are not ours."""
    keep = []
    dx0, dy0, dx1, dy1 = det_box
    ex, ey = 0.1 * (dx1 - dx0), 0.1 * (dy1 - dy0)
    dropped = []
    for it in nums:
        c = (it["cx"], it["cy"])
        in_det = dx0 - ex <= c[0] <= dx1 + ex and dy0 - ey <= c[1] <= dy1 + ey
        if not in_det and any(_edge_dist(c, (mb[0] + 0.05 * (mb[2] - mb[0]), mb[1] + 0.05 * (mb[3] - mb[1]),
                                             mb[2] - 0.05 * (mb[2] - mb[0]), mb[3] - 0.05 * (mb[3] - mb[1]))) == 0 for mb in map_boxes):
            dropped.append((it["text"], "in_map"))
            continue
        d_own = _edge_dist(c, det_box)
        vertical_det = (dy1 - dy0) >= (dx1 - dx0)
        a_c, a_lo, a_hi = (c[1], dy0, dy1) if vertical_det else (c[0], dx0, dx1)
        within_along = a_lo - 0.1 * (a_hi - a_lo) <= a_c <= a_hi + 0.1 * (a_hi - a_lo)
        if d_own > 0 and not within_along and any(_edge_dist(c, ob) < d_own for ob in other_bars):
            dropped.append((it["text"], "other_bar"))
            continue
        keep.append(it)
    own_items.last_dropped = dropped
    return keep


def classify(nums, strip_bbox, vertical):
    """Each number relative to the strip: kind in {end, inside, col, far}, pos along the strip (0..1), side across
    the axis (-1 / 0 / +1), near-edge offset from the strip, axis centre, and whether its box overlaps the strip."""
    sa0, sa1, sc0, sc1 = _ac(strip_bbox, vertical)
    L, w = max(1.0, sa1 - sa0), max(1.0, sc1 - sc0)
    out = []
    for it in nums:
        a0, a1, c0, c1 = _ac(it["bbox"], vertical)
        ca, cc = (a0 + a1) / 2, (c0 + c1) / 2
        along = (ca - sa0) / L
        centred = abs(cc - (sc0 + sc1) / 2) <= max(0.5 * w, 4.0)
        beyond_lo, beyond_hi = a1 <= sa0 + 2, a0 >= sa1 - 2
        rec = {**it, "along": along, "ca": ca, "overlaps": not (beyond_lo or beyond_hi), "ha": (a1 - a0)}
        if centred:
            if (beyond_lo or along < 0.0) and along >= -0.8:
                rec.update(kind="end", pos=0.0)
            elif (beyond_hi or along > 1.0) and along <= 1.8:
                rec.update(kind="end", pos=1.0)
            elif 0.0 <= along <= 1.0:
                rec.update(kind="inside", pos=along)
            else:
                continue
            rec.update(side=0, off=-1.0)
            out.append(rec)
            continue
        side = 1 if cc > (sc0 + sc1) / 2 else -1
        off = (c0 - sc1) if side > 0 else (sc0 - c1)
        if beyond_lo and along >= -0.5:
            rec.update(kind="end", pos=0.0)
        elif beyond_hi and along <= 1.5:
            rec.update(kind="end", pos=1.0)
        elif -0.15 <= along <= 1.15:
            rec.update(kind="col", pos=along)
        elif -1.0 <= along <= 2.0:
            rec.update(kind="far", pos=along)      # beyond the strip: usable only if it continues the label sequence
        else:
            continue
        rec.update(side=side, off=off)
        out.append(rec)
    return out


def _clusters(side_items, tol):
    cl = []
    for c in sorted(side_items, key=lambda d: d["off"]):
        if cl and c["off"] - cl[-1][0]["off"] <= tol:
            cl[-1].append(c)
        else:
            cl.append([c])
    return cl


def choose_column(cands, tol, max_off):
    """Labels inside the bar; else the nearest label column with >= 2 numbers spread along the bar; else the
    nearest numbers of either side. Returns (chosen, how)."""
    centred = [c for c in cands if c["side"] == 0]
    inside = [c for c in centred if c["kind"] == "inside"]
    if len(inside) >= 2 and max(c["pos"] for c in inside) - min(c["pos"] for c in inside) >= 0.2:
        return centred, "inside"
    best = {}
    for side in (-1, 1):
        items = [c for c in cands if c["side"] == side and c["off"] <= max_off]
        for cl in _clusters(items, tol):
            near = [c for c in cl if c["kind"] != "far"]
            if len(near) >= 2 and max(c["pos"] for c in near) - min(c["pos"] for c in near) >= 0.2:
                best[side] = cl
                break
    if len(best) == 2:
        s = min(best, key=lambda k: (np.median([c["off"] for c in best[k]]), -len(best[k])))
        return best[s] + centred, "column"
    if len(best) == 1:
        return next(iter(best.values())) + centred, "column"
    weak = []
    for side in (-1, 1):
        items = [c for c in cands if c["side"] == side and c["off"] <= max_off]
        cl = _clusters(items, tol)
        if cl:
            weak += cl[0]
    return weak + centred, "weak"


def cut_at_column_text(items, bbox, core, vertical):
    """A recovered light end must not run through text standing on the bar column (unit label, end label): cut the
    strip at the far edge of the text nearest to the core. Returns the new bbox."""
    sa0, sa1, sc0, sc1 = _ac(bbox, vertical)
    ca0, ca1 = (core[1], core[3]) if vertical else (core[0], core[2])
    w = max(1.0, sc1 - sc0)
    lo, hi = sa0, sa1
    for it in items:
        a0, a1, c0, c1 = _ac(_item_bbox(it), vertical)
        overlap_c = min(c1, sc1) - max(c0, sc0)
        if overlap_c < min(0.5 * w, 0.3 * (c1 - c0)) or (a1 - a0) > 0.3 * (sa1 - sa0):
            continue
        if a1 <= ca0 + 1 and a0 >= sa0 - 1:              # text in the recovered part above the core
            lo = max(lo, a1 + 1)
        if a0 >= ca1 - 1 and a1 <= sa1 + 1:              # text in the recovered part below the core
            hi = min(hi, a0 - 1)
    if hi - lo < 0.5 * (sa1 - sa0):
        return bbox
    return (bbox[0], int(lo), bbox[2], int(hi)) if vertical else (int(lo), bbox[1], int(hi), bbox[3])


def place_ends(chosen, sa0, sa1, ext_lo, ext_hi, text_h):
    """Decide the strip ends from the labels and return (sa0, sa1, anchors).

    - an inside/column label overlapping the strip whose centre lies in a recovered (extended) part, or within one
      text height of a strip end, marks that end (labels are centred on the bar ends);
    - a label beyond the strip that continues the linear sequence of the labels inside it extends the strip to it;
    - labels entirely beyond an end (kind 'end') label that end without moving it.
    """
    near = [c for c in chosen if c["kind"] != "far"]
    th = max(text_h, 6.0)
    lo_c = [c for c in near if c["overlaps"] and c["kind"] != "end" and
            (c["ca"] - sa0 <= 1.5 * th or (c["kind"] == "inside" and c["ca"] - sa0 <= 1.0 * max(th, c["ha"])))]
    hi_c = [c for c in near if c["overlaps"] and c["kind"] != "end" and
            (sa1 - c["ca"] <= 1.5 * th or (c["kind"] == "inside" and sa1 - c["ca"] <= 1.0 * max(th, c["ha"])))]
    snap0 = min(c["ca"] for c in lo_c) if lo_c else sa0
    snap1 = max(c["ca"] for c in hi_c) if hi_c else sa1
    # with three or more labels the labels themselves decide: a label at the end of the bar makes the positions of
    # all labels consistent with one linear scale only when the end is placed at it, an interior tick only when it is not
    lab = [c for c in near if c["kind"] != "end"]
    if len(lab) >= 3 and (snap0 != sa0 or snap1 != sa1):
        def resid(e0, e1):
            Lx = max(1.0, e1 - e0)
            pos = np.array([(c["ca"] - e0) / Lx for c in lab]); val = np.array([c["num"] for c in lab])
            if np.ptp(pos) < 0.2:
                return np.inf
            b, a = np.polyfit(pos, val, 1)
            return float(np.abs(a + b * pos - val).max() / (abs(b) + 1e-12))
        options = {(sa0, sa1): resid(sa0, sa1), (snap0, sa1): resid(snap0, sa1), (sa0, snap1): resid(sa0, snap1), (snap0, snap1): resid(snap0, snap1)}
        best_r = min(options.values())
        (new0, new1) = (sa0, sa1) if options[(sa0, sa1)] <= best_r + 0.005 else min(options, key=options.get)
    else:
        new0, new1 = snap0, snap1
    inside = [c for c in near if new0 - 1 <= c["ca"] <= new1 + 1]
    far = [c for c in chosen if c["kind"] == "far" or (c["kind"] == "end" and not c["overlaps"] and c["side"] != 0)]
    if len(inside) >= 2 and far:
        Li = max(1.0, new1 - new0)
        pre = fit_scale([((c["ca"] - new0) / Li, float(c["num"]), float(c["conf"]), float(c["ca"])) for c in inside])
        if pre.get("ok"):
            a, b = pre["value_at_start"], pre["value_at_end"] - pre["value_at_start"]
            e0, e1 = new0, new1
            for c in far:
                p = (c["ca"] - e0) / Li
                pred = a + b * p
                if abs(pred - c["num"]) <= 0.05 * (abs(b) + 1e-12):
                    new0, new1 = min(new0, c["ca"]), max(new1, c["ca"])
                    c["kind"] = "col"
                elif c["kind"] == "end" and (c["num"] - pred) * (-1 if p < 0 else 1) * np.sign(b) > 0:
                    # the sequence would reach this value beyond the label itself: the bar ends at its end label
                    new0, new1 = min(new0, c["ca"]), max(new1, c["ca"])
    Ln = max(1.0, new1 - new0)
    anchors = []
    for c in chosen:
        if c["kind"] == "far":
            continue
        if c["kind"] == "end":
            p = 0.0 if c["ca"] < (new0 + new1) / 2 else 1.0
        elif new0 - 1 <= c["ca"] <= new1 + 1:
            p = float(np.clip((c["ca"] - new0) / Ln, 0.0, 1.0))
        else:
            p = (c["ca"] - new0) / Ln
        anchors.append((p, float(c["num"]), float(c["conf"]), float(c["ca"]), c["kind"], c.get("unit_tag"), c.get("text", "")))
    return new0, new1, anchors


def _symmetric_sign_repair(anchors):
    """Three labels with ~0 in the middle of the bar (AFM software labels a range symmetric about zero as
    +A / 0 / -A): linearity forces the two end magnitudes to be equal. When they are equal but have the same
    sign, the sign of one end was lost by the OCR; when they differ by a factor of 10 or 100, one end lost its
    decimal point. The label written with a decimal point is trusted and the other end is set to minus it."""
    if len(anchors) != 3:
        return anchors
    top, mid, bot = sorted(anchors, key=lambda a: a[0])
    A = max(abs(top[1]), abs(bot[1]), 1e-12)
    if not (abs(mid[1]) <= 0.02 * A and top[0] <= 0.1 and bot[0] >= 0.9 and 0.35 <= mid[0] <= 0.65 and top[1] != 0 and bot[1] != 0):
        return anchors

    def with_value(a, v):
        return (a[0], float(v)) + tuple(a[2:])

    mt, mb = abs(top[1]), abs(bot[1])
    if abs(mt - mb) <= 0.02 * A:
        if np.sign(top[1]) == np.sign(bot[1]):                       # sign lost on one end: the end farther from 0 keeps its sign
            if top[1] > 0:
                return [top, mid, with_value(bot, -mb)]
            return [with_value(top, mt), mid, bot]
        return anchors
    ratio = max(mt, mb) / min(mt, mb)
    if abs(ratio - 10) <= 0.5 or abs(ratio - 100) <= 5:            # decimal point lost on one end
        t_dec, b_dec = "." in str(top[6] if len(top) > 6 else ""), "." in str(bot[6] if len(bot) > 6 else "")
        if t_dec and not b_dec:
            return [top, mid, with_value(bot, -abs(top[1]) * (1 if top[1] > 0 else -1))]
        if b_dec and not t_dec:
            return [with_value(top, -abs(bot[1]) * (1 if bot[1] > 0 else -1)), mid, bot]
    return anchors


def _fit(anchors, scale):
    fit = fit_scale([a[:4] for a in anchors], scale)
    if not fit.get("ok") and len(anchors) == 3:
        rep = _symmetric_sign_repair(anchors)
        if rep is not anchors:
            fit3 = fit_scale([a[:4] for a in rep], scale)
            if fit3.get("ok"):
                fit3["sign_repaired"] = True
                return fit3
    if not fit.get("ok") and len(anchors) >= 3:
        ends = [a for a in anchors if a[0] in (0.0, 1.0)]
        interior = [a for a in anchors if a[0] not in (0.0, 1.0)]
        if len(ends) == 2 and ends[0][0] != ends[1][0] and all(a[2] < 0.9 for a in interior):
            fit2 = fit_scale([a[:4] for a in ends], scale)
            if fit2.get("ok"):
                fit2["reason"] = None
                fit2["end_labels_only"] = True
                return fit2
    return fit


# ----------------------------------------------------------------------------------------------- main reader
def _attempt(items, bbox, ext, vertical, det_box, other_bars, map_boxes, core=None):
    if core is not None and (ext[0] > 0 or ext[1] > 0):
        bbox = cut_at_column_text(items, bbox, core, vertical)
    sa0, sa1, sc0, sc1 = _ac(bbox, vertical)
    w = sc1 - sc0
    nums = own_items(numeric_items(items), det_box, other_bars, map_boxes)
    cands = classify(nums, bbox, vertical)
    text_h = float(np.median([c["h"] for c in cands])) if cands else 8.0
    tol = max(0.8 * text_h, 0.6 * w, 4.0)
    max_off = max(6.0 * w, 4.0 * text_h, 24.0)
    dx0, dy0, dx1, dy1 = det_box
    mx, my = 0.25 * (dx1 - dx0), 0.25 * (dy1 - dy0)
    L = max(1.0, sa1 - sa0)
    if vertical:
        py0, py1 = min(dy0 - my, sa0 - 0.5 * L), max(dy1 + my, sa1 + 0.5 * L); px0, px1 = dx0 - mx, dx1 + mx
    else:
        px0, px1 = min(dx0 - mx, sa0 - 0.5 * L), max(dx1 + mx, sa1 + 0.5 * L); py0, py1 = dy0 - my, dy1 + my
    inside_det = [c for c in cands if px0 <= c["cx"] <= px1 and py0 <= c["cy"] <= py1]
    chosen, how = choose_column(inside_det, tol, max_off)
    if how == "weak":
        chosen_all, how_all = choose_column(cands, tol, max_off)
        if how_all != "weak":
            chosen, how = chosen_all, how_all
    # a nearer column with a single number was skipped in favour of a farther one: the near labels may be unread
    near_skipped = None
    side_chosen = [c for c in chosen if c["side"] != 0]
    if side_chosen:
        off_chosen = float(np.median([c["off"] for c in side_chosen]))
        nearer = [c for c in cands if c["side"] != 0 and c["kind"] != "far" and c["off"] < 0.5 * off_chosen and c["off"] <= max_off]
        if nearer:
            near_skipped = 1 if np.mean([c["side"] for c in nearer]) > 0 else -1
    new0, new1, anchors = place_ends([dict(c) for c in chosen], sa0, sa1, ext[0], ext[1], text_h)
    offsets = [parse_offset(it["text"]) for it in items]
    offsets = [o for o in offsets if o is not None]
    scale = offsets[0] if offsets else 1.0
    tags = [a[5] for a in anchors if a[5]]
    unit = None
    if tags:
        counts = {}
        for u in tags:
            counts[u.lower()] = counts.get(u.lower(), 0) + 1
        unit = max(counts, key=counts.get)
        keep = [a for a in anchors if a[5] is None or a[5].lower() == unit]
        if 2 <= len(keep) < len(anchors):
            anchors = keep
    fit = _fit(anchors, scale)
    new_bbox = (bbox[0], int(round(new0)), bbox[2], int(round(new1))) if vertical else (int(round(new0)), bbox[1], int(round(new1)), bbox[3])
    return dict(fit=fit, anchors=anchors, chosen=chosen, how=how, unit=unit, scale=scale, bbox=new_bbox, text_h=text_h,
                cands=cands, items=items, near_skipped=near_skipped, dropped=list(getattr(own_items, "last_dropped", [])),
                bbox_in=bbox)


def _score(att):
    f = att["fit"]
    return (1 if f.get("ok") else 0, 0 if att.get("near_skipped") else 1, 0 if (f.get("end_labels_only") or f.get("sign_repaired")) else 1,
            int(f.get("n_inliers", 0) or 0), len(att["anchors"]))


def read_colorbar_v2(crop, ocr, roi, det_box, other_bars, map_boxes, n=256, unit_mode="raw"):
    """crop: padded RGB crop; roi/det_box: detector bar box in crop coordinates; other_bars, map_boxes: the other
    detected bars and the maps in crop coordinates. Returns a dict with the same keys as read_colorbar."""
    H, W = crop.shape[:2]
    vs = verified_strip_bbox(crop, roi)
    if vs is None:
        return {"ok": False, "reason": "no solid gradient region", "bbox": None, "orientation": None, "fit": None, "ocr_items": [],
                "touches_border": None}
    bbox, orientation, core = vs
    vertical = orientation == "vertical"
    det_ac = (int(roi[1]), int(roi[3])) if vertical else (int(roi[0]), int(roi[2]))
    bbox, ext_lo, ext_hi = recover_light_end(crop, bbox, vertical, det_ac)
    sa0, sa1, sc0, sc1 = _ac(bbox, vertical)
    L, w = sa1 - sa0, sc1 - sc0
    touches = ("lo" if sa0 <= 2 else "") + ("hi" if sa1 >= (H if vertical else W) - 2 else "")
    if L < 20:
        return {"ok": False, "reason": f"gradient region too short ({L} px)", "bbox": bbox, "orientation": orientation, "fit": None,
                "ocr_items": [], "touches_border": touches}
    base_scale = max(1.0, ocr.min_short_side / max(1, min(H, W)))
    ctx = (vertical, det_box, other_bars, map_boxes, core)

    # pass 1: the crop at the default scale
    items1 = ocr.read(crop)
    if not vertical:
        extra = []
        for it in items1:
            vals = split_merged_decimals(it["text"])
            if vals and len(vals) >= 2:
                xs = np.linspace(it["box"][0][0], it["box"][1][0], len(vals))
                extra += [{"text": f"{v:g}", "conf": it["conf"], "cx": float(x), "cy": it["cy"], "h": it["h"], "box": it["box"]}
                          for v, x in zip(vals, xs)]
        if not extra and len(numeric_items(items1)) < 3:
            extra = ocr.read_horizontal_labels(crop, bbox, items1)
        if extra:
            items1 = [it for it in items1 if parse_number_unit(fix_glued(it["text"]))[0] is None] + extra
    attempts = [_attempt(items1, bbox, (ext_lo, ext_hi), *ctx)]
    best = attempts[0]
    f1 = best["fit"]
    clean = (f1.get("ok") and not best["near_skipped"] and not f1.get("sign_repaired") and not f1.get("end_labels_only") and
             int(f1.get("n_inliers", 0) or 0) == len(best["anchors"]) >= 2)
    if not clean:
        # pass 2: the crop at a larger scale when the labels are small
        h_med = best["text_h"] if best["cands"] else 0.0
        if not best["cands"] or h_med * base_scale < 28:
            target = 36.0 / max(h_med, 5.0) if h_med else max(2.0, 28.0 / max(w, 4.0))
            scale2 = float(np.clip(max(target, 1.5 * base_scale), 1.5, 8.0))
            if scale2 > base_scale * 1.1:
                attempts.append(_attempt(ocr.read(crop, scale=scale2), bbox, (ext_lo, ext_hi), *ctx))
        # pass 3: the label column alone, at a scale that makes the labels large
        side_items = [c for c in best["chosen"] if c["side"] != 0] if best["chosen"] else []
        side = (1 if np.mean([c["side"] for c in side_items]) > 0 else -1) if side_items else 0
        if best["near_skipped"]:
            side = best["near_skipped"]
        margin = max(3.0 * w, 2.5 * best["text_h"], 16.0)
        dx0, dy0, dx1, dy1 = det_box
        if side == 0:                                  # labels centred on the bar (above/below it) or none found
            if vertical:
                cx0, cx1 = sc0 - margin, sc1 + margin; cy0, cy1 = sa0 - 0.6 * L, sa1 + 0.6 * L
            else:
                cy0, cy1 = sc0 - margin, sc1 + margin; cx0, cx1 = sa0 - 0.6 * L, sa1 + 0.6 * L
        elif vertical:
            cx0, cx1 = (sc1, max(dx1 + 0.25 * (dx1 - dx0), sc1 + margin)) if side > 0 else (min(dx0 - 0.25 * (dx1 - dx0), sc0 - margin), sc0)
            cy0, cy1 = sa0 - 0.3 * L, sa1 + 0.3 * L
        else:
            cy0, cy1 = (sc1, max(dy1 + 0.25 * (dy1 - dy0), sc1 + margin)) if side > 0 else (min(dy0 - 0.25 * (dy1 - dy0), sc0 - margin), sc0)
            cx0, cx1 = sa0 - 0.3 * L, sa1 + 0.3 * L
        cx0, cy0 = int(max(0, cx0)), int(max(0, cy0)); cx1, cy1 = int(min(W, cx1)), int(min(H, cy1))
        if cx1 - cx0 >= 6 and cy1 - cy0 >= 6:
            sub = np.ascontiguousarray(crop[cy0:cy1, cx0:cx1])
            h_est = best["text_h"] if best["cands"] else max(6.0, 0.8 * w)
            scale3 = float(np.clip(40.0 / max(h_est, 5.0), 2.0, 8.0))
            raw = ocr.read(sub, scale=scale3)
            items3 = [{**it, "cx": it["cx"] + cx0, "cy": it["cy"] + cy0, "box": [[p[0] + cx0, p[1] + cy0] for p in it["box"]]} for it in raw]
            items3 += [it for it in items1 if parse_number_unit(fix_glued(it["text"]))[0] is None]   # unit texts of pass 1
            attempts.append(_attempt(items3, bbox, (ext_lo, ext_hi), *ctx))
        best = max(attempts, key=_score)

    fit, bbox = best["fit"], best["bbox"]
    sa0, sa1, sc0, sc1 = _ac(bbox, vertical)
    L = sa1 - sa0
    items, unit = best["items"], best["unit"]
    dx0, dy0, dx1, dy1 = det_box
    mx, my = 0.25 * (dx1 - dx0), 0.25 * (dy1 - dy0)
    if unit is None:
        units = []
        for it in items:
            u = parse_unit(it["text"])
            if u is None:
                continue
            x, y = it["cx"], it["cy"]
            a, c = (y, x) if vertical else (x, y)
            c0d, c1d = (dx0 - mx, dx1 + mx) if vertical else (dy0 - my, dy1 + my)
            along = (a - sa0) / max(1.0, L)
            if c0d <= c <= c1d and -0.6 <= along <= 1.6:
                units.append((u, _unit_score(u), it["conf"], _edge_dist((x, y), det_box)))
        if units:
            unit = max(units, key=lambda t: (t[1], -t[3], t[2]))[0]
        elif vertical:
            rot = ocr.read(np.ascontiguousarray(np.rot90(crop, k=-1)))
            ru = [(parse_unit(it["text"]), it["conf"]) for it in rot]
            ru = [u for u in ru if u[0] is not None]
            if ru:
                unit = max(ru, key=lambda u: u[1])[0]
    unit_raw = unit
    if unit_mode == "lexicon":
        unit = canonical_unit(unit)
    strip = sample_strip(crop, bbox, orientation, n=n)
    q = strip_quality(strip) if strip is not None else None
    out = {"ok": bool(fit.get("ok")) and strip is not None, "reason": fit.get("reason") if strip is not None else "strip sampling failed",
           "unit": unit, "unit_raw": unit_raw, "orientation": orientation, "bbox": bbox, "fit": fit,
           "ocr_items": [(it["text"], round(it["conf"], 3)) for it in items], "strip_quality": q,
           "n_anchors": fit.get("n_anchors", 0), "label_column": best["how"], "touches_border": touches,
           "anchors": [(round(a[0], 3), a[1], a[4]) for a in best["anchors"]], "n_passes": len(attempts),
           "label_debug": [(c["text"], c["kind"], c["side"], round(c["off"], 1), round(c["pos"], 2)) for c in best["cands"]],
           "geom_debug": {"core": tuple(int(v) for v in core), "strip_in": tuple(int(v) for v in best["bbox_in"]),
                          "strip_out": tuple(int(v) for v in bbox), "det": tuple(int(v) for v in roi), "ext": (int(ext_lo), int(ext_hi)),
                          "dropped": best["dropped"]}}
    if out["ok"] and q is not None and not (q["path_dE"] > 20 and q["n_distinct"] >= 8):
        out.update({"ok": False, "reason": "region is not a colour gradient"})
    if out["ok"]:
        out.update({"strip": strip if fit["start_is_max"] else strip[::-1], "vmin": fit["vmin"], "vmax": fit["vmax"]})
    return out


def read_bars_v2(pipe, image_rgb, bar_boxes, maps):
    """Read every detected bar; the crop is widened along the bar axis when the strip touches the crop border."""
    H, W = image_rgb.shape[:2]
    all_boxes = [b["xyxy"] for b in bar_boxes]
    map_boxes = [m["xyxy"] for m in maps]
    out = []
    for i, b in enumerate(bar_boxes):
        x0, y0, x1, y1 = b["xyxy"]
        bw, bh = x1 - x0, y1 - y0
        vertical = bh >= bw
        pad_c = 0.5 * min(bw, bh) + 12.0                           # across the bar (labels lie inside the detector box)
        pad_a = 0.35 * max(bw, bh) + 10.0                          # along the bar
        r = None
        for _ in range(3):
            px, py = (pad_c, pad_a) if vertical else (pad_a, pad_c)
            cx0, cy0 = int(max(0, x0 - px)), int(max(0, y0 - py))
            cx1, cy1 = int(min(W, x1 + px)), int(min(H, y1 + py))
            crop = np.ascontiguousarray(image_rgb[cy0:cy1, cx0:cx1])
            shift = lambda bx: (bx[0] - cx0, bx[1] - cy0, bx[2] - cx0, bx[3] - cy0)
            roi = tuple(int(round(v)) for v in shift(b["xyxy"]))
            r = read_colorbar_v2(crop, pipe.ocr, roi, roi, [shift(o) for j, o in enumerate(all_boxes) if j != i],
                                 [shift(m) for m in map_boxes], unit_mode=pipe.unit_mode)
            r["crop_box"] = (cx0, cy0, cx1, cy1)
            t = r.get("touches_border") or ""
            grow = ("lo" in t and (cy0 > 0 if vertical else cx0 > 0)) or ("hi" in t and (cy1 < H if vertical else cx1 < W))
            if not grow or pad_a >= 1.5 * max(bw, bh):
                break
            pad_a *= 2.0
        r.update({"box": b["xyxy"], "det_conf": b["conf"]})
        out.append(r)
    return out
