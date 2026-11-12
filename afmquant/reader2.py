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
label disagrees. With ``end_labels="verbatim"`` (the released default of the pipeline) an end that carries its own
label takes that label's value instead of the fitted one (:func:`verbatim_ends`). The unit is taken from the labels of the chosen column or the unit text at the bar
(lexicon-constrained when ``unit_mode="lexicon"``).
"""
from __future__ import annotations

import re

import numpy as np
from colorspacious import cspace_convert

from .colorbar import (find_gradient_region, choose_candidate, _extend_along_axis, _background_and_diff, sample_strip,
                       strip_quality)
from .ocr import (_normalize, parse_number_unit, parse_offset, parse_unit, fit_scale, canonical_unit, _unit_score,
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


def main_body(crop, core, vertical, bg, bg_thr=12.0, gap_frac=0.05, minor_frac=0.3):
    """When a long run of page-coloured rows (at least ``gap_frac`` of the core, 8 px or more) separates a minor
    part of the core (at most ``minor_frac`` of the main part) from the rest - a label row merged into the
    region - the core is cut back to its main part. Diverging colour maps with a short white centre are not
    split."""
    x0, y0, x1, y1 = core
    rows = _rows(crop, core, vertical)
    if len(rows) < 12:
        return core
    page = np.abs(np.median(rows, axis=1) - np.asarray(bg, float)).max(axis=1) <= bg_thr
    runs, start = [], None
    for i, p in enumerate(list(page) + [True]):
        if not p and start is None:
            start = i
        elif p and start is not None:
            runs.append((start, i))
            start = None
    if len(runs) < 2:
        return core
    main = max(runs, key=lambda r: r[1] - r[0])
    lo, hi = main
    gap_min = max(8, int(round(gap_frac * len(rows))))
    # absorb neighbouring runs separated by a short gap or comparable in length (a white band inside the bar)
    changed = True
    while changed:
        changed = False
        for r in runs:
            if r[1] <= lo and (lo - r[1] < gap_min or (r[1] - r[0]) > minor_frac * (hi - lo)):
                lo, changed = min(lo, r[0]), True
            elif r[0] >= hi and (r[0] - hi < gap_min or (r[1] - r[0]) > minor_frac * (hi - lo)):
                hi, changed = max(hi, r[1]), True
    if (lo, hi) == (0, len(rows)):
        return core
    return (x0, y0 + lo, x1, y0 + hi) if vertical else (x0 + lo, y0, x0 + hi, y1)


def cut_page_gaps(crop, diff, core, vertical, bg, bg_thr=12.0, min_gap=6, thr=20.0):
    """A run of page-coloured rows inside the core that is NOT flanked by the bar's outline on both sides (no frame
    beside those rows) separates something else from the bar - a label row of a neighbouring bar merged into the
    region. The core is cut back to the part on the gradient's side of the first such gap from each end. Page
    rows inside a frame (the white end of a bar) and page rows at the very end of the core are kept."""
    x0, y0, x1, y1 = core
    rows = _rows(crop, core, vertical)
    n = len(rows)
    if n < 12:
        return core
    page = np.abs(np.median(rows, axis=1) - np.asarray(bg, float)).max(axis=1) <= bg_thr
    H, W = diff.shape
    if vertical:
        left = list(range(max(0, x0 - 3), min(W, x0 + 3)))
        right = list(range(max(0, x1 - 3), min(W, x1 + 3)))
        score = np.minimum(diff[y0:y1, left].max(axis=1), diff[y0:y1, right].max(axis=1)) if left and right else np.zeros(n)
    else:
        top = list(range(max(0, y0 - 3), min(H, y0 + 3)))
        bottom = list(range(max(0, y1 - 3), min(H, y1 + 3)))
        score = np.minimum(diff[top, x0:x1].max(axis=0), diff[bottom, x0:x1].max(axis=0)) if top and bottom else np.zeros(n)
    framed = score >= thr
    gaps = []
    start = None
    for i, p in enumerate(list(page) + [False]):
        if p and start is None:
            start = i
        elif not p and start is not None:
            if i - start >= min_gap and framed[start:i].mean() < 0.8:
                gaps.append((start, i))
            start = None
    if not gaps:
        return core
    nonpage = [(s, e) for s, e in _runs(~page) if e - s > 0]
    if not nonpage:
        return core
    main = max(nonpage, key=lambda r: r[1] - r[0])
    lo, hi = 0, n
    for s, e in gaps:
        beyond = [r for r in nonpage if r[1] <= s] if e <= main[0] else [r for r in nonpage if r[0] >= e]
        if not beyond:                                           # a page run at the very end: left to the trimming
            continue
        if e <= main[0]:
            lo = max(lo, e)
        elif s >= main[1]:
            hi = min(hi, s)
    if (lo, hi) == (0, n):
        return core
    return (x0, y0 + lo, x1, y0 + hi) if vertical else (x0 + lo, y0, x0 + hi, y1)


def cut_flat_ends(crop, core, vertical, max_frac=0.45, min_len=6, flat=8.0, jump=60.0):
    """A flat block of one colour at an end of the core, separated from the rest by a colour jump, is not part of
    the gradient (a dark region above a bar merged into the region) and is cut off. A gradient that fades into a
    flat end colour has no jump and is kept."""
    x0, y0, x1, y1 = core
    rows = _rows(crop, core, vertical)
    n = len(rows)
    if n < 2 * min_len:
        return core
    med = np.median(rows, axis=1)
    lo, hi = 0, n
    limit = int(max_frac * n)
    for i in range(1, limit):                                   # low end
        if np.abs(med[i] - med[i - 1]).max() > jump and i >= min_len and np.ptp(med[:i], axis=0).max() < flat:
            lo = i
            break
    for i in range(n - 1, n - limit, -1):                       # high end
        if np.abs(med[i] - med[i - 1]).max() > jump and n - i >= min_len and np.ptp(med[i:], axis=0).max() < flat:
            hi = i
            break
    if (lo, hi) == (0, n):
        return core
    return (x0, y0 + lo, x1, y0 + hi) if vertical else (x0 + lo, y0, x0 + hi, y1)


def _gradient_fraction(crop, box, vertical, bg, max_spread=45.0, bg_thr=12.0):
    """Fraction of the rows of a region that look like a gradient strip: uniform across the bar and not the page
    colour. A column of tick labels (mostly page with a few text rows) scores low."""
    rows = _rows(crop, box, vertical)
    if len(rows) < 4 or rows.shape[1] < 1:
        return 0.0
    uniform = _uniform_rows(rows, max_spread=max_spread, ref=rows)
    page = np.abs(np.median(rows, axis=1) - np.asarray(bg, float)).max(axis=1) <= bg_thr
    return float((uniform & ~page).mean())


def verified_strip_bbox(crop, roi, map_boxes=None):
    """Core gradient region (text/background rows trimmed from its ends) + verified extension. Among the candidate
    regions of the gradient finder, those lying inside a map panel and continuing its colours and those whose rows
    are mostly not gradient-like (a label column) are dropped; the standard configuration's candidate is the core
    unless another is twice as long (orientation agreeing with an elongated detector box).
    Returns (bbox, orientation, core_bbox) or None."""
    bg, diff = _background_and_diff(crop)
    keep = []
    for box, orientation in find_gradient_region(crop, roi=roi, extend=False, candidates=True):
        if map_boxes:
            inside = _inside_map(box, roi, map_boxes)
            if inside is not None and _continues_into(crop, box, inside, orientation == "vertical"):
                continue
        if _gradient_fraction(crop, box, orientation == "vertical", bg) < 0.25:    # a label column is not a strip
            continue
        keep.append((box, orientation))
    found = choose_candidate(keep, roi, longer=2.0)
    if found is None:
        return None
    core, orientation = found
    vertical = orientation == "vertical"
    core = cut_page_gaps(crop, diff, core, vertical, bg)
    core = trim_core_ends(crop, core, vertical, bg)
    ext = _extend_along_axis(diff, core, orientation, thr=20)
    a0c, a1c = (core[1], core[3]) if vertical else (core[0], core[2])
    a0e, a1e = (ext[1], ext[3]) if vertical else (ext[0], ext[2])
    lo, hi = a0c, a1c
    ref = _rows(crop, core, vertical)
    def not_page(seg):                                   # the extension must be bar, not page
        return (np.abs(np.median(seg, axis=1) - np.asarray(bg, float)).max(axis=1) > 12.0).mean() >= 0.5
    if a0e < a0c:
        seg = _rows(crop, (core[0], a0e, core[2], a0c + 1) if vertical else (a0e, core[1], a0c + 1, core[3]), vertical)
        if len(seg) > 1 and _uniform_rows(seg, ref=ref).all() and _continuous(seg).all() and not_page(seg):
            lo = a0e
    if a1e > a1c:
        seg = _rows(crop, (core[0], a1c - 1, core[2], a1e) if vertical else (a1c - 1, core[1], a1e, core[3]), vertical)
        if len(seg) > 1 and _uniform_rows(seg, ref=ref).all() and _continuous(seg).all() and not_page(seg):
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
        if lstar < 65:
            continue
        end_row = end_cols[end]
        end_lum = float(np.mean(end_row))
        spread_thr = max(60.0, 2.0 * typ + 10.0)
        # a frame line within a short window outside the core marks the real end: the rows between the core end and
        # the frame belong to the bar (a light end trimmed as page background) when they are uniform across the bar
        # and continuous row to row; the frame may lie at most a few pixels beyond the detector box
        margin = max(4.0, 0.03 * L)
        win = int(max(4, round(0.12 * L), ((a0 - d0) if end == 0 else (d1 - a1)) + margin))
        fb = (x0, max(0, a0 - win), x1, a0) if (end == 0 and vertical) else (x0, a1, x1, min(H, a1 + win)) if vertical else \
             (max(0, a0 - win), y0, a0, y1) if end == 0 else (a1, y0, min(W, a1 + win), y1)
        fr = _rows(crop, fb, vertical)
        if len(fr):
            fr_med = np.median(fr, axis=1)
            fr_lum = np.mean(fr_med, axis=1)
            fr_spread = (np.percentile(fr, 95, axis=1) - np.percentile(fr, 5, axis=1)).max(axis=1)
            dip = np.zeros(len(fr_lum), bool)                   # a thin frame line: darker than a neighbour row
            if len(fr_lum) >= 2:
                dip[1:] |= (fr_lum[:-1] - fr_lum[1:]) > 35.0
                dip[:-1] |= (fr_lum[1:] - fr_lum[:-1]) > 35.0
            frame = (fr_lum < 180) & ((end_lum - fr_lum > 60) | (dip & (end_lum - fr_lum > 30))) & (fr_spread <= spread_thr)
            if frame.any():
                idx = list(range(len(fr) - 1, -1, -1)) if end == 0 else list(range(len(fr)))   # outward from the core end
                n_ok, prev = 0, end_row
                for i in idx:
                    if frame[i]:
                        break
                    if np.abs(fr_med[i] - prev).max() > 35.0:
                        n_ok = -1
                        break
                    prev = fr_med[i]
                    n_ok += 1
                if n_ok > 0:
                    pos = (a0 - n_ok) if end == 0 else (a1 + n_ok)
                    beyond = (d0 - pos) if end == 0 else (pos - d1)          # extension beyond the detector box end
                    if beyond <= margin:
                        if end == 0:
                            new0 = pos
                        else:
                            new1 = pos
                        continue
                if n_ok == 0:
                    continue                                                 # the frame sits right at the core end
        gap = (a0 - d0) if end == 0 else (d1 - a1)
        if gap <= min_gap * L:
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
            if spread[i] > spread_thr or np.abs(med[i] - end_row).max() > max_step:
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
    if it.get("segmented"):            # box of the segmented pass is in sub-crop coordinates; cx/cy are in crop coordinates
        w, h = float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))
        return it["cx"] - w / 2, it["cy"] - h / 2, it["cx"] + w / 2, it["cy"] + h / 2
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _ac(box, vertical):
    x0, y0, x1, y1 = box
    return (y0, y1, x0, x1) if vertical else (x0, x1, y0, y1)


def _edge_dist(pt, box):
    x, y = pt; x0, y0, x1, y1 = box
    return float(np.hypot(max(0.0, x0 - x, x - x1), max(0.0, y0 - y, y - y1)))


_RE_SIGNED_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def split_merged_numbers(text):
    """A merged row of tick labels ('0.000.160.32', '-1.5-0.9-0.3 0.3', '0 25 50 75') -> list of values, or None.
    The run must consist of numbers only (signs, spaces), give at least two values and be monotonic."""
    vals = split_merged_decimals(text)
    if vals:
        return vals
    t = _normalize(text).strip()
    if not t or not re.fullmatch(r"[-+\d.\s]+", t) or t.count(".") + t.count(" ") + t.count("-") < 1:
        return None
    parts = _RE_SIGNED_NUM.findall(t)
    if len(parts) < 2 or "".join(parts) != t.replace(" ", ""):
        return None
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return None
    d = np.diff(vals)
    return vals if (np.all(d > 0) or np.all(d < 0)) else None


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


def _with_ink(crop, items, vertical):
    """Add ``ink_ca`` (centre of the glyphs along the bar axis, crop coordinates) to every OCR item whose box
    contains ink; the box itself is kept."""
    H, W = crop.shape[:2]
    out = []
    for it in items:
        it = dict(it)
        x0, y0, x1, y1 = _item_bbox(it)
        x0, y0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
        x1, y1 = int(min(W, np.ceil(x1))), int(min(H, np.ceil(y1)))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            lum = crop[y0:y1, x0:x1].astype(float).mean(axis=2)
            bg = float(np.percentile(lum, 90))
            ink = np.clip(bg - lum, 0, None)
            if ink.max() > 0:
                dark = ink > 0.3 * ink.max()                                 # ink pixels
                cnt = dark.sum(axis=1) if vertical else dark.sum(axis=0)     # per row / column along the bar axis
                on = np.where(cnt >= 1)[0]
                if len(on) >= 2:                                             # keep runs of >= 2 lines: a 1-px minus
                    runs = np.split(on, np.where(np.diff(on) > 1)[0] + 1)    # sign counts, isolated speckles do not
                    keep = [r for r in runs if len(r) >= 2]
                    if keep:
                        it["ink_ca"] = (y0 if vertical else x0) + 0.5 * (keep[0][0] + keep[-1][-1]) + 0.5
        out.append(it)
    return out


def classify(nums, strip_bbox, vertical):
    """Each number relative to the strip: kind in {end, inside, col, far}, pos along the strip (0..1), side across
    the axis (-1 / 0 / +1), near-edge offset from the strip, axis centre, and whether its box overlaps the strip."""
    sa0, sa1, sc0, sc1 = _ac(strip_bbox, vertical)
    L, w = max(1.0, sa1 - sa0), max(1.0, sc1 - sc0)
    out = []
    for it in nums:
        a0, a1, c0, c1 = _ac(it["bbox"], vertical)
        ca, cc = (a0 + a1) / 2, (c0 + c1) / 2
        if it.get("ink_ca") is not None and a0 - 1 <= it["ink_ca"] <= a1 + 1:
            ca = float(it["ink_ca"])                       # glyph centre, not text-box centre
        along = (ca - sa0) / L
        centred = abs(cc - (sc0 + sc1) / 2) <= max(0.5 * w, 4.0)
        beyond_lo, beyond_hi = a1 <= sa0 + 2, a0 >= sa1 - 2
        rec = {**it, "along": along, "ca": ca, "overlaps": not (beyond_lo or beyond_hi), "ha": (a1 - a0)}
        if centred:
            reach = max(2.5 * (a1 - a0), 0.12 * L)
            if (beyond_lo or along < 0.0) and along >= -0.8 and sa0 - a1 <= reach:
                rec.update(kind="end", pos=0.0)
            elif (beyond_hi or along > 1.0) and along <= 1.8 and a0 - sa1 <= reach:
                rec.update(kind="end", pos=1.0)
            elif 0.0 <= along <= 1.0:
                rec.update(kind="inside", pos=along)
            elif -1.0 <= along <= 2.0:
                rec.update(kind="far", pos=along)     # centred but far beyond an end (a label of something else)
            else:
                continue
            rec.update(side=0, off=-1.0)
            out.append(rec)
            continue
        side = 1 if cc > (sc0 + sc1) / 2 else -1
        off = (c0 - sc1) if side > 0 else (sc0 - c1)
        rec["off_far"] = (c1 - sc1) if side > 0 else (sc0 - c0)
        reach = max(2.5 * (a1 - a0), 0.12 * L)          # an end label sits just beyond the end, not far away
        if beyond_lo and along >= -0.5 and sa0 - a1 <= reach:
            rec.update(kind="end", pos=0.0)
        elif beyond_hi and along <= 1.5 and a0 - sa1 <= reach:
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
    """Labels of one column: aligned at their near edge (left-aligned labels) or at their far edge (right-aligned
    labels, whose near edges shift with the number of digits and with the minus sign)."""
    cl = []
    for c in sorted(side_items, key=lambda d: d["off"]):
        if cl and (c["off"] - cl[-1][0]["off"] <= tol or
                   any(abs(c.get("off_far", c["off"]) - m.get("off_far", m["off"])) <= tol for m in cl[-1])):
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
        # a side whose nearest cluster has a label along the strip beats a side whose cluster consists
        # of end labels only (two numbers of a neighbouring axis beyond the strip ends line up like a column)
        along = {k: sum(1 for c in best[k] if c["kind"] in ("col", "inside")) for k in best}
        s = min(best, key=lambda k: (0 if along[k] >= 1 else 1, np.median([c["off"] for c in best[k]]), -len(best[k])))
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
    # several labels beyond the same end: only the nearest one is that end's label
    for pos_end, edge in ((0.0, sa0), (1.0, sa1)):
        ends = [c for c in chosen if c["kind"] == "end" and c["pos"] == pos_end]
        if len(ends) > 1:
            ends.sort(key=lambda c: abs(c["ca"] - edge))
            for c in ends[1:]:
                if abs(c["ca"] - edge) > abs(ends[0]["ca"] - edge) + max(text_h, 6.0):
                    c["kind"] = "far"
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
            col = [c for c in chosen if c["kind"] in ("col", "inside") and c["side"] != 0]
            side_c = (1 if np.mean([c["side"] for c in col]) > 0 else -1) if col else None
            off_c = float(np.median([c["off"] for c in col])) if col else None
            for c in far:
                p = (c["ca"] - e0) / Li
                pred = a + b * p
                if side_c is not None and (c["side"] != side_c or c["off"] < 0 or abs(c["off"] - off_c) > max(0.5 * th, 3.0)):
                    c["kind"] = "far"              # not part of the label column: it cannot extend the strip
                    continue
                if abs(pred - c["num"]) <= 0.05 * (abs(b) + 1e-12):
                    new0, new1 = min(new0, c["ca"]), max(new1, c["ca"])
                    c["kind"] = "col"
                elif c["kind"] == "end" and (c["num"] - pred) * (-1 if p < 0 else 1) * np.sign(b) > 0 \
                        and abs(pred - c["num"]) <= 0.25 * (abs(b) + 1e-12):
                    # the sequence would reach this value a little beyond the label itself: the bar ends at its end
                    # label (a value far off the sequence belongs to something else, e.g. a neighbouring axis)
                    new0, new1 = min(new0, c["ca"]), max(new1, c["ca"])
                elif c["kind"] == "end":
                    c["kind"] = "far"
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


_RE_POW10 = re.compile(r"^10\d$")


def _fit(anchors, scale):
    texts = [str(a[6]).strip() for a in anchors if len(a) > 6]
    pow10 = [t for t in texts if _RE_POW10.match(t)]
    if len(pow10) >= 3 and len({t[-1] for t in pow10}) >= 3:
        return {"ok": False, "reason": "logarithmic colour scale (powers of ten)", "n_anchors": len(anchors), "scale": "log"}
    fit = fit_scale([a[:4] for a in anchors], scale)
    if fit.get("ok") and abs(fit["value_at_end"] - fit["value_at_start"]) <= 1e-6 * max(abs(fit["value_at_end"]), abs(fit["value_at_start"])):
        return {**fit, "ok": False, "reason": "degenerate scale (equal end values)"}
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
        if len(ends) == 2 and ends[0][0] != ends[1][0]:
            fit2 = fit_scale([a[:4] for a in ends], scale)
            if fit2.get("ok"):
                # the ends alone are trusted only when every interior label is at least roughly consistent with
                # them (a label whose position is a little off), not when its value contradicts them (a misread end)
                v0, v1 = fit2["value_at_start"], fit2["value_at_end"]
                span = abs(v1 - v0) + 1e-12
                if all(abs(v0 + (v1 - v0) * a[0] - a[1] * scale) <= 0.15 * span for a in interior):
                    fit2["reason"] = None
                    fit2["end_labels_only"] = True
                    return fit2
    return fit


def extend_by_walk(crop, bbox, vertical, det_ac, bg, max_step=35.0, max_run=2, margin_frac=0.03, limits=None,
                   light_only=True, min_lstar=65.0, margin_min=4.0, page_end=True):
    """Grow the strip outward at both ends, row by row, while the rows continue the gradient: uniform across the
    bar (rows broken by tick marks crossing a thin strip are tolerated in runs of at most ``max_run`` rows when
    the gradient continues behind them), continuous row to row and not the page colour (a strip whose end already
    has the page colour - a white end on a white page - may continue to the detector box end, as nothing else
    marks the bar's end there). With ``light_only`` only ends whose colour is light (L* >= ``min_lstar``) are grown: a
    bright end fades into the page and is cut short by the foreground threshold, a dark end is not, and dark
    neighbours (a frame, a map edge) are uniform and continuous without being part of the bar. Both ends are
    bounded by the detector box plus a small margin (or by ``limits`` = (lo, hi) when given); with ``page_end``
    false an end whose colour is already the page colour is not grown at all.
    Returns (bbox, extension at the low end in px, extension at the high end in px)."""
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    a0, a1 = (y0, y1) if vertical else (x0, x1)
    H, W = crop.shape[:2]
    lim = H if vertical else W
    d0, d1 = det_ac
    margin = max(margin_min, margin_frac * max(1, d1 - d0))
    lo_lim, hi_lim = int(max(0, np.floor(d0 - margin))), int(min(lim, np.ceil(d1 + margin)))
    if limits is not None:
        lo_lim, hi_lim = int(max(0, limits[0])), int(min(lim, limits[1]))
    if a1 - a0 < 4 or a0 < 0 or a1 > lim:
        return bbox, 0, 0
    rows = _rows(crop, (x0, 0, x1, lim) if vertical else (0, y0, lim, y1), vertical)
    if len(rows) != lim or rows.shape[1] < 1:
        return bbox, 0, 0
    core = rows[a0:a1]
    typ = float(np.median((np.percentile(core, 95, axis=1) - np.percentile(core, 5, axis=1)).max(axis=1)))
    spread_thr = max(45.0, 2.0 * typ + 10.0)
    med = np.median(rows, axis=1)
    spread = (np.percentile(rows, 95, axis=1) - np.percentile(rows, 5, axis=1)).max(axis=1)
    uniform = spread <= spread_thr
    bgdist = np.abs(med - np.asarray(bg, float)).max(axis=1)
    new0, new1 = a0, a1
    k = min(4, max(1, len(core) // 2))
    end_lab = _lab(np.stack([np.median(med[a0:a0 + k], axis=0), np.median(med[a1 - k:a1], axis=0)]))
    for end in (0, 1):
        if light_only and end_lab[end, 0] < min_lstar:
            continue
        if end == 0:
            idx, last = range(a0 - 1, lo_lim - 1, -1), a0
        else:
            idx, last = range(a1, hi_lim), a1 - 1
        ref_col = np.median(med[a0:a0 + k], axis=0) if end == 0 else np.median(med[a1 - k:a1], axis=0)   # interior colour
        end_is_page = float(np.abs(ref_col - np.asarray(bg, float)).max()) <= 25.0
        if end_is_page and not page_end:                   # an end already on the page: nothing to follow
            continue
        skipped, page_run, last_nonpage = 0, 0, last
        for i in idx:
            if not uniform[i]:
                skipped += 1
                if skipped > max_run:
                    break
                continue
            if np.abs(med[i] - ref_col).max() > max_step * (1 + skipped):
                break
            ref_col = med[i]
            if bgdist[i] <= 10.0 and not end_is_page:
                page_run += 1
                if page_run >= 3:
                    break
            else:
                page_run = 0
                last_nonpage = i
            skipped, last = 0, i
        stop = last if end_is_page else last_nonpage
        if end == 0:
            new0 = min(a0, stop)
        else:
            new1 = max(a1, stop + 1)
    out = (x0, new0, x1, new1) if vertical else (new0, y0, new1, y1)
    return out, a0 - new0, new1 - a1


def _restore_beyond(crop, clipped, walked, vertical, other_bars, map_boxes, flat_levels=4.0):
    """Of the strip part beyond the detector box that the row walk found continuous, keep each end only when it
    lies in no other bar's box and no map box (a stacked bar or a touching map continues the colours) and is not
    a flat block of one colour (a uniform region is not a gradient)."""
    out = list(clipped)
    lo_i, hi_i = (1, 3) if vertical else (0, 2)
    for end in (0, 1):
        c_end, w_end = (clipped[lo_i], walked[lo_i]) if end == 0 else (clipped[hi_i], walked[hi_i])
        seg = (w_end, c_end) if end == 0 else (c_end, w_end)
        if seg[1] - seg[0] <= 0:
            continue
        box = (clipped[0], seg[0], clipped[2], seg[1]) if vertical else (seg[0], clipped[1], seg[1], clipped[3])
        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        if any(_edge_dist(centre, b) == 0 for b in list(other_bars or []) + list(map_boxes or [])):
            continue
        rows = _rows(crop, box, vertical)
        if len(rows) > 4:
            lum = np.median(rows, axis=1).mean(axis=1)
            if float(lum.max() - lum.min()) < flat_levels:
                continue
        out[lo_i if end == 0 else hi_i] = w_end
    return tuple(int(v) for v in out)


# ----------------------------------------------------------------------------------------------- signs from ink
def _ink_mask(band):
    """Ink pixels of a small text band: far from the band's median colour (dark text on light page or the reverse)."""
    lum = band.astype(float).mean(axis=2)
    ink = np.abs(lum - float(np.median(lum)))
    if ink.max() < 40.0:
        return np.zeros(lum.shape, bool)
    return ink >= max(40.0, 0.4 * float(ink.max()))


def _runs(flags):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def _strokes_before_digits(crop, box, x_from, h):
    """Thin horizontal strokes (a minus sign or a tick mark) left of the digits of an upright label. The rows of
    the OCR box are scanned from ``x_from`` to the box's right edge; the digits are the first run of columns whose
    ink is at least half the text height tall. Returns (strokes, digit_x0) with strokes = [(x0, x1, attached,
    gap_to_digits)] - attached: the stroke starts within 2 px of ``x_from`` or of the end of a frame line of the
    strip found within the first columns of the band (which is ignored) - or (None, None) when no digits are found."""
    H, W = crop.shape[:2]
    bx0, by0, bx1, by1 = box
    y0, y1 = int(max(0, np.floor(by0) - 1)), int(min(H, np.ceil(by1) + 1))
    xf, xt = int(max(0, np.floor(x_from))), int(min(W, np.ceil(bx1) + 1))
    if y1 - y0 < 3 or xt - xf < 3:
        return None, None
    mask = _ink_mask(crop[y0:y1, xf:xt])
    frame_end = 0                                        # a frame line of the strip at the band's start: a tick
    for c in range(min(3, mask.shape[1])):               # mark attached to the frame starts right after it
        if mask[:, c].mean() >= 0.9:
            mask[:, c] = False
            frame_end = c + 1
    info = []
    for c0, c1 in _runs(mask.sum(axis=0) >= 1):
        rows = np.where(mask[:, c0:c1].any(axis=1))[0]
        info.append((c0, c1, int(rows.min()), int(rows.max()) + 1))
    digit = None
    for k, (c0, c1, r0, r1) in enumerate(info):
        if (r1 - r0) >= 0.5 * h and xf + c1 > bx0 + 1:
            digit = k
            break
    if digit is None:
        return None, None
    dc0, dc1, dr0, dr1 = info[digit]
    hd = dr1 - dr0
    strokes = []
    for c0, c1, r0, r1 in info[:digit]:
        wid, hei, mid = c1 - c0, r1 - r0, (r0 + r1) / 2.0
        if hei <= 0.35 * h and max(2.0, 0.2 * h) <= wid <= 1.5 * h and dr0 + 0.25 * hd <= mid <= dr1 - 0.25 * hd:
            strokes.append((xf + c0, xf + c1, c0 <= frame_end + 2, dc0 - c1))
    return strokes, xf + dc0


def _rotated_minus(crop, box):
    """A label written along a vertical bar (tall OCR box): True when a thin vertical stroke, detached from the
    glyphs, stands just beyond either end of the text in the middle of its columns (the minus sign of rotated
    text)."""
    H, W = crop.shape[:2]
    bx0, by0, bx1, by1 = box
    gw = max(4.0, bx1 - bx0)                              # glyph height of the rotated text
    xa, xb = int(max(0, np.floor(bx0) - 2)), int(min(W, np.ceil(bx1) + 2))
    if xb - xa < 3:
        return False
    for end in (0, 1):
        ya, yb = (int(max(0, by0 - 1.5 * gw)), int(min(H, by0 + 0.3 * gw))) if end == 0 else \
                 (int(max(0, by1 - 0.3 * gw)), int(min(H, by1 + 1.5 * gw)))
        if yb - ya < 3:
            continue
        mask = _ink_mask(crop[ya:yb, xa:xb])
        runs = _runs(mask.sum(axis=1) >= 1)              # runs of rows
        if not runs:
            continue
        glyph = None
        for k, (r0, r1) in enumerate(runs):
            if ya + r1 > by0 + 1 and ya + r0 < by1 - 1:
                glyph = k
                break
        if glyph is None:
            continue
        cand = runs[:glyph] if end == 0 else runs[glyph + 1:]
        if not cand:
            continue
        r0, r1 = cand[-1] if end == 0 else cand[0]
        cols = np.where(mask[r0:r1].any(axis=0))[0]
        wid, hei = int(cols.max()) - int(cols.min()) + 1, r1 - r0
        gap = (runs[glyph][0] - r1) if end == 0 else (r0 - runs[glyph][1])
        mid = (cols.min() + cols.max()) / 2.0 + xa
        if wid <= 0.35 * gw and 0.25 * gw <= hei <= 1.5 * gw and gap <= 0.7 * gw and bx0 + 0.25 * gw <= mid <= bx1 - 0.25 * gw:
            return True
    return False


def _set_sign(c, sign):
    if abs(c["num"]) < 1e-12:
        return
    cur = -1 if c["num"] < 0 else 1
    if cur == sign:
        return
    c["num"] = sign * abs(c["num"])
    t = str(c.get("text", "")).lstrip(" -\u2212\u2013+")
    c["text"] = ("-" + t) if sign < 0 else t
    c["sign_ink"] = sign


def _apply_ink_signs(crop, chosen, bbox, vertical, text_h):
    """Decide the sign of the chosen labels from the ink next to their digits. A thin stroke at mid-height of
    the digits is a minus sign when it is detached from the strip, a tick mark when it is attached to the strip
    edge. For labels right of a vertical strip, where a tick mark precedes the digits exactly like a minus sign,
    the labels of the column are judged together: a detached stroke is a minus; among attached strokes the
    strokes are minus signs when some label (the zero label included) has no stroke at all and the OCR agrees on
    at least half of them (ticks would be drawn at every label); otherwise the narrow attached strokes are bare
    ticks when a wider (tick plus minus) or a detached stroke exists in the column or the zero label carries one,
    and equal attached strokes on every label are ticks (the OCR's minus signs are then dropped only on labels
    whose box starts at the strip edge). Elsewhere (labels left of the strip, below or
    above a horizontal strip, end labels) a detached stroke next to the digits adds the minus sign. Labels written
    along the bar get the minus from a vertical stroke beyond their ends. Modifies ``num``/``text`` in place."""
    sa0, sa1, sc0, sc1 = _ac(bbox, vertical)
    h = max(float(text_h), 6.0)
    groups = {}
    for c in chosen:
        groups.setdefault(int(c["side"]), []).append(c)
    for side, labs in groups.items():
        S, D, N, zero_tick, had_minus = [], [], [], False, {}
        for c in labs:
            had_minus[id(c)] = c["num"] < 0
            bx0, by0, bx1, by1 = c["bbox"]
            txt = re.sub(r"\s", "", str(c.get("text", "")))
            if vertical and (by1 - by0) > 1.8 * max(1.0, bx1 - bx0) and len(txt) >= 2:
                if _rotated_minus(crop, c["bbox"]):
                    _set_sign(c, -1)
                continue
            column = vertical and side > 0
            x_from = (sc1 + 1) if column else (bx0 - 1.5 * h)
            if side == 0 and bx0 >= sc0 - 1 and bx1 <= sc1 + 1:        # a label written inside the bar
                x_from = max(x_from, sc0 + 1)
            strokes, dx0 = _strokes_before_digits(crop, c["bbox"], x_from, h)
            if strokes is None:
                continue
            if not column:
                if "+" not in str(c.get("text", "")) and \
                        any(not s[2] and s[3] <= 0.5 * h and (s[1] - s[0]) >= 0.35 * h for s in strokes):
                    _set_sign(c, -1)
                continue
            att = [s for s in strokes if s[2]]
            det = [s for s in strokes if not s[2] and s[3] <= 0.7 * h]
            is_zero = abs(c["num"]) < 1e-12
            if det and not is_zero:
                D.append(c)
            if att:
                if is_zero:
                    zero_tick = True
                else:
                    S.append((c, max(s[1] - s[0] for s in att)))
            if not att and not det:
                N.append(c)                                   # no stroke at all (a zero label counts: no tick there)
        if not (vertical and side > 0):
            continue
        for c in D:
            _set_sign(c, -1)
        if not S:
            continue
        wmin = min(aw for _, aw in S)
        wide = [c for c, aw in S if aw >= wmin + max(2.0, 0.25 * h)]
        narrow = [c for c, aw in S if aw < wmin + max(2.0, 0.25 * h)]
        consensus = sum(1 for c, _ in S if had_minus[id(c)]) * 2 >= len(S)

        def positive(c):                                   # a bare tick: drop an OCR minus only when the OCR box
            if not had_minus[id(c)] or c["bbox"][0] <= sc1 + 5:   # included the tick (starts at the strip edge)
                _set_sign(c, 1)

        if not zero_tick and N and consensus:
            # a label without any stroke (ticks are drawn at every label): the strokes are minus signs
            for c, _ in S:
                _set_sign(c, -1)
        elif wide or D or zero_tick:
            for c in wide:
                _set_sign(c, -1)
            for c in narrow:
                if c not in D:
                    positive(c)
        elif any(not had_minus[id(c)] for c, _ in S):
            for c, _ in S:
                if c not in D:
                    positive(c)


# ----------------------------------------------------------------------------------------------- main reader
_RE_KMULT = re.compile(r"^[-+]?\d+(?:\.\d+)?k$")


def _k_multiplier(chosen):
    """True when every chosen label is written with a trailing 'k' (3k, 9k, 15k, 21k): a thousands multiplier."""
    texts = [re.sub(r"\s", "", str(c.get("text", ""))) for c in chosen if c.get("num") is not None]
    return len(texts) >= 2 and all(_RE_KMULT.match(t) for t in texts)


def _pow10_labels(chosen):
    """True when the column labels are powers of ten written as superscripts and read as '100', '101', ... (two or
    more distinct exponents and nothing else in the column): a logarithmic scale."""
    texts = [re.sub(r"\s", "", str(c.get("text", ""))) for c in chosen
             if c.get("num") is not None and c.get("kind") in ("col", "inside")]
    if len(texts) < 2 or not all(_RE_POW10.match(t) for t in texts):
        return False
    return len({t[-1] for t in texts}) >= 2


def _attempt(items, bbox, ext, vertical, det_box, other_bars, map_boxes, core=None, crop=None):
    if core is not None and (ext[0] > 0 or ext[1] > 0):
        bbox = cut_at_column_text(items, bbox, core, vertical)
    sa0, sa1, sc0, sc1 = _ac(bbox, vertical)
    w = sc1 - sc0
    nums = own_items(numeric_items(items), det_box, other_bars, map_boxes)
    cands = classify(nums, bbox, vertical)
    text_h = float(np.median([c["h"] for c in cands])) if cands else 8.0
    tol = max(0.8 * text_h, 0.6 * w, 4.0)
    max_off = max(2.5 * text_h, 1.0 * w, 12.0)          # tick labels sit within a few text heights of the strip
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
    if crop is not None and chosen:
        _apply_ink_signs(crop, chosen, bbox, vertical, text_h)
    if chosen and _k_multiplier(chosen):                    # '3k' ... '21k': a thousands multiplier on every label
        for c in chosen:
            c["num"] = float(c["num"]) * 1e3
    log_labels = _pow10_labels(chosen)
    # a nearer column with a single number was skipped in favour of a farther one: the near labels may be unread
    near_skipped = None
    side_chosen = [c for c in chosen if c["side"] != 0]
    if side_chosen:
        off_chosen = float(np.median([c["off"] for c in side_chosen]))
        nearer = [c for c in cands if c["side"] != 0 and c["kind"] == "col" and c["off"] < 0.5 * off_chosen and c["off"] <= max_off]
        if nearer:
            near_skipped = 1 if np.mean([c["side"] for c in nearer]) > 0 else -1
    new0, new1, anchors = place_ends([dict(c) for c in chosen], sa0, sa1, ext[0], ext[1], text_h)
    if crop is not None and (new0 < sa0 - 1 or new1 > sa1 + 1):
        # labels extended the strip beyond the gradient found: the bar may continue beyond the outermost label
        dac = (det_box[1], det_box[3]) if vertical else (det_box[0], det_box[2])
        ext_box = (bbox[0], int(round(new0)), bbox[2], int(round(new1))) if vertical else \
                  (int(round(new0)), bbox[1], int(round(new1)), bbox[3])
        wb, w_lo, w_hi = extend_by_walk(crop, ext_box, vertical, dac, _background_and_diff(crop)[0],
                                        margin_frac=0.0, margin_min=0.0, page_end=False)
        if w_lo or w_hi:
            e0, e1 = (wb[1], wb[3]) if vertical else (wb[0], wb[2])
            if new0 < sa0 - 1 and w_lo:
                new0 = e0
            if new1 > sa1 + 1 and w_hi:
                new1 = e1
            Ln = max(1.0, new1 - new0)
            anchors = [((float(np.clip((a[3] - new0) / Ln, 0.0, 1.0)) if new0 - 1 <= a[3] <= new1 + 1 else a[0]),) + tuple(a[1:])
                       for a in anchors]
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
    if log_labels:
        fit = {"ok": False, "reason": "logarithmic colour scale (powers of ten)", "n_anchors": len(anchors), "scale": "log"}
    new_bbox = (bbox[0], int(round(new0)), bbox[2], int(round(new1))) if vertical else (int(round(new0)), bbox[1], int(round(new1)), bbox[3])
    return dict(fit=fit, anchors=anchors, chosen=chosen, how=how, unit=unit, scale=scale, bbox=new_bbox, text_h=text_h,
                cands=cands, items=items, near_skipped=near_skipped, dropped=list(getattr(own_items, "last_dropped", [])),
                bbox_in=bbox)


def _score(att):
    f = att["fit"]
    return (1 if f.get("ok") else 0, 0 if att.get("near_skipped") else 1, 0 if (f.get("end_labels_only") or f.get("sign_repaired")) else 1,
            int(f.get("n_inliers", 0) or 0),
            sum(1 for a in att["anchors"] if len(a) > 6 and str(a[6]).lstrip().startswith("-")),   # minus signs are lost, not invented
            len(att["anchors"]))


def _overlap_frac(a, b):
    """Fraction of box a (x0, y0, x1, y1) that lies inside box b."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return iw * ih / area


def _strip_in_box(strip, box, vertical, min_along=0.5):
    """The detector box covers at least ``min_along`` of the strip's length along the bar and overlaps it across."""
    sa0, sa1, sc0, sc1 = _ac(strip, vertical)
    ba0, ba1, bc0, bc1 = _ac(box, vertical)
    along = max(0.0, min(sa1, ba1) - max(sa0, ba0)) / max(1.0, sa1 - sa0)
    across = min(sc1, bc1) - max(sc0, bc0)
    return along >= min_along and across > 0


def _continues_into(crop, strip, map_box, vertical, max_de=12.0):
    """True when the colours of the strip continue into the adjacent pixels on at least one side across the bar
    (the strip is a slice of the map's edge); a real bar is separated from its map by background or a frame."""
    H, W = crop.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in strip]
    w = max(1, (x1 - x0) if vertical else (y1 - y0))
    rows = _rows(crop, strip, vertical)
    if len(rows) < 4:
        return False
    med = np.median(rows, axis=1)
    for side in (-1, 1):
        if vertical:
            bx0, bx1 = (x0 - 1 - w, x0 - 1) if side < 0 else (x1 + 1, x1 + 1 + w)
            band = (max(0, bx0), y0, min(W, bx1), y1)
        else:
            by0, by1 = (y0 - 1 - w, y0 - 1) if side < 0 else (y1 + 1, y1 + 1 + w)
            band = (x0, max(0, by0), x1, min(H, by1))
        if band[2] - band[0] < 1 or band[3] - band[1] < 1:
            continue
        if _overlap_frac(band, map_box) < 0.9:
            continue                                    # the band must lie inside the map
        brows = _rows(crop, band, vertical)
        if len(brows) != len(rows):
            continue
        bmed = np.median(brows, axis=1)
        de = np.linalg.norm(_lab(med) - _lab(bmed), axis=1)
        if float(np.median(de)) <= max_de:
            return True
    return False


def _inside_map(strip, det_box, map_boxes, strip_frac=0.9, det_frac=0.5):
    """The map box that contains the strip (>= strip_frac of its area) while containing less than det_frac of the
    detector box, or None. Such a strip is the edge of the map panel next to the detector box."""
    for m in map_boxes or []:
        if _overlap_frac(strip, m) >= strip_frac and _overlap_frac(det_box, m) < det_frac:
            return tuple(int(v) for v in m)
    return None


def _unit_near_labels(crop, ocr, items, best, bbox, vertical, det_box, other_bars, map_boxes, unit_mode):
    """The bar's unit label: a unit text beside the label column (outside it), in line with the labels beyond an
    end, or centred beyond an end, and nearer to this strip than to any map panel or other bar. When the upright
    read gives none, the band beside the labels is read rotated (both directions)."""
    H, W = crop.shape[:2]
    sa0, sa1, sc0, sc1 = _ac(bbox, vertical)
    L, w = max(1.0, sa1 - sa0), max(1.0, sc1 - sc0)
    chosen = [c for c in (best.get("chosen") or []) if c["side"] != 0]
    th = float(best.get("text_h") or 8.0)
    side = (1 if np.mean([c["side"] for c in chosen]) > 0 else -1) if chosen else None
    col = [c for c in chosen if c["side"] == side]
    if col:
        edges = [_ac(c["bbox"], vertical)[3 if side > 0 else 2] for c in col]
        outer = max(edges) if side > 0 else min(edges)
        inner = min(_ac(c["bbox"], vertical)[2] for c in col) if side > 0 else max(_ac(c["bbox"], vertical)[3] for c in col)
    else:
        outer = inner = (sc1 if side == 1 else sc0) if side else None

    def accept(u):
        if u is None:
            return False
        return canonical_unit(u) is not None if unit_mode == "lexicon" else _unit_score(u) >= 1

    cands = []
    chosen_all = list(best.get("chosen") or [])
    dx0, dy0, dx1, dy1 = det_box
    # a parenthesised unit slot of the bar's title at an end of the bar that is not a readable unit ('Length (uu)'):
    # the bar's unit is unknown; the band beside the labels is not consulted for it
    end_slot_unreadable = False
    for it in items:
        txt = str(it["text"]).strip()
        if "(" not in txt or ")" not in txt:
            continue
        u = parse_unit(txt)
        if u is None or accept(u):
            continue
        a0, a1, c0, c1 = _ac(_item_bbox(it), vertical)
        hb = max(4.0, a1 - a0)
        if abs((c0 + c1) / 2 - (sc0 + sc1) / 2) <= 0.5 * w + 4 and max(0.0, sa0 - a1, a0 - sa1) <= max(2.5 * hb, 0.12 * L):
            end_slot_unreadable = True
    ex, ey = 0.1 * (dx1 - dx0), 0.1 * (dy1 - dy0)
    for it in items:
        u = parse_unit(it["text"])
        if not accept(u):
            continue
        ix0, iy0, ix1, iy1 = _item_bbox(it)
        a0, a1, c0, c1 = _ac(_item_bbox(it), vertical)
        h = max(4.0, a1 - a0 if not vertical else c1 - c0)      # glyph height of an upright text
        hb = max(4.0, a1 - a0)
        if ((a1 - a0) if vertical else (c1 - c0)) > 2.5 * th:    # a rotated text read upright (an axis title)
            continue
        cc = (c0 + c1) / 2
        beyond = max(0.0, sa0 - a1, a0 - sa1)
        kind = None
        if abs(cc - (sc0 + sc1) / 2) <= 0.5 * w + 4 and beyond <= max(2.5 * hb, 0.12 * L):
            kind = "end"
        elif side is not None and outer is not None and (cc > (sc0 + sc1) / 2) == (side > 0):
            gap = (c0 - outer) if side > 0 else (outer - c1)
            in_col = c0 < (outer if side > 0 else inner) and c1 > (inner if side > 0 else outer)
            near_label = any(min(abs(a0 - _ac(c["bbox"], vertical)[1]), abs(_ac(c["bbox"], vertical)[0] - a1)) <= 1.5 * max(th, hb)
                             or (_ac(c["bbox"], vertical)[0] < a1 and _ac(c["bbox"], vertical)[1] > a0) for c in col)
            # beside the labels: a known unit token within the strip's extent (an axis title beyond the end of the
            # bar is not); in line with the labels: inside the strip's extent or next to one of the labels
            if a0 < sa1 and a1 > sa0 and -0.5 * h <= gap <= max(1.5 * h, 0.6 * w) and _unit_score(u) == 2:
                kind = "beside"
            elif in_col and (beyond <= 0.0 or near_label):
                kind = "inline"
        if kind is None:
            # a unit written next to one of the labels ('72.5' | 'nm'): the same text row, a small gap, after the
            # number and on the label's side of the strip (the unit of an axis below the map shares the row of the
            # bar's lowest label but lies across the strip from it)
            scx = (sc0 + sc1) / 2
            for c in chosen_all:
                lx0, ly0, lx1, ly1 = c["bbox"]
                row = min(iy1, ly1) - max(iy0, ly0) >= 0.5 * min(iy1 - iy0, ly1 - ly0)
                after = ix0 >= lx1 - 0.5 * th
                same_side = (not vertical) or c["side"] == 0 or ((it["cx"] > scx) == (c["side"] > 0))
                within = sa0 - 0.1 * L <= a0 and a1 <= sa1 + 0.1 * L          # a title along the bar on its far side
                if row and after and (same_side or within) and max(ix0 - lx1, lx0 - ix1) <= 1.0 * max(th, iy1 - iy0):
                    kind = "inline"
                    break
        if kind is None:
            continue
        centre = (it["cx"], it["cy"])
        d_strip = _edge_dist(centre, bbox)
        in_det = dx0 - ex <= centre[0] <= dx1 + ex and dy0 - ey <= centre[1] <= dy1 + ey
        slack = max(0.5 * th, 4.0)         # a unit above the bar is often about as near to the map as to the strip
        if not in_det and (any(_edge_dist(centre, mb) == 0 for mb in (map_boxes or []))
                           or any(_edge_dist(centre, mb) < d_strip - slack for mb in (map_boxes or []))
                           or any(_edge_dist(centre, ob) < d_strip - slack for ob in (other_bars or []))):
            continue
        cands.append((u, {"end": 0, "inline": 1, "beside": 2}[kind], -float(it["conf"]), d_strip))
    if cands:
        return min(cands, key=lambda t: (t[1], t[2], t[3]))[0]
    if end_slot_unreadable or not vertical or side is None or outer is None:
        return None
    # a second column of numbers on the same side (the tick labels of a neighbouring axis) makes a rotated text
    # beside the labels ambiguous - it is probably that axis's title - so no unit is taken from the band then
    foreign = 0
    for it in items:
        if parse_number_unit(fix_glued(it["text"]))[0] is None:
            continue
        a0, a1, c0, c1 = _ac(_item_bbox(it), vertical)
        ca, cc = (a0 + a1) / 2, (c0 + c1) / 2
        if not (sa0 - 0.3 * L <= ca <= sa1 + 0.3 * L):
            continue
        if any(abs(cc - (_ac(c["bbox"], vertical)[2] + _ac(c["bbox"], vertical)[3]) / 2) < 0.5 * th and abs(ca - c["ca"]) < th for c in col):
            continue
        beyond_col = (cc - outer) if side > 0 else (outer - cc)
        if 0 < beyond_col <= 8.0 * th:
            foreign += 1
    if foreign >= 2:
        return None
    # rotated unit label beside the labels: read only that band, in both rotations
    band_w = max(4.0 * th, 24.0)
    bx0, bx1 = (outer, outer + band_w) if side > 0 else (outer - band_w, outer)
    by0, by1 = sa0 - 0.3 * L, sa1 + 0.3 * L
    for ob in list(map_boxes or []) + list(other_bars or []):      # the band ends at a map or another bar
        ox0, oy0, ox1, oy1 = ob
        if oy1 <= by0 or oy0 >= by1:
            continue
        if side > 0 and ox0 >= outer:
            bx1 = min(bx1, outer + 0.5 * (ox0 - outer))       # the title of that map's axis sits in the nearer half
        elif side < 0 and ox1 <= outer:
            bx0 = max(bx0, outer - 0.5 * (outer - ox1))
    bx0, by0, bx1, by1 = int(max(0, bx0)), int(max(0, by0)), int(min(W, bx1)), int(min(H, by1))
    if bx1 - bx0 < 6 or by1 - by0 < 12:
        return None
    sub_img = np.ascontiguousarray(crop[by0:by1, bx0:bx1])
    found, texts = [], []
    for k in (-1, 1):
        for it in ocr.read(np.ascontiguousarray(np.rot90(sub_img, k=k))):
            texts.append(str(it["text"]))
            u = parse_unit(it["text"])
            if accept(u):
                found.append((u, float(it["conf"])))
    if any("log" in t.lower() or ("/" in t and canonical_unit(t) is None) for t in texts):
        return None                                     # 'Log(E/MPa)', 'kcounts/80 ms': a quantity, not the bar's unit
    return max(found, key=lambda t: t[1])[0] if found else None


def verbatim_ends(fit, anchors, tol=0.05):
    """End labels taken as written (``end_labels="verbatim"``, revision R1, last change). An anchor that the reader placed
    at an end of the strip (position 0 or 1: a label beyond the end, or within about a text height of it) gives that end
    its own value instead of the value of the line fitted through all labels, when the two agree within ``tol`` of the
    span; a label far off the line is an outlier that the fit rejected and is not used. Why: AFM software writes the
    range of the data at the ends of the bar (0.00 / 354.88 nm), and the least-squares line through slightly misplaced
    label positions turned such ranges into -3.47..358.3. Returns the fit, updated when an end changed."""
    if not fit.get("ok") or fit.get("sign_repaired") or fit.get("end_labels_only") or fit.get("scale") == "log":
        return fit
    sc = float(fit.get("offset_scale") or 1.0)
    s, e = float(fit["value_at_start"]), float(fit["value_at_end"])
    span = abs(e - s) + 1e-12
    e0 = {float(a[1]) * sc for a in anchors if a[0] == 0.0 and abs(float(a[1]) * sc - s) <= tol * span}
    e1 = {float(a[1]) * sc for a in anchors if a[0] == 1.0 and abs(float(a[1]) * sc - e) <= tol * span}
    s2 = next(iter(e0)) if len(e0) == 1 else s
    e2 = next(iter(e1)) if len(e1) == 1 else e
    if s2 == s and e2 == e:
        return fit
    return {**fit, "value_at_start": s2, "value_at_end": e2, "vmin": min(s2, e2), "vmax": max(s2, e2),
            "start_is_max": s2 > e2, "end_labels_verbatim": [s2 != s, e2 != e]}


def read_colorbar_v2(crop, ocr, roi, det_box, other_bars, map_boxes, n=256, unit_mode="raw", end_labels="fit"):
    """crop: padded RGB crop; roi/det_box: detector bar box in crop coordinates; other_bars, map_boxes: the other
    detected bars and the maps in crop coordinates; end_labels: "fit" or "verbatim" (see verbatim_ends). Returns a dict
    with the same keys as read_colorbar."""
    H, W = crop.shape[:2]
    vs = verified_strip_bbox(crop, roi, map_boxes)
    if vs is None:
        return {"ok": False, "reason": "no solid gradient region", "bbox": None, "orientation": None, "fit": None, "ocr_items": [],
                "touches_border": None}
    bbox, orientation, core = vs
    vertical = orientation == "vertical"
    det_ac = (int(roi[1]), int(roi[3])) if vertical else (int(roi[0]), int(roi[2]))
    # the gradient component may run into a stacked bar or a touching map: along the bar the core is trusted only
    # within the detector box plus a small margin (the light-end recovery may still extend it to a frame line)
    m_clip = max(4.0, 0.03 * (det_ac[1] - det_ac[0]))
    lo_c, hi_c = det_ac[0] - m_clip, det_ac[1] + m_clip
    def _clip(b):
        if vertical:
            return (b[0], int(max(b[1], lo_c)), b[2], int(min(b[3], hi_c)))
        return (int(max(b[0], lo_c)), b[1], int(min(b[2], hi_c)), b[3])
    if _clip(core)[3 if vertical else 2] - _clip(core)[1 if vertical else 0] >= 8 and _clip(bbox) != tuple(bbox):
        # the part beyond the box is kept only as far as it continues the gradient row by row (a tight detector
        # box), not when it is another bar or a touching map (a colour jump or a page gap)
        walked, _, _ = extend_by_walk(crop, _clip(bbox), vertical, det_ac, _background_and_diff(crop)[0],
                                      limits=(bbox[1], bbox[3]) if vertical else (bbox[0], bbox[2]), light_only=False)
        bbox = _restore_beyond(crop, _clip(bbox), tuple(int(v) for v in walked), vertical, other_bars, map_boxes)
        core = (core[0], max(core[1], bbox[1]), core[2], min(core[3], bbox[3])) if vertical else \
               (max(core[0], bbox[0]), core[1], min(core[2], bbox[2]), core[3])
    bbox, ext_lo, ext_hi = recover_light_end(crop, bbox, vertical, det_ac)
    bbox, w_lo, w_hi = extend_by_walk(crop, bbox, vertical, det_ac, _background_and_diff(crop)[0], margin_frac=0.0, margin_min=0.0)
    ext_lo, ext_hi = ext_lo + w_lo, ext_hi + w_hi
    geom0 = {"core": tuple(int(v) for v in core), "strip_out": tuple(int(v) for v in bbox), "det": tuple(int(v) for v in roi)}
    if not _strip_in_box(bbox, roi, vertical):
        return {"ok": False, "reason": "gradient region lies outside the detector box", "bbox": bbox, "orientation": orientation,
                "fit": None, "ocr_items": [], "touches_border": None, "geom_debug": geom0}
    inside = _inside_map(bbox, roi, map_boxes)
    if inside is not None and not _continues_into(crop, bbox, inside, vertical):
        inside = None                                   # a bar enclosed by an over-wide map box, not the map's edge
    if inside is not None:
        return {"ok": False, "reason": "gradient region lies inside a map panel", "bbox": bbox, "orientation": orientation,
                "fit": None, "ocr_items": [], "touches_border": None, "geom_debug": {**geom0, "map": inside}}
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
            vals = split_merged_numbers(it["text"])
            if vals and len(vals) >= 2:
                bx0, by0, bx1, by1 = _item_bbox(it)
                edges = np.linspace(bx0, bx1, len(vals) + 1)      # one sub-box per label, side by side
                extra += [{"text": f"{v:g}", "conf": it["conf"], "cx": float(0.5 * (e0 + e1)), "cy": it["cy"], "h": it["h"],
                           "box": [[float(e0), by0], [float(e1), by0], [float(e1), by1], [float(e0), by1]]}
                          for v, e0, e1 in zip(vals, edges[:-1], edges[1:])]
        if not extra and len(numeric_items(items1)) < 3:
            extra = ocr.read_horizontal_labels(crop, bbox, items1)
        if extra:
            items1 = [it for it in items1 if parse_number_unit(fix_glued(it["text"]))[0] is None
                      or parse_offset(it["text"]) is not None] + extra          # keep exponent offsets ('1e-12')
    items1 = _with_ink(crop, items1, vertical)
    attempts = [_attempt(items1, bbox, (ext_lo, ext_hi), *ctx, crop=crop)]
    best = attempts[0]
    f1 = best["fit"]
    pos1 = [a[0] for a in best["anchors"]]
    covers = bool(pos1) and (min(pos1) <= 0.2 and max(pos1) >= 0.8)
    clean = (f1.get("ok") and not best["near_skipped"] and not f1.get("sign_repaired") and not f1.get("end_labels_only") and
             int(f1.get("n_inliers", 0) or 0) == len(best["anchors"]) >= 2 and (covers or len(best["anchors"]) >= 3))
    if not clean:
        # pass 2: the crop at a larger scale when the labels are small
        h_med = best["text_h"] if best["cands"] else 0.0
        if not best["cands"] or h_med * base_scale < 28:
            target = 36.0 / max(h_med, 5.0) if h_med else max(2.0, 28.0 / max(w, 4.0))
            scale2 = float(np.clip(max(target, 1.5 * base_scale), 1.5, 8.0))
            if scale2 > base_scale * 1.1:
                attempts.append(_attempt(_with_ink(crop, ocr.read(crop, scale=scale2), vertical), bbox, (ext_lo, ext_hi), *ctx, crop=crop))
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
            items3 += [it for it in items1 if parse_number_unit(fix_glued(it["text"]))[0] is None
                       or parse_offset(it["text"]) is not None]   # unit texts and exponent offsets of pass 1
            attempts.append(_attempt(_with_ink(crop, items3, vertical), bbox, (ext_lo, ext_hi), *ctx, crop=crop))
        best = max(attempts, key=_score)

    fit, bbox = best["fit"], best["bbox"]
    if end_labels == "verbatim":
        fit = verbatim_ends(fit, best["anchors"])
    sa0, sa1, sc0, sc1 = _ac(bbox, vertical)
    L = sa1 - sa0
    items, unit = best["items"], best["unit"]
    touch_c = False                                       # a label of this bar cut by the crop border across the bar
    dx0, dy0, dx1, dy1 = det_box
    for it in items:
        if parse_number_unit(fix_glued(it["text"]))[0] is None and parse_unit(it["text"]) is None:
            continue
        bx0, by0, bx1, by1 = _item_bbox(it)
        ia0, ia1, ic0, ic1 = _ac((bx0, by0, bx1, by1), vertical)
        if ia1 < sa0 - 0.3 * L or ia0 > sa1 + 0.3 * L:                       # not along this bar
            continue
        centred = ic0 < sc1 and ic1 > sc0                                    # an end label centred on the strip
        if not centred:
            continue
        touch_c = touch_c or ((bx0 <= 3 or bx1 >= W - 4) if vertical else (by0 <= 3 or by1 >= H - 4))
    mx, my = 0.25 * (dx1 - dx0), 0.25 * (dy1 - dy0)
    if unit is None or (unit_mode == "lexicon" and canonical_unit(unit) is None):
        unit = _unit_near_labels(crop, ocr, items, best, bbox, vertical, det_box, other_bars, map_boxes, unit_mode) or unit
    unit_raw = unit
    if unit_mode == "lexicon":
        unit = canonical_unit(unit)
    strip = sample_strip(crop, bbox, orientation, n=n)
    q = strip_quality(strip) if strip is not None else None
    out = {"ok": bool(fit.get("ok")) and strip is not None, "reason": fit.get("reason") if strip is not None else "strip sampling failed",
           "unit": unit, "unit_raw": unit_raw, "orientation": orientation, "bbox": bbox, "fit": fit,
           "ocr_items": [(it["text"], round(it["conf"], 3)) for it in items], "strip_quality": q,
           "n_anchors": fit.get("n_anchors", 0), "label_column": best["how"], "touches_border": touches, "touches_across": touch_c,
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
        for _ in range(4):
            px, py = (pad_c, pad_a) if vertical else (pad_a, pad_c)
            cx0, cy0 = int(max(0, x0 - px)), int(max(0, y0 - py))
            cx1, cy1 = int(min(W, x1 + px)), int(min(H, y1 + py))
            crop = np.ascontiguousarray(image_rgb[cy0:cy1, cx0:cx1])
            shift = lambda bx: (bx[0] - cx0, bx[1] - cy0, bx[2] - cx0, bx[3] - cy0)
            roi = tuple(int(round(v)) for v in shift(b["xyxy"]))
            r_new = read_colorbar_v2(crop, pipe.ocr, roi, roi, [shift(o) for j, o in enumerate(all_boxes) if j != i],
                                     [shift(m) for m in map_boxes], unit_mode=pipe.unit_mode,
                                     end_labels=getattr(pipe, "end_labels", "fit"))
            r_new["crop_box"] = (cx0, cy0, cx1, cy1)
            if r is not None and r.get("ok") and not r_new.get("ok"):
                break                                            # the regrown crop read worse: keep the earlier result
            r = r_new
            t = r.get("touches_border") or ""
            grow_a = (("lo" in t and (cy0 > 0 if vertical else cx0 > 0)) or ("hi" in t and (cy1 < H if vertical else cx1 < W))) \
                and pad_a < 1.5 * max(bw, bh)
            # an end label wider than the crop (centred on the bar) is cut by the crop border across the bar
            grow_c = bool(r.get("touches_across")) and pad_c < 1.5 * min(bw, bh) + 36 and \
                ((cx0 > 0 or cx1 < W) if vertical else (cy0 > 0 or cy1 < H))
            if not grow_a and not grow_c:
                break
            if grow_a:
                pad_a *= 2.0
            if grow_c:
                pad_c *= 2.0
        r.update({"box": b["xyxy"], "det_conf": b["conf"]})
        out.append(r)
    return out
