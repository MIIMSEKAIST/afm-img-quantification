"""Read the numeric scale of a colour bar with OCR and convert it into the
(vmin, vmax) endpoints required by the quantification engine.

OCR engine: PP-OCRv4 detection/recognition models (PaddleOCR family) executed
with RapidOCR / ONNX Runtime (``rapidocr-onnxruntime``), which needs no Paddle
installation and gives deterministic CPU results.

Pipeline (:func:`read_colorbar`)
--------------------------------
1. Locate and sample the gradient strip (:mod:`afmquant.colorbar`).
2. Upscale the crop so that tick labels are >= ~20 px high and run OCR.
   Numeric labels become *anchors* (position along the bar axis, value); texts
   with letters are parsed as the unit; matplotlib-style offset texts ("1e-9",
   "x10^-3") become a scale factor.
3. Anchors are kept only if they sit beside the gradient strip (a corridor of
   a few bar widths on either side) - this rejects panel labels, scale-bar text
   and other numbers inside the crop.
4. Tick labels always lie inside the bar: if the outermost anchors fall outside
   the gradient region found in step 1 (bar ends invisible against the page),
   the region is widened to them and the strip is re-sampled.
5. The value-position relation is fitted linearly (>= 2 anchors), which also
   handles outermost tick labels lying inside the bar ("interior ticks"). A
   log10 fit is tried when the linear residual is large and all values are
   positive; such bars are flagged (``scale == "log"``) and not quantified.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import cv2

from .colorbar import extract_strip

_NUM = r"[-−–+]?\d+(?:[.,]\d+)?"
_RE_NUMBER = re.compile(rf"^\s*({_NUM})\s*$")
_RE_SCI = re.compile(rf"^\s*({_NUM})\s*(?:[eE]|[x×]\s*10\s*\^?\s*)([-−–+]?\d+)\s*$")
_RE_OFFSET = re.compile(r"^\s*(?:1\s*[eE]\s*([-+]?\d+)|[x×]\s*10\s*\^?\s*([-+]?\d+)|10\s*\^\s*([-+]?\d+))\s*$")
_RE_UNIT = re.compile(r"[A-Za-zµμ°]")
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻",
                             "0123456789-")


def _normalize(s: str) -> str:
    s = s.replace("−", "-").replace("–", "-").translate(_SUPERSCRIPT).strip()
    # OCR confusions in offset texts: "le-7", "Ie-7", "|e-7" -> "1e-7"
    s = re.sub(r"^[lI|]\s*(?=[eE]\s*[-+]?\d)", "1", s)
    # "1,000" -> thousands separator; "0,5" -> decimal comma
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    return s.replace(",", ".")


def parse_number(text: str) -> Optional[float]:
    """Parse a tick label; returns None if the text is not a number."""
    t = _normalize(text)
    m = _RE_SCI.match(t)
    if m:
        return float(m.group(1)) * 10.0 ** float(m.group(2))
    m = _RE_NUMBER.match(t)
    if m:
        return float(m.group(1))
    return None


def parse_offset(text: str) -> Optional[float]:
    """Matplotlib offset text such as '1e-9', 'x10^-3' or '10^-3' -> factor."""
    m = _RE_OFFSET.match(_normalize(text))
    if not m:
        return None
    exp = next(g for g in m.groups() if g is not None)
    return 10.0 ** float(exp)


_KNOWN_UNITS = {
    "nm", "um", "µm", "μm", "pm", "mm", "Å", "mv", "µv", "uv", "kv",
    "pa", "na", "µa", "ua", "ma", "deg", "°", "rad", "mrad", "hz", "khz", "mhz",
    "n/m", "nn", "pn", "µn", "un", "mn", "a.u.", "a.u", "arb.", "arb", "%", "mpa", "gpa",
    "ns", "µs", "us", "ms", "ev", "mev", "°c", "ff", "pf", "nm/v", "pm/v",
    "mv/v", "mj/m2", "j/m2", "nm2", "µm2", "um2", "cps", "counts", "mv/nm", "nn/nm",
}   # single letters (V, A, K, s, e) are left out: indistinguishable from panel labels
_UNIT_TOKEN = r"[A-Za-zµμ°Å/%.²³]{1,10}"
_RE_NUM_UNIT = re.compile(rf"^\s*({_NUM}(?:\s*(?:[eE]|[x×]\s*10\s*\^?\s*)[-+]?\d+)?)\s*({_UNIT_TOKEN})\s*$")


def _unit_score(u: str) -> int:
    """2 = known physical unit, 1 = plausible short token, 0 = reject."""
    key = u.strip().lower().replace("²", "2")
    if key in _KNOWN_UNITS:
        return 2
    if 2 <= len(key) <= 6 and _RE_UNIT.search(key) and not key.isdigit():
        return 1
    return 0


def parse_number_unit(text: str):
    """'80 nm', '+4.5 Hz', '-0.5um' -> (value, unit); '80' -> (80.0, None)."""
    t = _normalize(text)
    v = parse_number(t)
    if v is not None:
        return v, None
    m = _RE_NUM_UNIT.match(t)
    if m:
        v = parse_number(m.group(1))
        if v is not None:
            u = m.group(2).strip(".")
            return v, (u if _unit_score(u) > 0 else None)
    return None, None


def parse_unit(text: str) -> Optional[str]:
    """Unit-only label ('nm', 'Amplitude (pm)', 'deg'); None if not a unit."""
    t = text.strip()
    if parse_number(t) is not None or parse_offset(t) is not None:
        return None
    if not _RE_UNIT.search(t):
        return None
    m = re.search(r"[\(\[]([^\)\]]{1,20})[\)\]]?", t)      # "Amplitude (pm)" -> "pm"
    cand = m.group(1).strip() if m else t
    if _unit_score(cand) > 0:
        return cand
    # last token of a label such as "Height nm"
    tok = cand.split()[-1] if cand.split() else cand
    return tok if _unit_score(tok) == 2 else None


def split_merged_decimals(text: str):
    """'0.000.160.320.48' (adjacent labels merged by OCR) -> [0.0, 0.16, 0.32, 0.48].
    Only unambiguous cases (all labels with the same number of decimals, no
    separators, monotonic result) are split; returns None otherwise."""
    t = _normalize(text).replace(" ", "")
    if t.count(".") < 2 or not re.fullmatch(r"[-+]?[\d.]+", t):
        return None
    for d in (1, 2, 3, 4):
        parts = re.findall(rf"-?\d+\.\d{{{d}}}", t)
        if len(parts) >= 2 and "".join(parts) == t:
            vals = [float(p) for p in parts]
            if all(np.diff(vals) > 0) or all(np.diff(vals) < 0):
                return vals
    return None


def _segments_along(crop_rgb, band, axis_range, min_gap_px):
    """Split a label band (rows for horizontal bars) into text segments by the
    whitespace columns of its intensity profile. Returns (start, end) pairs."""
    y0, y1 = band
    x0, x1 = axis_range
    strip = crop_rgb[y0:y1, x0:x1].astype(int)
    bg = np.median(np.concatenate([strip[0], strip[-1]]), axis=0)
    ink = (np.abs(strip - bg).max(axis=2) > 60).any(axis=0)
    segs, start, gap = [], None, 0
    for i, v in enumerate(ink):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap_px:
                segs.append((start + x0, i - gap + 1 + x0))
                start, gap = None, 0
    if start is not None:
        segs.append((start + x0, len(ink) + x0))
    return [s for s in segs if s[1] - s[0] >= 3]


class ColorbarOCR:
    """Thin wrapper around RapidOCR with image upscaling."""

    def __init__(self, min_short_side=240):
        self.min_short_side = min_short_side
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def read(self, crop_rgb):
        """OCR on an RGB crop. Returns list of dicts with text, conf, cx, cy
        (crop pixel coordinates), h and the box."""
        h, w = crop_rgb.shape[:2]
        scale = max(1.0, self.min_short_side / max(1, min(h, w)))
        img = crop_rgb if scale == 1.0 else cv2.resize(
            crop_rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
        bgr = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        result, _ = self.engine(bgr)
        items = []
        for box, text, conf in (result or []):
            pts = np.asarray(box, dtype=float) / scale
            items.append({"text": str(text), "conf": float(conf),
                          "cx": float(pts[:, 0].mean()), "cy": float(pts[:, 1].mean()),
                          "h": float(pts[:, 1].max() - pts[:, 1].min()),
                          "box": pts.tolist()})
        return items

    def read_horizontal_labels(self, crop_rgb, bar_bbox, items):
        """Second pass for horizontal bars: tick labels sit in a row below (or
        above) the bar and are often merged into one text box by the detector.
        The label row is cut into segments at whitespace columns and each
        segment is recognised separately (upscaled). Returns new items."""
        x0, y0, x1, y1 = bar_bbox
        H, W = crop_rgb.shape[:2]
        text_h = float(np.median([it["h"] for it in items])) if items else max(8.0, 0.3 * (y1 - y0))
        rows = [it for it in items if parse_number_unit(it["text"])[0] is not None or
                split_merged_decimals(it["text"]) is not None]
        if rows:
            cy = float(np.median([it["cy"] for it in rows]))
            band = (int(max(0, cy - 0.8 * text_h)), int(min(H, cy + 0.8 * text_h)))
        else:
            band = (int(min(H - 1, y1 + 2)), int(min(H, y1 + 2 + 2.0 * text_h)))
        if band[1] - band[0] < 4:
            return []
        segs = _segments_along(crop_rgb, band, (max(0, x0 - 3 * int(text_h)), min(W, x1 + 3 * int(text_h))),
                               min_gap_px=max(2, int(0.35 * text_h)))
        out = []
        for sx0, sx1 in segs:
            pad = int(0.5 * text_h) + 2
            sub = crop_rgb[max(0, band[0] - pad):min(H, band[1] + pad), max(0, sx0 - pad):min(W, sx1 + pad)]
            if sub.size == 0:
                continue
            s = max(1.0, 64.0 / max(1, sub.shape[0]))
            sub_up = cv2.resize(sub, (int(sub.shape[1] * s), int(sub.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
            res, _ = self.engine(cv2.cvtColor(np.ascontiguousarray(sub_up), cv2.COLOR_RGB2BGR))
            for box, text, conf in (res or []):
                pts = np.asarray(box, dtype=float) / s
                out.append({"text": str(text), "conf": float(conf),
                            "cx": float(pts[:, 0].mean()) + max(0, sx0 - pad),
                            "cy": float(pts[:, 1].mean()) + max(0, band[0] - pad),
                            "h": float(pts[:, 1].max() - pts[:, 1].min()), "box": pts.tolist(),
                            "segmented": True})
        return out


def anchors_from_items(items, bar_bbox, orientation, corridor=4.0, axis_slack=0.6, crop_shape=None):
    """Numeric OCR items beside the bar -> (anchors, unit, offset_scale).

    Each anchor is ``(pos_frac, value, conf, axis_px)``; ``pos_frac`` is the
    position along the strip axis (0 = top/left end of the gradient region,
    1 = bottom/right end) and ``axis_px`` the same position in crop pixels.
    Labels are accepted within a corridor beside the bar whose width is the
    largest of ``corridor`` bar widths, three text heights and 35 % of the
    crop (thin bars would otherwise lose their labels).
    """
    x0, y0, x1, y1 = bar_bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    text_h = float(np.median([it["h"] for it in items])) if items else 0.0
    if orientation == "vertical":
        corr_px = max(corridor * bw, 3 * text_h, 0.35 * (crop_shape[1] if crop_shape else 0))
    else:
        corr_px = max(corridor * bh, 3 * text_h, 0.35 * (crop_shape[0] if crop_shape else 0))
    anchors, units, offsets, anchor_units = [], [], [], []
    for it in items:
        txt = it["text"]
        off = parse_offset(txt)
        if off is not None:
            offsets.append(off)
            continue
        # offset text glued to a unit label by the OCR, e.g. "(pA)1e7", "Amplitude (nm) 1e-9"
        m_off = re.search(r"[\)\]\s][1lI]\s*[eE]\s*([-+−–]?\d{1,2})\s*$", _normalize(txt))
        if m_off is not None and parse_number_unit(txt)[0] is None:
            offsets.append(10.0 ** float(m_off.group(1).replace("−", "-").replace("–", "-")))
            txt = txt[:m_off.start() + 1]
        val, trailing_unit = parse_number_unit(txt)
        if val is None:
            unit = parse_unit(txt)
            if unit is not None:
                units.append((unit, it["conf"]))
            continue
        if trailing_unit is not None:
            units.append((trailing_unit, it["conf"]))
        if orientation == "vertical":
            beside = (x0 - corr_px <= it["cx"] <= x1 + corr_px) and \
                     (y0 - axis_slack * bh <= it["cy"] <= y1 + axis_slack * bh)
            axis_px, pos = it["cy"], (it["cy"] - y0) / bh
        else:
            beside = (y0 - corr_px <= it["cy"] <= y1 + corr_px) and \
                     (x0 - axis_slack * bw <= it["cx"] <= x1 + axis_slack * bw)
            axis_px, pos = it["cx"], (it["cx"] - x0) / bw
        if beside:
            side = (it["cx"] > (x0 + x1) / 2) if orientation == "vertical" else (it["cy"] > (y0 + y1) / 2)
            anchors.append((float(pos), float(val), float(it["conf"]), float(axis_px), bool(side)))
            anchor_units.append(trailing_unit)
    # tick labels of one bar sit on one side of it; numbers on the other side
    # belong to something else (e.g. the tick labels of a neighbouring axis)
    if len(anchors) >= 3:
        n_true = sum(1 for a in anchors if a[4])
        majority = n_true >= len(anchors) - n_true
        keep = [i for i, a in enumerate(anchors) if a[4] == majority]
        if 2 <= len(keep) < len(anchors):
            anchors = [anchors[i] for i in keep]
            anchor_units = [anchor_units[i] for i in keep]
    anchors = [a[:4] for a in anchors]
    unit = None
    tagged = [u for u in anchor_units if u is not None]
    if tagged:
        # unit attached to the tick labels themselves: majority vote
        counts = {}
        for u in tagged:
            counts[u.lower()] = counts.get(u.lower(), 0) + 1
        unit = max(counts, key=counts.get)
        # drop numeric labels carrying a different unit (e.g. a "5 um" scale bar)
        keep = [i for i, u in enumerate(anchor_units) if u is None or u.lower() == unit]
        if 2 <= len(keep) < len(anchors):
            anchors = [anchors[i] for i in keep]
    elif units:
        unit = max(units, key=lambda u: (_unit_score(u[0]), u[1]))[0]
    scale = offsets[0] if offsets else 1.0
    return anchors, unit, scale


def fit_scale(anchors, scale=1.0, max_rel_residual=0.03):
    """Fit value = a + b * pos_frac. Returns dict with vmin/vmax at the strip
    ends (pos 0 and 1), the direction and diagnostics."""
    if len(anchors) < 2:
        return {"ok": False, "reason": f"need >= 2 numeric tick labels, found {len(anchors)}",
                "n_anchors": len(anchors)}
    pos = np.array([a[0] for a in anchors])
    val = np.array([a[1] for a in anchors]) * scale
    conf = np.array([a[2] for a in anchors])
    if np.ptp(pos) < 0.05:
        return {"ok": False, "reason": "tick labels not spread along the bar",
                "n_anchors": len(anchors)}
    n_in = len(anchors)
    b, a = np.polyfit(pos, val, 1)
    pred = a + b * pos
    span = abs(b) + 1e-12
    resid = float(np.abs(pred - val).max() / span)
    if resid > max_rel_residual and len(anchors) >= 3:
        # one misread label breaks the fit: pairwise consensus (RANSAC-style)
        best = None
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if abs(pos[i] - pos[j]) < 0.05:
                    continue
                bb = (val[j] - val[i]) / (pos[j] - pos[i])
                aa = val[i] - bb * pos[i]
                inl = np.abs(aa + bb * pos - val) <= max_rel_residual * (abs(bb) + 1e-12)
                k = int(inl.sum())
                # a consensus needs at least three labels: with three anchors and
                # one misread label every pair "agrees" with itself, so the bar is
                # rejected as inconsistent rather than read with a wrong scale
                if k >= max(3, int(np.ceil(0.6 * len(pos)))) and (best is None or k > best[0]):
                    best = (k, inl)
        if best is not None:
            inl = best[1]
            b, a = np.polyfit(pos[inl], val[inl], 1)
            pos, val, conf = pos[inl], val[inl], conf[inl]
            pred = a + b * pos
            span = abs(b) + 1e-12
            resid = float(np.abs(pred - val).max() / span)
            n_in = int(inl.sum())
    out = {"n_anchors": int(len(anchors)), "n_inliers": int(n_in), "residual_rel": resid,
           "scale": "linear", "ocr_conf_min": float(conf.min()), "offset_scale": float(scale)}
    if resid > max_rel_residual and np.all(val > 0) and len(anchors) >= 3:
        bl, al = np.polyfit(pos, np.log10(val), 1)
        resid_log = float(np.abs(al + bl * pos - np.log10(val)).max() / (abs(bl) + 1e-12))
        if resid_log < resid:
            out.update({"ok": False, "scale": "log", "residual_rel": resid_log,
                        "reason": "logarithmic colour scale (not supported)"})
            return out
    if resid > max_rel_residual:
        out.update({"ok": False, "reason": f"inconsistent tick labels (residual {resid:.2%})"})
        return out
    v_start, v_end = float(a), float(a + b)       # values at pos 0 and pos 1
    out.update({"ok": True, "reason": None, "value_at_start": v_start, "value_at_end": v_end,
                "vmin": min(v_start, v_end), "vmax": max(v_start, v_end),
                "start_is_max": v_start > v_end})
    return out


def read_colorbar(crop_rgb, ocr: ColorbarOCR, n=256, margin_frac=0.01, roi=None):
    """Colour-bar crop -> calibration strip ordered vmax->vmin, vmin, vmax.

    ``roi`` is the detector's bar box in crop coordinates (optional prior).

    Returns a dict: ``ok``, ``reason``, ``strip`` (n x 3, vmax->vmin),
    ``vmin``, ``vmax``, ``unit``, ``orientation``, ``bbox``, ``fit`` (all
    fit diagnostics), ``ocr_items``, ``strip_quality``.
    """
    ex = extract_strip(crop_rgb, n=n, roi=roi)
    if not ex["ok"]:
        return {"ok": False, "reason": ex.get("reason"), "bbox": ex.get("bbox"),
                "orientation": ex.get("orientation"), "fit": None, "ocr_items": []}
    items = ocr.read(crop_rgb)
    shp = crop_rgb.shape[:2]
    anchors, unit, scale = anchors_from_items(items, ex["bbox"], ex["orientation"], crop_shape=shp)
    if ex["orientation"] == "horizontal" and len(anchors) < 3:
        # labels merged into one text box: split merged decimals, else re-read segments
        extra = []
        for it in items:
            vals = split_merged_decimals(it["text"])
            if vals and len(vals) >= 2:
                xs = np.linspace(it["box"][0][0], it["box"][1][0], len(vals))
                extra += [{"text": f"{v:g}", "conf": it["conf"], "cx": float(x), "cy": it["cy"],
                           "h": it["h"], "box": it["box"]} for v, x in zip(vals, xs)]
        if not extra:
            extra = ocr.read_horizontal_labels(crop_rgb, ex["bbox"], items)
        if extra:
            keep = [it for it in items if parse_number_unit(it["text"])[0] is None]
            a2, u2, s2 = anchors_from_items(keep + extra, ex["bbox"], ex["orientation"], crop_shape=shp)
            if len(a2) > len(anchors):
                items, anchors, unit, scale = keep + extra, a2, u2 or unit, s2 if s2 != 1.0 else scale
    if ex["orientation"] == "vertical" and unit is None:
        # unit labels of vertical bars are usually rotated by 90 degrees
        rot = ocr.read(np.ascontiguousarray(np.rot90(crop_rgb, k=-1)))
        units = [(parse_unit(it["text"]), it["conf"]) for it in rot]
        units = [u for u in units if u[0] is not None]
        if units:
            unit = max(units, key=lambda u: u[1])[0]

    # widen the gradient extent to the outermost tick labels if they lie outside;
    # only labels consistent with a preliminary fit of the labels inside the bar
    # are trusted (a number beside the bar end may belong to another axis)
    if anchors:
        inside = [a for a in anchors if -0.05 <= a[0] <= 1.05]
        trusted = list(inside)
        outside = [a for a in anchors if a not in inside]
        if outside and len(inside) >= 2 and np.ptp([a[0] for a in inside]) >= 0.05:
            b0, a0 = np.polyfit([a[0] for a in inside], [a[1] for a in inside], 1)
            trusted += [a for a in outside if abs(a0 + b0 * a[0] - a[1]) <= 0.05 * (abs(b0) + 1e-12)]
        px = [a[3] for a in trusted] if trusted else [a[3] for a in anchors]
        x0, y0, x1, y1 = ex["bbox"]
        lo_b, hi_b = (y0, y1) if ex["orientation"] == "vertical" else (x0, x1)
        L = hi_b - lo_b
        lo, hi = min(px) - margin_frac * L, max(px) + margin_frac * L
        if lo < lo_b - 0.02 * L or hi > hi_b + 0.02 * L:
            ex2 = extract_strip(crop_rgb, n=n, axis_extent=(min(lo, lo_b), max(hi, hi_b)), roi=roi)
            if ex2["ok"]:
                ex = ex2
                anchors, unit, scale = anchors_from_items(items, ex["bbox"], ex["orientation"], crop_shape=shp)

    fit = fit_scale(anchors, scale)
    out = {"ok": bool(fit.get("ok")), "reason": fit.get("reason"), "unit": unit,
           "orientation": ex["orientation"], "bbox": ex["bbox"], "fit": fit,
           "ocr_items": [(it["text"], round(it["conf"], 3)) for it in items],
           "strip_quality": ex["quality"], "n_anchors": fit.get("n_anchors", 0)}
    if fit.get("ok"):
        strip = ex["strip"] if fit["start_is_max"] else ex["strip"][::-1]
        out.update({"strip": strip, "vmin": fit["vmin"], "vmax": fit["vmax"]})
    return out
