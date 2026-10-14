"""Read the numeric scale of a colour bar with OCR and convert it into the
(vmin, vmax) endpoints required by the quantification engine.

OCR engine: PP-OCRv4 detection/recognition models (PaddleOCR family) executed
with RapidOCR / ONNX Runtime (``rapidocr-onnxruntime``), which needs no Paddle
installation and gives deterministic CPU results.

Pipeline
--------
1. Upscale the colour-bar crop so that tick labels are >= ~20 px high.
2. OCR -> text boxes. Numeric labels become *anchors* (position along the bar
   axis, value); texts with letters are parsed as the unit; matplotlib-style
   offset/exponent texts ("1e-9", "x10^-3") become a scale factor.
3. Anchors are kept only if they sit beside the gradient strip (a corridor of
   a few bar widths on either side) - this rejects panel labels, scale-bar text
   and other numbers inside the crop.
4. The value-position relation is fitted linearly (>= 2 anchors). The fitted
   line gives the values at the two ends of the strip, including the common case
   where the outermost tick labels lie inside the bar ("interior tick labels").
   A log10 fit is tried when the linear residual is large and all values are
   positive; such bars are flagged (``scale == "log"``) and not quantified.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import cv2

_NUM = r"[-−–+]?\d+(?:[.,]\d+)?"
_RE_NUMBER = re.compile(rf"^\s*({_NUM})\s*$")
_RE_SCI = re.compile(rf"^\s*({_NUM})\s*(?:[eE]|[x×]\s*10\s*\^?\s*)([-−–+]?\d+)\s*$")
_RE_OFFSET = re.compile(r"^\s*(?:1\s*e|[x×]?\s*10\s*\^?\s*)([-−–+]?\d+)\s*$")
_RE_UNIT = re.compile(r"[A-Za-zµμ°]")
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _norm_minus(s: str) -> str:
    return s.replace("−", "-").replace("–", "-").replace(",", ".").translate(_SUPERSCRIPT)


def parse_number(text: str) -> Optional[float]:
    """Parse a tick label; returns None if the text is not a number."""
    t = _norm_minus(text).strip()
    t = re.sub(r"\s+", "", t) if re.fullmatch(r"[-+]?\d[\d\s]*([.]\d+)?", t) else t
    m = _RE_SCI.match(t)
    if m:
        return float(m.group(1)) * 10.0 ** float(m.group(2))
    m = _RE_NUMBER.match(t)
    if m:
        return float(m.group(1))
    return None


def parse_offset(text: str) -> Optional[float]:
    """Matplotlib offset text such as '1e-9' or 'x10^-3' -> scale factor."""
    m = _RE_OFFSET.match(_norm_minus(text).strip())
    return 10.0 ** float(m.group(1)) if m else None


def parse_unit(text: str) -> Optional[str]:
    t = text.strip()
    if parse_number(t) is not None or parse_offset(t) is not None:
        return None
    if _RE_UNIT.search(t):
        t = re.sub(r"^[\(\[]|[\)\]]$", "", t)
        return t if 0 < len(t) <= 20 else None
    return None


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
        (in crop pixel coordinates), and the box."""
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


def anchors_from_items(items, bar_bbox, orientation, corridor=4.0):
    """Numeric OCR items beside the bar -> list of (pos_frac, value, conf).

    ``pos_frac`` is the position along the strip axis (0 = top/left end of the
    gradient region, 1 = bottom/right end).
    """
    x0, y0, x1, y1 = bar_bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    anchors, units, offsets = [], [], []
    for it in items:
        txt = it["text"]
        off = parse_offset(txt)
        if off is not None:
            offsets.append(off)
            continue
        unit = parse_unit(txt)
        if unit is not None:
            units.append((unit, it["conf"]))
            continue
        val = parse_number(txt)
        if val is None:
            continue
        if orientation == "vertical":
            beside = (x1 - corridor * bw <= it["cx"] <= x1 + corridor * bw) and \
                     (y0 - 0.5 * bh <= it["cy"] <= y1 + 0.5 * bh)
            pos = (it["cy"] - y0) / bh
        else:
            beside = (y0 - corridor * bh <= it["cy"] <= y1 + corridor * bh) and \
                     (x0 - 0.5 * bw <= it["cx"] <= x1 + 0.5 * bw)
            pos = (it["cx"] - x0) / bw
        if beside:
            anchors.append((float(pos), float(val), float(it["conf"])))
    unit = max(units, key=lambda u: u[1])[0] if units else None
    scale = offsets[0] if offsets else 1.0
    return anchors, unit, scale


def fit_scale(anchors, scale=1.0, max_rel_residual=0.03):
    """Fit value = a + b * pos_frac. Returns dict with vmin/vmax at the strip
    ends (pos 0 and 1), the direction and diagnostics."""
    if len(anchors) < 2:
        return {"ok": False, "reason": f"need >= 2 numeric tick labels, found {len(anchors)}",
                "n_anchors": len(anchors)}
    pos = np.array([a[0] for a in anchors]); val = np.array([a[1] for a in anchors]) * scale
    conf = np.array([a[2] for a in anchors])
    if np.ptp(pos) < 0.05:
        return {"ok": False, "reason": "tick labels not spread along the bar", "n_anchors": len(anchors)}
    b, a = np.polyfit(pos, val, 1)
    pred = a + b * pos
    span = abs(b) + 1e-12
    resid = float(np.abs(pred - val).max() / span)
    out = {"n_anchors": int(len(anchors)), "residual_rel": resid, "scale": "linear",
           "ocr_conf_min": float(conf.min()), "offset_scale": float(scale)}
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


def read_scale(crop_rgb, bar_bbox, orientation, ocr: ColorbarOCR):
    """Full scale reading for one colour-bar crop whose gradient region is
    ``bar_bbox`` (from :func:`afmquant.colorbar.find_gradient_region`)."""
    items = ocr.read(crop_rgb)
    anchors, unit, scale = anchors_from_items(items, bar_bbox, orientation)
    fit = fit_scale(anchors, scale)
    fit.update({"unit": unit, "ocr_items": [(it["text"], round(it["conf"], 3)) for it in items],
                "anchors": anchors})
    return fit


def oriented_strip_and_endpoints(strip, fit):
    """Order the strip vmax->vmin as the engine expects and return
    ``(strip, vmin, vmax)``."""
    s = strip if fit["start_is_max"] else strip[::-1]
    return s, fit["vmin"], fit["vmax"]
