"""End-to-end figure processing: detection -> colour-bar reading -> pairing ->
quantification -> quality control.

This is the "full pipeline" whose realistic accuracy is evaluated in the
revised manuscript (Tier 2 synthetic figures and Tier 3 published figures).
Detection uses the released YOLO11m map/colour-bar detector; everything after
detection is deterministic.

The defaults are the released configuration of the revised manuscript: the
fine-tuned detector (v15) at confidence 0.25, the label-column colour-bar
reader (``reader="v2"``), lexicon-constrained units (``unit_mode="lexicon"``)
and evidence-based map/colour-bar pairing (``pairing_mode="evidence"``). The
configuration of the submitted manuscript is reproduced with the v13 weights
and ``conf=0.5, reader="v1", unit_mode="raw", pairing_mode="nearest"``.

The defaults are the released configuration of the revised manuscript: the
fine-tuned detector (v15) at confidence 0.25, the label-column colour-bar
reader (``reader="v2"``), lexicon-constrained units (``unit_mode="lexicon"``)
and evidence-based map/colour-bar pairing (``pairing_mode="evidence"``). The
configuration of the submitted manuscript is reproduced with the v13 weights
and ``conf=0.5, reader="v1", unit_mode="raw", pairing_mode="nearest"``.

The defaults are the released configuration of the revised manuscript: the
fine-tuned detector (v15) at confidence 0.25, the label-column colour-bar
reader (``reader="v2"``), lexicon-constrained units (``unit_mode="lexicon"``)
and evidence-based map/colour-bar pairing (``pairing_mode="evidence"``). The
configuration of the submitted manuscript is reproduced with the v13 weights
and ``conf=0.5, reader="v1", unit_mode="raw", pairing_mode="nearest"``.

The defaults are the released configuration of the revised manuscript: the
fine-tuned detector (v15) at confidence 0.25, the label-column colour-bar
reader (``reader="v2"``), lexicon-constrained units (``unit_mode="lexicon"``)
and evidence-based map/colour-bar pairing (``pairing_mode="evidence"``). The
configuration of the submitted manuscript is reproduced with the v13 weights
and ``conf=0.5, reader="v1", unit_mode="raw", pairing_mode="nearest"``.

The defaults are the released configuration of the revised manuscript: the
fine-tuned detector (v15) at confidence 0.25, the label-column colour-bar
reader (``reader="v2"``), lexicon-constrained units (``unit_mode="lexicon"``)
and evidence-based map/colour-bar pairing (``pairing_mode="evidence"``). The
configuration of the submitted manuscript is reproduced with the v13 weights
and ``conf=0.5, reader="v1", unit_mode="raw", pairing_mode="nearest"``.

    from afmquant.pipeline import FigurePipeline
    pipe = FigurePipeline("detection/weights/afm_detect_best.pt")
    result = pipe.process(image_rgb)
    for panel in result["panels"]:
        panel["status"], panel["value_map"], panel["valid_mask"], panel["qc"]
"""
from __future__ import annotations

import numpy as np

from .engine import quantify_map
from .ocr import ColorbarOCR, read_colorbar
from .pairing import pair_maps_to_bars

CLASS_MAP, CLASS_BAR = 0, 1          # detection/data.yaml: ['afm_map', 'colorbar']


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_boxes(boxes, iou_threshold=0.3):
    """Greedy per-class NMS (same rule as detection/detect.py)."""
    boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if b["cls"] != best["cls"] or _iou(best["xyxy"], b["xyxy"]) < iou_threshold]
    return kept



class FigurePipeline:
    def __init__(self, weights, conf=0.25, iou_merge=0.3, color_space="cielab",
                 mask_space="cielab", distance_threshold=20.0, margin_trim=0.005,
                 bar_pad=0.5, min_compat=0.5, ocr: ColorbarOCR | None = None, device=None,
                 pairing_mode="evidence", unit_mode="lexicon", reader="v2"):
        self.weights = str(weights)
        self.conf = conf
        self.iou_merge = iou_merge
        self.engine_kw = dict(color_space=color_space, mask_space=mask_space,
                              distance_threshold=distance_threshold)
        self.margin_trim = margin_trim
        self.bar_pad = bar_pad
        self.min_compat = min_compat
        self.pairing_mode = pairing_mode          # "nearest" (submitted) | "evidence" (layout or exclusivity required)
        self.unit_mode = unit_mode                # "raw" (submitted) | "lexicon" (canonical SPM unit or unknown)
        self.reader = reader                      # "v1" (submitted) | "v2" (label-column reader, revision R1)
        self.ocr = ocr or ColorbarOCR()
        self.device = device
        self._model = None

    # ------------------------------------------------------------------ detection
    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, image_rgb):
        res = self.model.predict(image_rgb[:, :, ::-1].copy(), conf=self.conf, verbose=False,
                                 device=self.device)   # ultralytics expects BGR arrays
        boxes = []
        for r in res:
            for i in range(len(r.boxes)):
                boxes.append({"xyxy": [float(v) for v in r.boxes.xyxy[i].cpu().numpy()],
                              "conf": float(r.boxes.conf[i]), "cls": int(r.boxes.cls[i])})
        boxes = merge_boxes(boxes, self.iou_merge)
        maps = [b for b in boxes if b["cls"] == CLASS_MAP]
        bars = [b for b in boxes if b["cls"] == CLASS_BAR]
        return maps, bars

    # --------------------------------------------------------------- colour bars
    def _padded_crop(self, image_rgb, box):
        H, W = image_rgb.shape[:2]
        x0, y0, x1, y1 = box
        pad = self.bar_pad * min(x1 - x0, y1 - y0) + 8
        x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        x1, y1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
        return image_rgb[y0:y1, x0:x1], (x0, y0, x1, y1)

    def read_bars(self, image_rgb, bar_boxes, maps=None):
        if self.reader == "v2":
            from .reader2 import read_bars_v2
            return read_bars_v2(self, image_rgb, bar_boxes, maps or [])
        out = []
        for b in bar_boxes:
            crop, cbox = self._padded_crop(image_rgb, b["xyxy"])
            x0, y0, x1, y1 = b["xyxy"]
            roi = (x0 - cbox[0], y0 - cbox[1], x1 - cbox[0], y1 - cbox[1])   # detector box in crop coords
            r = read_colorbar(crop, self.ocr, roi=roi, unit_mode=self.unit_mode)
            r.update({"box": b["xyxy"], "crop_box": cbox, "det_conf": b["conf"]})
            out.append(r)
        return out

    # ------------------------------------------------------------------- process
    def process(self, image_rgb, figure_id=None):
        maps, bars_det = self.detect(image_rgb)
        bars = self.read_bars(image_rgb, bars_det, maps=maps)
        pairs = pair_maps_to_bars(image_rgb, [m["xyxy"] for m in maps],
                                  [{"box": b["box"], "strip": b.get("strip"), "orientation": b.get("orientation")} for b in bars],
                                  min_compat=self.min_compat, mode=self.pairing_mode)
        H, W = image_rgb.shape[:2]
        panels = []
        for m, p in zip(maps, pairs):
            x0, y0, x1, y1 = m["xyxy"]
            tx, ty = (x1 - x0) * self.margin_trim, (y1 - y0) * self.margin_trim
            mx0, my0 = int(max(0, x0 + tx)), int(max(0, y0 + ty))
            mx1, my1 = int(min(W, x1 - tx)), int(min(H, y1 - ty))
            panel = {"map_box": m["xyxy"], "map_conf": m["conf"], "bar_idx": p["bar_idx"],
                     "compat": p["compat"], "bar_distance": p["distance"], "pairing_evidence": p.get("evidence"),
                     "status": None, "value_map": None, "valid_mask": None, "qc": {}}
            if p["bar_idx"] is None:
                any_readable = any(b.get("ok") for b in bars)
                panel["status"] = (p.get("reason") or ("no_compatible_bar" if any_readable else
                                   ("no_bar_detected" if not bars else "bar_unreadable")))
                panel["qc"] = {"pairing_evidence": p.get("evidence")}
                panels.append(panel)
                continue
            bar = bars[p["bar_idx"]]
            if not bar.get("ok"):
                panel["status"] = "bar_unreadable"
                panel["qc"] = {"bar_reason": bar.get("reason")}
                panels.append(panel)
                continue
            crop = image_rgb[my0:my1, mx0:mx1]
            vm, valid, qc = quantify_map(crop, bar["strip"], bar["vmin"], bar["vmax"], **self.engine_kw)
            qc.update({"vmin": bar["vmin"], "vmax": bar["vmax"], "unit": bar.get("unit"),
                       "n_anchors": bar.get("n_anchors"), "ocr_conf_min": bar["fit"].get("ocr_conf_min"),
                       "fit_residual_rel": bar["fit"].get("residual_rel"),
                       "strip_tortuosity": bar["strip_quality"]["tortuosity"],
                       "bar_orientation": bar["orientation"]})
            panel.update({"status": "ok" if qc["valid_pixel_ratio"] >= 0.5 else "low_valid_ratio",
                          "value_map": vm, "valid_mask": valid, "qc": qc,
                          "map_crop_box": (mx0, my0, mx1, my1)})
            panels.append(panel)
        return {"figure_id": figure_id, "image_shape": (H, W), "maps": maps, "bars": bars,
                "pairs": pairs, "panels": panels}
