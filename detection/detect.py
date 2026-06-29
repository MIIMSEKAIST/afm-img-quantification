"""Run the trained detector on figures and crop AFM maps + color bars.

For each input figure image, detects afm_map and colorbar regions, removes
duplicate boxes (greedy IoU NMS per class), crops each map, pairs it with the
nearest color bar, and writes the crops plus a detections table.

The crops feed the quantification engine (`afmquant`).

Usage
-----
    python detection/detect.py --images path/to/figures \
        --weights detection/weights/afm_detect_best.pt --out out/afm_panels

`--images` may be a directory (globbed for *.png/*.jpg) or a single image.

Source: migrated from scripts/run_detection.py. The original read a
figures.parquet produced by the (copyright-restricted) collection pipeline; here
the input is a plain image directory so the detector is usable standalone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

CLASSES = ["afm_map", "colorbar"]


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def merge_overlapping_boxes(boxes, iou_threshold=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes
                 if b["cls"] != best["cls"]
                 or compute_iou(best["xyxy"], b["xyxy"]) < iou_threshold]
    return kept


def iter_images(images: Path):
    if images.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
            yield from sorted(images.glob(ext))
    else:
        yield images


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("detection/weights/afm_detect_best.pt"))
    ap.add_argument("--out", type=Path, default=Path("out/afm_panels"))
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--margin-trim", type=float, default=0.005)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO  # lazy import
    model = YOLO(str(args.weights))

    detections, pair_count = [], 0
    for img_path in iter_images(args.images):
        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        boxes = []
        for r in results:
            for i in range(len(r.boxes)):
                boxes.append({
                    "xyxy": r.boxes.xyxy[i].cpu().numpy().tolist(),
                    "conf": float(r.boxes.conf[i]),
                    "cls": int(r.boxes.cls[i]),
                })
        cleaned = merge_overlapping_boxes(boxes, iou_threshold=0.3)
        maps = [b for b in cleaned if b["cls"] == 0]
        colorbars = [b for b in cleaned if b["cls"] == 1]
        if not maps:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for m in maps:
            x1, y1, x2, y2 = map(int, m["xyxy"])
            bw, bh = x2 - x1, y2 - y1
            tx, ty = int(bw * args.margin_trim), int(bh * args.margin_trim)
            x1, y1 = max(0, x1 + tx), max(0, y1 + ty)
            x2, y2 = min(w, x2 - tx), min(h, y2 - ty)
            map_crop = img[y1:y2, x1:x2]
            if map_crop.size == 0:
                continue

            det_id = f"DET{pair_count:06d}"
            map_path = args.out / f"{det_id}_map.png"
            cv2.imwrite(str(map_path), map_crop)

            # nearest color bar by edge-to-edge distance
            cb_path, cb_conf, best_dist, best_cb = "", 0.0, float("inf"), None
            for cb in colorbars:
                cx1, cy1, cx2, cy2 = cb["xyxy"]
                dx = max(0, max(x1 - cx2, cx1 - x2))
                dy = max(0, max(y1 - cy2, cy1 - y2))
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_cb = dist, cb
            if colorbars and best_dist < max(w, h) * 0.4:
                cx1, cy1, cx2, cy2 = map(int, best_cb["xyxy"])
                cb_crop = img[max(0, cy1):min(h, cy2), max(0, cx1):min(w, cx2)]
                if cb_crop.size > 0:
                    cb_path = str(args.out / f"{det_id}_colorbar.png")
                    cv2.imwrite(cb_path, cb_crop)
                    cb_conf = best_cb["conf"]

            detections.append({
                "detection_id": det_id, "map_path": str(map_path),
                "map_conf": m["conf"], "colorbar_path": cb_path,
                "colorbar_conf": cb_conf, "has_colorbar": cb_path != "",
                "source_figure": str(img_path),
            })
            pair_count += 1

    det_df = pd.DataFrame(detections)
    det_df.to_parquet(args.out / "detections.parquet", index=False)
    print(f"Total detections: {len(det_df)}  with colorbar: "
          f"{int(det_df['has_colorbar'].sum()) if len(det_df) else 0}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
