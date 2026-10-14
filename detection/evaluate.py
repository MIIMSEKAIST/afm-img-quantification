"""Evaluate the released detector on the labelled validation and test splits.

Reproduces the held-out metrics quoted in detection/README.md with
``ultralytics`` ``model.val`` (box mAP@0.5, mAP@0.5:0.95, precision, recall,
per class) and writes them to results/detection_eval.json.

Usage
-----
    python detection/evaluate.py --weights detection/weights/afm_detect_best.pt \
        --data detection/data.yaml --out results/detection_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("detection/weights/afm_detect_best.pt"))
    ap.add_argument("--data", type=Path, default=Path("detection/data.yaml"))
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--out", type=Path, default=Path("results/detection_eval.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO  # lazy import
    model = YOLO(str(args.weights))
    report = {"weights": str(args.weights), "data": str(args.data), "splits": {}}
    for split in args.splits:
        r = model.val(data=str(args.data.resolve()), split=split, verbose=False, plots=False)
        names = r.names
        per_class = {}
        for i, cid in enumerate(r.box.ap_class_index.tolist()):
            per_class[names[cid]] = {"AP50": float(r.box.ap50[i]), "AP50_95": float(r.box.ap[i]),
                                     "precision": float(r.box.p[i]), "recall": float(r.box.r[i])}
        report["splits"][split] = {
            "n_images": int(r.seen) if hasattr(r, "seen") else None,
            "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
            "precision": float(r.box.mp), "recall": float(r.box.mr),
            "per_class": per_class,
        }
        print(f"[{split}] mAP50={r.box.map50:.3f} mAP50-95={r.box.map:.3f} "
              f"P={r.box.mp:.3f} R={r.box.mr:.3f}  per-class: "
              + ", ".join(f"{k}: AP50 {v['AP50']:.3f}" for k, v in per_class.items()))
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
