"""Train the YOLO AFM / color-bar detector.

Fine-tunes an Ultralytics YOLO model on the labelled AFM-detection dataset
(classes: afm_map, colorbar) defined by data.yaml.

Usage
-----
    python detection/train.py --data detection/data.yaml --model yolo11m.pt \
        --epochs 150 --imgsz 640 --batch 8 --name afm_detect

Source: migrated from scripts/train_yolo.py (paths parameterized).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("detection/data.yaml"))
    ap.add_argument("--model", default="yolo11m.pt", help="base/pretrained weights")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--project", default="runs/afm_detect")
    ap.add_argument("--name", default="afm_detect")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported lazily so the repo imports without it

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, save=True,
        project=args.project, name=args.name, workers=0,
        # augmentation
        hsv_h=0.02, hsv_s=0.4, hsv_v=0.3,
        flipud=0.3, fliplr=0.5, degrees=5.0,
        scale=0.3, translate=0.1, mosaic=1.0, mixup=0.1,
    )
    print(f"Training complete. Best model under {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
