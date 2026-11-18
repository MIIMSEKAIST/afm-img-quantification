"""Detector v15: fine-tune the submitted detector (v13, YOLO11m) on dataset v2 (original labelled set + figures of the
first corpus sample chosen with the first-pass rating: positive box sets and hard negatives), then evaluate on the unchanged held-out splits.

    python train_detector_v15.py [--epochs 60]

Outputs: runs/detect/runs/afm_detect/v15/weights/best.pt, revision/detection/v15/detection_eval_v15.json (held-out
valid/test metrics with detection/evaluate.py) and the same for v13 for comparison."""
import argparse, json, subprocess, sys
from pathlib import Path

from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
RUNS = ROOT / "runs" / "detect" / "runs" / "afm_detect"
DST = ROOT / "datasets" / "afm_detection_v2"
OUT = ROOT / "revision" / "detection" / "v15"
PY = sys.executable


def train(epochs):
    from ultralytics import YOLO
    for batch in (8, 4):
        try:
            model = YOLO(str(RUNS / "v13" / "weights" / "best.pt"))
            model.train(data=str(DST / "data.yaml"), epochs=epochs, imgsz=640, batch=batch, patience=15, seed=0,
                        project=str(RUNS), name="v15", exist_ok=True, workers=0, save=True, lr0=0.002, warmup_epochs=1.0,
                        hsv_h=0.015, hsv_s=0.3, hsv_v=0.2, flipud=0.3, fliplr=0.5, degrees=5.0,
                        scale=0.3, translate=0.1, mosaic=1.0, mixup=0.0, verbose=False)
            return batch
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or batch == 4:
                raise
            print("OOM at batch 8 -> retry with batch 4", flush=True)
            import torch
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    batch = train(a.epochs)
    report = {"train_batch": batch, "epochs": a.epochs, "dataset": str(DST)}
    for name in ("v13", "v15"):
        w = RUNS / name / "weights" / "best.pt"
        out = OUT / f"detection_eval_{name}.json"
        with open(OUT / f"eval_{name}.log", "w", encoding="utf-8") as f:
            subprocess.run([PY, "-X", "utf8", "-W", "ignore", "detection/evaluate.py", "--weights", str(w),
                            "--data", str(ROOT / "datasets" / "afm_detection" / "data.yaml"), "--out", str(out)],
                           cwd=REPO, check=True, stdout=f, stderr=subprocess.STDOUT)
        report[name] = json.loads(out.read_text(encoding="utf-8")).get("splits")
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
