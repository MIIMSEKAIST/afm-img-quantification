# AFM detection (YOLO, two-step)

A two-step YOLO detector locates AFM images within composite figures, then
separates the map region from the color bar:

1. **Step 1** — detect AFM image regions within full figures.
2. **Step 2** — separate map vs. color bar within each detected AFM image.

| script | role | migrate from |
|--------|------|--------------|
| `train.py`  | train the detector | `scripts/train_yolo.py` |
| `detect.py` | run inference on figures | `scripts/run_detection.py` |
| `data.yaml` | YOLO dataset config | `datasets/afm_detection/data.yaml` |

Weights (`weights/afm_detect_best.pt`) are distributed as a GitHub Release asset
— see `weights/README.md`. STATUS: scripts to be migrated.
