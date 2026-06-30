# AFM detection (YOLO, two-step)

A two-step YOLO detector locates AFM images within composite figures, then
separates the map region from the color bar:

1. **Step 1** — detect AFM image regions within full figures.
2. **Step 2** — separate map vs. color bar within each detected AFM image.

| script | role |
|--------|------|
| `train.py`  | train the detector |
| `detect.py` | run inference on figures, crop AFM maps + color bars |
| `data.yaml` | YOLO dataset config |

Weights (`weights/afm_detect_best.pt`) are distributed as a GitHub Release asset
— see `weights/README.md`.

## Released model & evaluation

The released weight `weights/afm_detect_best.pt` is the **Step-2 model**
(map vs. color-bar separation; classes `afm_map`, `colorbar`), a YOLO11m detector.

**Training (this checkpoint):** 100 epochs, batch 8, image size 640, lr0 0.01,
Ultralytics online augmentation (hsv_s 0.3, rotation ±5°, scale 0.3, fliplr 0.5,
mosaic 1.0). Dataset: 131 train / 38 val / 19 test images, disjoint across splits
(verified by image-content hashing — no leakage).

**Held-out performance** (Ultralytics `model.val`, deterministic across re-runs):

| split | mAP@50 | mAP@50–95 | precision | recall |
|-------|--------|-----------|-----------|--------|
| validation (38 images) | 0.991 | 0.905 | 0.963 | 0.972 |
| test (19 images) | 0.983 | 0.910 | 0.951 | 0.974 |

Reproduce: load the weight and run
`YOLO(weight).val(data=detection/data.yaml, split='test')`.
