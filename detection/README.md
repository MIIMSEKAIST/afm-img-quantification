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
— see `weights/README.md`. The published models were trained with the two-step
protocol and hyperparameters described in the manuscript Methods.
