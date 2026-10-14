# Detection of colour-mapped panels and colour bars (YOLO11m)

A single-stage YOLO11m detector is applied to the whole figure and returns two
classes: `afm_map` (a colour-mapped image panel) and `colorbar`. The detector
recognises colour-mapped panels accompanied by a colour bar regardless of the
measurement technique; AFM specificity is established separately by the
caption / unit filter (`afmquant/afm_filter.py`).

> The submitted version of the manuscript described a two-stage prototype
> (figure-level AFM-region detection followed by map / colour-bar separation).
> The first-stage model of that prototype is not part of the released pipeline;
> everything reported in the revised manuscript refers to the single-stage
> detector documented here.

| script | role |
|--------|------|
| `train.py`    | train the detector (defaults = configuration of the released run) |
| `evaluate.py` | held-out metrics (Ultralytics `val`) -> `results/detection_eval.json` |
| `detect.py`   | run inference on figures, crop maps and colour bars |
| `data.yaml`   | YOLO dataset configuration |

Weights (`weights/afm_detect_best.pt`) are distributed as a GitHub Release asset
— see `weights/README.md`.

## Training data

188 manually annotated figures with 923 boxes (504 map panels, 419 colour bars);
47 figures contain no colour-mapped panel and serve as negatives. Split 131 train /
38 validation / 19 test, disjoint by image-content hash (no duplicates across
splits). 57 figures come from the open-access literature corpus described in
`data/README.md`; 131 were collected earlier from AFM publications. The
annotations of the test split are part of the Zenodo deposit; the figure images
themselves are copyrighted and are not redistributed.

## Training configuration (released run, `runs/afm_detect/v13`)

Ultralytics 8.4, pretrained `yolo11m.pt`; 100 epochs, batch 8, image size 640,
lr0 0.01, patience 20, augmentation hsv_h 0.015 / hsv_s 0.3 / hsv_v 0.2,
rotation ±5°, scale 0.3, translate 0.1, fliplr 0.5, flipud 0.3, mosaic 1.0,
mixup 0. `train.py` uses exactly these defaults.

## Held-out performance

Computed with `evaluate.py` (Ultralytics `model.val`, deterministic across re-runs):

| split | mAP@50 | mAP@50–95 | precision | recall |
|-------|--------|-----------|-----------|--------|
| validation (38 images) | 0.991 | 0.905 | 0.963 | 0.972 |
| test (19 images) | 0.983 | 0.910 | 0.951 | 0.974 |

Per class on the test split: `afm_map` AP@50 0.995, `colorbar` AP@50 0.971.

Behaviour on realistic figures (416 rendered figures, 13 variants) and on a
random sample of 1,000 corpus figures and 100 figures with non-AFM colour-mapped
panels is reported in the manuscript (Section 2.5–2.6 and Supplementary Note 7).
Known limitation: horizontal colour bars are not detected (0 of 26 in the
realistic-figure benchmark); the column of y-axis tick labels is sometimes
returned as a colour bar and is rejected by the colour-bar reader.
