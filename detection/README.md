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
| `train.py`    | train the detector from the pretrained backbone (defaults = configuration of run v13, the submitted detector) |
| `../scripts/corpus/build_dataset_v2.py`, `finetune_v15.py` | training set v2 (figures of the first corpus sample) and the fine-tuning that produced the released detector v15 |
| `../scripts/corpus/compare_detectors_heldout.py` | development check: submitted vs released detector against the first-pass counts on figures of the first corpus sample not used for training |
| `evaluate.py` | held-out metrics (Ultralytics `val`) -> `results/detection_eval.json` |
| `detect.py`   | run inference on figures, crop maps and colour bars |
| `data.yaml`   | YOLO dataset configuration |

Weights (`weights/afm_detect_best.pt` = released detector v15; `weights/afm_detect_v13.pt`
= submitted detector) are distributed as GitHub Release assets — see `weights/README.md`.

## Training data

188 manually annotated figures with 923 boxes (504 map panels, 419 colour bars);
47 figures contain no colour-mapped panel and serve as negatives. Split 131 train /
38 validation / 19 test, disjoint by image-content hash (no duplicates across
splits). 57 figures come from the open-access literature corpus described in
`data/README.md`; 131 were collected earlier from AFM publications. The
annotations of the test split are part of the Zenodo deposit; the figure images
themselves are copyrighted and are not redistributed.

## Training configuration (submitted detector, `runs/afm_detect/v13`)

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

## Released detector (v15, revision R1)

A first-pass rating of the first 1,000-figure corpus sample, whose verdicts were produced by an AI
coding assistant (manuscript Methods), indicated that the submitted detector (v13, the initial
detector of the article) produced boxes on many map-free figures (photographs, schematics) and
missed colour bars. The released detector is v13 fine-tuned on training set v2: the original labels
plus 91 figures of that sample in which, by that rating, the detector had found every map and
colour bar and returned no false box (their detections serve as labels) and 198 map-free figures
with empty labels (158 on which v13 had produced boxes, 40 drawn at random). The validation and
test splits are unchanged, and the later corpus samples were never used.
Fine-tuning: 60 epochs from the v13 weights, lr0 0.002, warm-up 1 epoch, otherwise the
configuration above (`scripts/corpus/finetune_v15.py`). The pipeline applies v15 at
confidence 0.25 (0.5 for v13): the additional low-confidence boxes are rejected
downstream unless a readable, legitimately paired colour bar is found.

| split | mAP@50 | mAP@50–95 | precision | recall |
|-------|--------|-----------|-----------|--------|
| validation (38 images) | 0.950 | 0.823 | 0.924 | 0.936 |
| test (19 images) | 0.956 | 0.842 | 0.901 | 0.936 |

On literature figures the two detectors were compared on 200 figures drawn at random from the
held-out fourth corpus sample (manuscript Section 2.8): two of the authors counted the
colour-mapped panels (212 in 55 figures) and, without knowing which detector had drawn which
boxes, the boxes that enclose a panel. v15 at 0.25 found 161 panels (recall 0.76) with 60 false
boxes (precision 0.73); v13 at 0.5 found 155 (recall 0.73) with 123 false boxes (precision 0.56).
In the 145 figures without maps v15 returned boxes in 24 and v13 in 37. The hand-labelled test
split is small (19 images): there v15 scores mAP@50 0.956 against 0.983 for v13.
(`compare_detectors_heldout.py` is an earlier development check against the counts of the
first-pass rating; the article does not use it.)

Behaviour on realistic figures (468 rendered figures, 15 variants), on four samples of the
literature corpus and on 100 figures of other colour-mapped techniques is reported in the
manuscript (Sections 2.6 and 2.8 and Supplementary Note 7).
Known limitation: horizontal colour bars are rarely quantified by the released pipeline (4 of 26 in the realistic-figure benchmark); the column of y-axis tick labels is sometimes
returned as a colour bar and is rejected by the colour-bar reader.
