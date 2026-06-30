# Detection weights

Trained YOLO weights are **not stored in git** (they are large binaries). They
are attached as **GitHub Release assets**.

| file | description |
|------|-------------|
| `afm_detect_best.pt` | final trained two-step AFM / color-bar detector (run v13) |

Base pretrained backbone: Ultralytics `yolo11m.pt` (downloaded automatically by
`ultralytics` on first use, or available from the Ultralytics repository).

## Download

Place `afm_detect_best.pt` in this directory before running `detection/detect.py`:

```
detection/weights/afm_detect_best.pt
```

- **Release URL:** _TBD — to be attached before submission_
