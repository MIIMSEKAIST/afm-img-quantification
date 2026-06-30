# Detection weights

Trained YOLO weights are **not stored in git** (they are large binaries). They
are attached as **GitHub Release assets**.

| file | description |
|------|-------------|
| `afm_detect_best.pt` | Step-2 map/color-bar detector, YOLO11m (run v13). Held-out test mAP@50 0.98 (mAP@50–95 0.91); see `../README.md`. |

Base pretrained backbone: Ultralytics `yolo11m.pt` (downloaded automatically by
`ultralytics` on first use, or available from the Ultralytics repository).

## Download

Download `best.pt` from the GitHub Release and save it in this directory as
`afm_detect_best.pt` (the filename `detection/detect.py` expects):

```
detection/weights/afm_detect_best.pt
```

- **Release:** https://github.com/MIIMSEKAIST/afm-img-quantification/releases/tag/v1.0.0
- **Direct download:** https://github.com/MIIMSEKAIST/afm-img-quantification/releases/download/v1.0.0/best.pt
