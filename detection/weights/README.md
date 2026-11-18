# Detection weights

Trained YOLO weights are **not stored in git** (they are large binaries). They
are attached as **GitHub Release assets** of release v1.1.1 (the same files as in v1.1.0).

| file | description |
|------|-------------|
| `afm_detect_best.pt` | **Released detector (run v15)**: YOLO11m, the submitted detector fine-tuned on figures of the first corpus sample (training set v2); applied at confidence 0.25. Test split mAP@50 0.96; see `../README.md`. |
| `afm_detect_v13.pt` | Submitted detector (run v13): YOLO11m trained on the hand-labelled set; applied at confidence 0.5. Test split mAP@50 0.98 (mAP@50–95 0.91). |

Base pretrained backbone: Ultralytics `yolo11m.pt` (downloaded automatically by
`ultralytics` on first use, or available from the Ultralytics repository).

## Download

Download both files from the GitHub Release and save them in this directory
under the names above (`afm_detect_best.pt` is the default of every script;
`afm_detect_v13.pt` reproduces the submitted version together with
`--conf 0.5 --reader v1 --unit-mode raw --pairing-mode nearest`):

```
detection/weights/afm_detect_best.pt
detection/weights/afm_detect_v13.pt
```

- **Release:** https://github.com/MIIMSEKAIST/afm-img-quantification/releases/tag/v1.1.1
- **Direct download:** https://github.com/MIIMSEKAIST/afm-img-quantification/releases/download/v1.1.1/afm_detect_best.pt
  and https://github.com/MIIMSEKAIST/afm-img-quantification/releases/download/v1.1.1/afm_detect_v13.pt
- The v1.0.0 release asset `best.pt` is the same file as `afm_detect_v13.pt`.
