# Changelog

## v1.1.0 — revision R1 (2026-10)

Added
- `afmquant/reader2.py`: label-column colour-bar reader (`reader="v2"`, the default): tick labels are
  selected by their geometry (the column of numbers beside the strip, numbers inside maps or nearer to
  another bar discarded), the strip ends are verified against the label positions and light ends are
  recovered, and a lost sign or decimal point on a three-label bar is repaired from linearity.
- `afmquant/pairing.py`: evidence-based pairing (`pairing_mode="evidence"`, the default): a map is paired
  only with a colour bar that the figure layout assigns to it (exclusive, row or column evidence); otherwise
  the panel is held (`no_exclusive_bar`, `own_bar_unreadable`) instead of being quantified with a
  neighbour's bar. `afmquant/ocr.py`: lexicon-constrained units (`unit_mode="lexicon"`, the default).
- Released detector v15: the submitted detector fine-tuned on the human-verified corpus figures (dataset v2 =
  original labels + 91 verified positives + 198 map-free negatives), applied at confidence 0.25.
- `scripts/corpus/`: corpus-sample evaluations (held-out sample runs, verification workbooks, reader and
  pairing evaluations on verified figures, dataset v2, fine-tuning, held-out detector comparison);
  `data/corpus_sample/bar_truth.csv`.
- `--conf`, `--reader`, `--unit-mode`, `--pairing-mode` on `scripts/run_figure.py`, `benchmark/tier2_evaluate.py`
  and `benchmark/tier3_evaluate.py`; `--reader` on `benchmark/ocr_synthetic_test.py`.
- `results/detection_eval_v13.json` (submitted detector) next to `results/detection_eval.json` (released).
- `afmquant/colorbar.py`, `afmquant/ocr.py`, `afmquant/pairing.py`, `afmquant/pipeline.py`:
  colour-bar strip extraction, tick-label OCR (RapidOCR / PP-OCRv4) and scale fitting,
  map–colour-bar pairing and the end-to-end `FigurePipeline` (these stages were described
  in the submitted manuscript but not released).
- `afmquant/afm_filter.py`: caption / unit based technique filter.
- `benchmark/ocr_synthetic_test.py`: colour-bar reading test (300 synthetic colour bars).
- `benchmark/tier2_render.py`, `benchmark/tier2_evaluate.py`: realistic-figure benchmark
  (468 complete figures in 15 rendering variants, including two-map figures that test the
  map–colour-bar pairing and a PDF route).
- `benchmark/subsampling_sweep.py`, `benchmark/baseline_comparison.py`,
  `benchmark/colormap_analysis.py`: JPEG chroma-subsampling decomposition, lookup
  baselines (sRGB vs CIELAB, projection, ideal strip) and colormap descriptors.
- `detection/evaluate.py`: held-out detector metrics; `scripts/run_figure.py`: CLI.
- `benchmark/tier3_evaluate.py`, `benchmark/tier3_match.py`: evaluation on previously published
  figures with archived raw scans (figure extraction from the publisher PDF as for the corpus,
  full pipeline, automatic matching of each panel to the raw scan, channel and orientation it
  shows, Tier 2 metrics); `benchmark/tier3_selftest.py`: synthetic self-test of that tool.
- `reproduce_all.py`: one command regenerates every number of the manuscript
  (`results/numbers_manifest.json`).
- `tests/`: 49 unit tests; `requirements.lock`: exact versions.

Changed
- Defaults of `FigurePipeline` and of every script are the released configuration (detector v15 at
  confidence 0.25, reader v2, lexicon units, evidence pairing); the submitted configuration is
  `conf=0.5, reader="v1", unit_mode="raw", pairing_mode="nearest"` with `afm_detect_v13.pt`.
- `afmquant/engine.py`: `color_space` ("cielab" | "rgb"), `interpolate` and `mask_space`
  options; the validity mask is always a CIELAB ΔE.
- `afmquant/metrics.py`: additional metrics (nMAE/IQR, nMAE/SD, signed mean / SD error,
  all-pixel nMAE, levels in IQR).
- `afmquant/render.py`: explicit JPEG chroma-subsampling control; matplotlib ≥ 3.9 colormap API.
- `detection/train.py`: defaults synchronised with the released training run (v13).
- `detection/detect.py`, `scripts/run_figure.py`: no double processing of `*.png` / `*.PNG` on Windows.
- `data/README.md`: corrected corpus inventory.

Fixed
- Reported scan-median deviation and Table 1 values now match the code output exactly.

## v1.0.0 — submission (2026-07)

Initial release: quantification engine, engine benchmark, robustness sweep, meta-analysis.
