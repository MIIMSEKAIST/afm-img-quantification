# Changelog

## v1.1.0 — revision R1 (2026-10)

Added
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
- `benchmark/dart_sho.py`: SHO drive amplitude, resonance frequency, Q and phase from the raw
  channels of Asylum DART images (amplitude ratio and phase difference, independent of the phase
  offset), written as .gsf so that published SHO-processed figures can be compared with raw data.
- `reproduce_all.py`: one command regenerates every number of the manuscript
  (`results/numbers_manifest.json`).
- `tests/`: 42 unit tests; `requirements.lock`: exact versions.

Changed
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
