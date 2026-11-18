# afm-img-quantification

Colour-bar-guided reconstruction of quantitative Atomic Force Microscopy (AFM)
data from published figures — the code and data accompanying:

> **Recovering quantitative atomic force microscopy data from published figures:
> a ground-truth-validated pipeline**
> S. Ryu, D. Chen, B. Madika, S. Hong (KAIST). *Digital Discovery*, under revision (2026).

A large amount of quantitative AFM data is locked inside published figures as
colour-rendered images. This pipeline detects colour-mapped image panels and
their colour bars within composite figures, reads the colour-bar scale (tick
labels, unit, exponent offset) with OCR, pairs each map with its colour bar, and
reconstructs the underlying numerical values by colour-bar-guided nearest-colour
lookup. Every stage is validated against ground truth:

| validation | what is measured | result (release v1.1.0) |
|---|---|---|
| **Engine benchmark** (26 ESM scans × 4 colormaps, JPEG q85) | lookup only, exact scale given | nMAE **1.50 ± 0.51 %** of range, SSIM 0.90, scan median within 0.39 % (CIELAB matching of the submitted version: 2.09 %; sRGB baseline: 1.78 % over all pixels) |
| **Colour-bar reading** (300 synthetic colour bars) | OCR + scale fit | scale read in 99.3 %, both end values within 2 % in **98.3 %** (median end error 0.21 %) |
| **Realistic figures** (468 complete figures, 15 rendering variants) | detector → OCR → pairing → lookup | default variant: 100 % of panels quantified, scale error 0.40 %, scan median within 0.28 %, nMAE **2.4 %** |
| **Detector** (hand-labelled test split; literature figures) | map / colour-bar boxes | test split mAP@50 0.96 (initial, i.e. submitted, detector 0.98); on 200 random figures of the held-out fourth corpus sample (212 colour-mapped panels in 55 figures, counted by two of the authors) the released detector finds 161 panels with 60 false boxes (recall 0.76, precision 0.73), the initial detector 155 with 123 false boxes (0.73, 0.56) (manuscript Section 2.8) |
| **Literature figures** (four samples of an open-access corpus, 3,819 figures; rated by two of the authors) | detector → OCR → pairing on published figures | held-out fourth sample, released pipeline: panels quantified in 87 of 819 figures (10.6 %); pairing and scale both correct in **70 of these 87 figures (80 %)** and in 194 of the 217 quantified panels (89 %); initial pipeline on the first sample: 31 of 88 figures (35 %) (manuscript Section 2.8, Table 2) |

Known limitations: horizontal colour bars are rarely quantified by the released pipeline (4 of 26 in the realistic-figure benchmark); only linear colour scales with at least two legible
numeric labels are read (logarithmic scales are flagged and skipped); a panel whose
colour bar cannot be identified from the figure layout, or whose own colour bar is
not readable, is withheld rather than quantified with a neighbour's bar (status
`no_exclusive_bar` / `own_bar_unreadable`); on real figures the remaining scale errors
are mostly labels of a neighbouring axis taken for the bar's, lost minus signs and
saturated white bar ends that no frame line bounds; the detector recognises colour-mapped
panels of any technique, so AFM specificity comes from the caption / unit filter.
The accuracy of the pipeline on real literature figures was rated by two of the authors on four
samples of the open-access corpus (manuscript Section 2.8, Table 2); the pipeline outputs, the
rating screens and the answers are part of the Zenodo deposit. `scripts/corpus/` holds the sampling
and run scripts and documents the earlier AI-assisted first pass that guided the development.

---

## Repository layout

```
afmquant/            importable package
  engine.py            colour-bar-guided nearest-colour lookup (luma-weighted YCbCr distance; CIELAB or sRGB as
                       options; validity = CIELAB ΔE to the nearest strip colour)
  colorbar.py          colour-bar crop -> gradient region -> 256-sample calibration strip
  ocr.py               tick-label OCR (RapidOCR / PP-OCRv4), label parsing, unit lexicon, scale fitting (submitted reader)
  reader2.py           label-column colour-bar reader (released): geometry-based label selection, verified strip ends,
                       ink-based sign decision, unit placement rules, laid-out-bar candidates
  crop.py              map crop trimmed to the panel border (page margins removed), colour-bar pixels excluded
  pairing.py           map <-> colour-bar pairing by gamut compatibility and layout evidence (ambiguous panels are held)
  pipeline.py          FigurePipeline: whole figure -> detected panels -> value maps + QC record
  afm_filter.py        caption / unit based technique filter (AFM vs other colour-mapped techniques)
  render.py            publication-style rendering of ground-truth matrices (benchmark generation)
  metrics.py           nMAE, nMAE/IQR, nMAE/SD, signed errors, EMD, SSIM, levels-in-IQR
benchmark/
  reproduce_benchmark.py     engine benchmark: 26 IBW × 4 colormaps = 104 cases
  robustness_sweep.py        JPEG-quality and resolution sweep
  subsampling_sweep.py       PNG / JPEG 4:4:4 / JPEG 4:2:0 error decomposition
  baseline_comparison.py     lookup variants (sRGB vs CIELAB, projection, ideal strip)
  colormap_analysis.py       colormap descriptors (CIELAB path length, luminance share)
  ocr_synthetic_test.py      colour-bar reading test on synthetic colour bars
  tier2_render.py            realistic-figure benchmark: figure generation (15 variants, PDF route)
  tier2_evaluate.py          realistic-figure benchmark: end-to-end evaluation
  diffusivity_consistency.py reconstruction-to-diffusivity consistency check
detection/           YOLO11m detector: train.py, evaluate.py, detect.py, data.yaml (weights via Releases)
analysis/            scan-level statistics and meta-analysis (rankings, Cliff's delta, Mann–Whitney)
figures/             manuscript figure parts
scripts/run_figure.py   command-line pipeline for a figure or a folder (JSON + overlay output)
scripts/corpus/      corpus samples: sampling, pipeline runs, first-pass rating workbooks (AI-assisted), reader / pairing
                     evaluations, detector training set v2 and fine-tuning, detector comparison (see its README)
collection/          literature collection pipeline (code only; see copyright note)
tests/               unit tests (python -m pytest tests)
data/                ground-truth scans (via Zenodo) and data provenance
results/             precomputed result tables (regenerated by reproduce_all.py)
reproduce_all.py     regenerates every number of the manuscript -> results/numbers_manifest.{json,md}
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt                          # exact versions used: requirements.lock
```

Python 3.10 or newer. The detector (`ultralytics`, PyTorch) and the OCR engine
(`rapidocr-onnxruntime`; the PP-OCRv4 models are downloaded on first use) are
needed only for the detection, colour-bar reading and realistic-figure stages;
the engine benchmark runs without them.

## Quick start

**Reproduce the engine benchmark** (needs the 26 ground-truth scans from Zenodo in `data/esm_ibw/`):

```bash
python benchmark/reproduce_benchmark.py --ibw-dir data/esm_ibw --out results/benchmark104.csv
```

**Reproduce every number in the manuscript** (engine benchmarks and OCR test ≈ 5 min;
with the detector weights and the labelled detection set also the detector
metrics and the realistic-figure benchmark, ≈ 40 min on a GPU):

```bash
python reproduce_all.py --ibw-dir data/esm_ibw
python reproduce_all.py --ibw-dir data/esm_ibw --weights detection/weights/afm_detect_best.pt \
                        --data path/to/afm_detection/data.yaml
```

The output `results/numbers_manifest.md` lists every quantity reported in the
paper next to the file it was computed from.

**Run the pipeline on your own figure**:

```bash
python scripts/run_figure.py figure.png --weights detection/weights/afm_detect_best.pt --out out/
```

writes `out/figure.json` (detected boxes, read scale and unit, pairing, QC
record and the value map of each quantified panel) and an overlay PNG. The
defaults are the released configuration (fine-tuned detector at confidence 0.25,
label-column colour-bar reader, lexicon-constrained units, evidence-based pairing);
`--weights detection/weights/afm_detect_v13.pt --conf 0.5 --reader v1 --unit-mode raw --pairing-mode nearest`
reproduces the submitted version.
From Python:

```python
from afmquant.pipeline import FigurePipeline
pipe = FigurePipeline("detection/weights/afm_detect_best.pt")
result = pipe.process(image_rgb)            # H x W x 3 uint8
for panel in result["panels"]:
    panel["status"], panel["value_map"], panel["valid_mask"], panel["qc"]
```

**Use the engine alone** when the colour-bar strip and the scale are already known:

```python
from afmquant import quantify_map
value_map, valid_mask, qc = quantify_map(map_rgb, colorbar_strip_rgb, vmin, vmax)
```

`colorbar_strip_rgb` is the colour bar sampled along its axis (vmax → vmin,
N × 3 uint8). `color_space="rgb"` selects Euclidean-sRGB matching (the baseline
of the paper); the ΔE validity mask is always evaluated in CIELAB.

## Data and weights

- **Ground-truth ESM scans** (26 IBW files), the rendered benchmark images, the
  per-case metrics, the realistic-figure benchmark figures and the detection
  test-set annotations are archived on Zenodo (see `data/README.md`). Not stored in git.
- **Detector weights** (`afm_detect_best.pt` = released detector v15, `afm_detect_v13.pt` =
  submitted detector; YOLO11m, training configuration in `detection/README.md`) are
  attached to the GitHub Release. Not stored in git.
- **Corpus samples** (sample manifests with DOIs, the pipeline output of every state on every
  sample, the authors' rating with its item lists, answers and time log, the workbooks of the
  earlier AI-assisted first pass, the colour-bar regression set and the labels added to detector
  training set v2) are part of the Zenodo deposit; the figure images are copyrighted and not
  redistributed (see `scripts/corpus/README.md`).

## ⚠️ Note on copyright (literature corpus)

The `collection/` pipeline reproduces how the literature figure corpus was
assembled, but **the collected PDFs and extracted figures are not redistributed**:
they are copyrighted by their publishers. The corpus-sample files released
with the paper contain figure identifiers, captions, pipeline outputs and ratings, and no figure images.

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
