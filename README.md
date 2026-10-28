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
| **Engine benchmark** (26 ESM scans × 4 colormaps, JPEG q85) | lookup only, exact scale given | nMAE **2.09 ± 0.76 %** of range, SSIM 0.81, scan median within 0.46 % |
| **Colour-bar reading** (300 synthetic colour bars) | OCR + scale fit | scale read in 99.3 %, both end values within 2 % in **98.3 %** (median end error 0.21 %) |
| **Realistic figures** (468 complete figures, 15 rendering variants) | detector → OCR → pairing → lookup | default variant: 100 % of panels quantified, scale error 0.40 %, scan median within 0.36 %, nMAE **2.8 %** |
| **Detector** (hand-labelled test split; verified corpus figures) | map / colour-bar boxes | test split mAP@50 0.96 (submitted detector 0.98); on 711 verified corpus figures not used for training, the released detector finds 197 colour bars against 146 for the submitted one and returns the exact number of map panels in 51 figures against 26; false boxes on 4.3 % of the map-free figures (manuscript Section 2.1) |

Known limitations: horizontal colour bars are rarely quantified by the released pipeline (4 of 26 in the realistic-figure benchmark); only linear colour scales with at least two legible
numeric labels are read (logarithmic scales are flagged and skipped); a panel whose
colour bar cannot be identified from the figure layout, or whose own colour bar is
not readable, is withheld rather than quantified with a neighbour's bar (status
`no_exclusive_bar` / `own_bar_unreadable`); on real figures the remaining scale errors
are mostly labels of a neighbouring axis taken for the bar's, lost minus signs and
saturated white bar ends that no frame line bounds; the detector recognises colour-mapped
panels of any technique, so AFM specificity comes from the caption / unit filter.
The accuracy of the pipeline on real literature figures (human verification of two
1,000-figure samples, before and after the revision) is reported in the manuscript
(Section 2.8) and reproduced by `scripts/corpus/`.

---

## Repository layout

```
afmquant/            importable package
  engine.py            colour-bar-guided nearest-colour lookup (CIELAB or sRGB matching, ΔE validity mask)
  colorbar.py          colour-bar crop -> gradient region -> 256-sample calibration strip
  ocr.py               tick-label OCR (RapidOCR / PP-OCRv4), label parsing, unit lexicon, scale fitting (submitted reader)
  reader2.py           label-column colour-bar reader (released): geometry-based label selection, verified strip ends,
                       ink-based sign decision, unit placement rules, laid-out-bar candidates
                       ink-based sign decision, unit placement rules, laid-out-bar candidates
                       ink-based sign decision, unit placement rules, laid-out-bar candidates
                       ink-based sign decision, unit placement rules, laid-out-bar candidates
                       ink-based sign decision, unit placement rules, laid-out-bar candidates
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
scripts/corpus/      corpus-sample evaluations: held-out sample runs, human-verification workbooks, reader / pairing
                     evaluations, detector training set v2 and fine-tuning, held-out detector comparison (see its README)
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
- **Corpus-sample verification** (sample manifests with DOIs, pipeline results, the
  human-verification workbooks of both samples, the colour-bar truth table and the
  labels added to detector training set v2) is part of the Zenodo deposit; the figure
  images are copyrighted and not redistributed (see `scripts/corpus/README.md`).

## ⚠️ Note on copyright (literature corpus)

The `collection/` pipeline reproduces how the literature figure corpus was
assembled, but **the collected PDFs and extracted figures are not redistributed**:
they are copyrighted by their publishers. The corpus-sample annotations released
with the paper contain figure identifiers, captions and human judgements only.

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
