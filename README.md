# afm-img-quantification

Color-bar-guided reconstruction of quantitative Atomic Force Microscopy (AFM)
data from published figures — the code and data accompanying:

> **Quantitative Reconstruction of Atomic Force Microscopy Data from Published
> Figures: A Validated Machine Learning Pipeline**
> S. Ryu, D. Chen, B. Madika, S. Hong (KAIST), *Digital Discovery*, 2026.

A large amount of quantitative AFM data is locked inside published figures as
color-rendered images. This pipeline detects AFM maps within composite figures,
separates the map and color-bar regions, and reconstructs the underlying
numerical values through color-bar-guided quantification in CIELAB perceptual
color space.

The **validated core** is the quantification engine (`afmquant/engine.py`),
benchmarked against 104 ground-truth cases (26 in-house Electrochemical Strain
Microscopy scans × 4 colormaps, JPEG quality 85):

| metric | result |
|--------|--------|
| nMAE (range-normalized MAE) | **2.09 ± 0.76 %** |
| SSIM | **0.807 ± 0.105** |
| scan-median amplitude recovery | within **0.48 %** |

---

## Repository layout

```
afmquant/        # ★ validated core (importable package)
  engine.py        color-bar-guided CIELAB + KD-tree quantification
  render.py        publication-style rendering (benchmark figure generation)
  metrics.py       nMAE / SSIM / EMD / evaluate()
benchmark/       # ★ ground-truth validation reproduction
  reproduce_benchmark.py     26 IBW × 4 colormaps = 104 cases -> results/benchmark104.csv
  robustness_sweep.py        JPEG-quality & resolution sweep (1040 cases)
  diffusivity_consistency.py reconstruction-to-diffusivity consistency check
analysis/        # meta-analysis: rankings, Cliff's delta, Mann-Whitney
figures/         # manuscript figure generation (Fig. 3-5, Fig. S4)
detection/       # YOLO two-step AFM / color-bar detection (weights via Releases)
collection/      # literature collection pipeline (EXPERIMENTAL; see note below)
data/            # ground-truth IBW scans (via Zenodo) + data provenance
results/         # precomputed metric CSVs (regenerable)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

Python ≥ 3.10 recommended.

## Quick start — reproduce the validation benchmark

1. Download the ground-truth ESM scans from Zenodo (see `data/README.md`) into
   `data/esm_ibw/`.
2. Run:

```bash
python benchmark/reproduce_benchmark.py --ibw-dir data/esm_ibw --out results/benchmark104.csv
```

This regenerates `results/benchmark104.csv` and prints the summary table
(nMAE / SSIM overall and per colormap).

## Use the engine on your own figure

```python
from afmquant import quantify_map
value_map, valid_mask, qc = quantify_map(map_rgb, colorbar_strip_rgb, vmin, vmax)
```

`map_rgb` is the rendered map (H×W×3 uint8), `colorbar_strip_rgb` is the color
bar sampled along its axis (vmax→vmin), and `vmin`/`vmax` are the scale endpoints
(e.g. from OCR of the color-bar labels).

## Data and weights

- **Ground-truth ESM data** (26 IBW scans) — archived on Zenodo with a DOI; see
  `data/README.md`. Not stored in git.
- **Trained detection weights** (`best.pt`) — attached as a GitHub Release asset;
  see `detection/weights/README.md`. Not stored in git.

## ⚠️ Note on copyright (literature corpus)

The `collection/` pipeline reproduces how the literature figure corpus is
assembled, but **the collected PDFs and extracted figures are NOT redistributed
here**: they are copyrighted by their original publishers. Only the collection
*code* is provided. Large-scale detection over the full corpus is ongoing work;
the validated contribution of the paper is the quantification engine and the
ground-truth benchmark, both fully reproducible from the in-house data above.

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
