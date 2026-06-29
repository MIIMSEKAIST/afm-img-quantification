# Figure generation

Scripts that render the manuscript figures from the ground-truth IBW (via the
validated `afmquant` engine) and the precomputed results. Presentation-only — no
new computation that affects the reported metrics. Statistics live in
`../analysis/`, not here.

| script | produces | covers |
|--------|----------|--------|
| `make_figure_parts.py` | individual figure "parts" (one PNG per panel) under `figures/parts/{fig3,fig4,fig5,figS4}/` | Fig. 3, 4, 5 and Fig. S4 |
| `make_supplementary.py` | `Table S1` (xlsx) + `Fig. S1` (rendering example) + `Fig. S2` (nMAE by material × colormap) under `figures/parts/supp/` | SI |

Parts are saved with minimal margins for assembly (panel labels a/b/c and layout
are added in a vector editor).

## Run

```bash
IBW_DIR=data/esm_ibw python figures/make_figure_parts.py
IBW_DIR=data/esm_ibw python figures/make_supplementary.py
```

`IBW_DIR` defaults to `data/esm_ibw`; both scripts read the ground-truth scans
(see `../data/README.md`). Fig. 5 also reads `../results/meta_scanlevel.csv` if
present (otherwise regenerates it from the IBW). Outputs under `figures/parts/`
are git-ignored (regenerable).
