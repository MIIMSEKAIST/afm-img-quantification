# Ground-truth data

The validation benchmark is built from **26 in-house DART-ESM measurements**
(Igor Binary Wave `.ibw`, acquired on a Cypher ES AFM, Oxford Instruments)
spanning three lithium-ion conductors:

| material | scans |
|----------|-------|
| LMNO  (LiMn1.5Ni0.5O4)            | 9  |
| LICGC (Li1+xAlxGe2-x(PO4)3)       | 10 |
| NCM811 (LiNi0.8Co0.1Mn0.1O2)      | 7  |

For each scan the DART ESM amplitude channel serves as the ground-truth matrix.

## Download

These raw scans are archived on Zenodo with a DOI (open data, required by
*Digital Discovery*). Download the archive and place the `.ibw` files in this
directory:

```
data/esm_ibw/
    LICGC_01_4um.ibw
    ...
    NCM811_xx.ibw
```

- **Zenodo DOI:** _TBD — to be minted before submission_

## Not included (copyright)

The literature figure corpus used for the detection demonstrations
(4,853 source PDFs and ~39,000 extracted figures) is **not** redistributed: it
is copyrighted by the original publishers. The `collection/` scripts show how it
is assembled from public metadata (Crossref) and open tools.
