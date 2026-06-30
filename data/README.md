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
    NCM811_pristine_07_4um.ibw
```

- **Zenodo DOI:** https://doi.org/10.5281/zenodo.21055266 (CC-BY)

## Which files define the benchmark

The 104-case benchmark uses **exactly the 26 scans listed in
`benchmark_scans.txt`** (10 LICGC + 9 LMNO_pristine + 7 NCM811_pristine). All
analysis scripts read that manifest and use only those files, so the reported
counts (26 scans → 104 cases) are reproduced even if extra `.ibw` files are
present in this directory.

The Zenodo deposit additionally contains **3 auxiliary single-scan files**
(`LICGC.ibw`, `LMNO_pristine.ibw`, `NCM811_pristine.ibw`) that are *not* part of
the benchmark; they are provided only as standalone examples and are ignored by
the scripts.

## Not included (copyright)

The literature figure corpus used for the detection demonstrations
(4,853 source PDFs and ~39,000 extracted figures) is **not** redistributed: it
is copyrighted by the original publishers. The `collection/` scripts show how it
is assembled from public metadata (Crossref) and open tools.
