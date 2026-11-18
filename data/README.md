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

These raw scans are archived on Zenodo with a DOI (open data, to support
independent reproduction). Download the archive and place the `.ibw` files in
this directory:

```
data/esm_ibw/
    LICGC_01_4um.ibw
    ...
    NCM811_pristine_07_4um.ibw
```

- **Zenodo DOI:** https://doi.org/10.5281/zenodo.24329899 (version 2.0, CC BY 4.0; the scans are in
  `esm_ibw.zip`). All versions: https://doi.org/10.5281/zenodo.21055265; version 1:
  https://doi.org/10.5281/zenodo.21055266.

## Which files define the benchmark

The 104-case benchmark uses **exactly the 26 scans listed in
`benchmark_scans.txt`** (10 LICGC + 9 LMNO_pristine + 7 NCM811_pristine). All
analysis scripts read that manifest and use only those files, so the reported
counts (26 scans → 104 cases) are reproduced even if extra `.ibw` files are
present in this directory.

The Zenodo deposit (v1) additionally contains **3 auxiliary single-scan files**
(`LICGC.ibw`, `LMNO_pristine.ibw`, `NCM811_pristine.ibw`). They are byte-identical
copies of `LICGC_01_4um.ibw`, `LMNO_pristine_03_4um.ibw` and
`NCM811_pristine_03_4um.ibw`, are *not* part of the benchmark and are ignored by
the scripts. Version 2.0 no longer contains them.

Scan sizes are encoded in the file names; note that `NCM811_pristine_07_4um.ibw`
was acquired with a 2 µm scan size according to its Igor metadata (the name is
kept for continuity with the published case identifiers).

## Literature corpus (not redistributed; copyright)

The literature figure corpus used for the detection evaluation is **not**
redistributed: it is copyrighted by the original publishers. The `collection/`
scripts show how it is assembled from public metadata (Crossref) and open tools.
Its inventory, for reference:

| stage | count |
|-------|-------|
| Crossref records matching the AFM query (2010–) | 50,000 |
| open-access PDFs obtained (Unpaywall) | 4,853 |
| PDFs processed by pdffigures2 (280 failed) | 4,573 |
| extracted images: figures / tables | 17,232 / 2,262 |
| figures registered with captions in `figures.parquet` (pilot subset) | 4,422 (1,164 papers) |

The detection evaluation of the revised manuscript uses a paper-wise random
sample of 1,000 figures (one per paper) drawn from the 17,232 extracted figures,
excluding papers represented in the detector training set.
