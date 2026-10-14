"""Scan-level statistics of original vs reconstructed maps (Fig. 5, Table 1,
Fig. S5 and the reliability flags of the manuscript).

Reads the per-case reconstructions written by
``benchmark/reproduce_benchmark.py --recon-dir`` and the IBW scans (for the
topography and contact-resonance-frequency channels used by the automated
reliability diagnostics) and writes ``results/meta_scanlevel.csv`` with:

    scan, material, colormap,
    gt_med, rec_med          scan medians (m); gt over all pixels, rec over valid
    gt_iqr, rec_iqr          interquartile ranges (m)
    gt_cv, rec_cv            spatial heterogeneity IQR/median
    med_dev_pct, cv_dev_pct  relative deviation of rec vs gt (%)
    qc_topo, qc_freq         Pearson r of amplitude vs height / vs frequency
    qc_flag                  |r| > 0.5 on either -> lower-interpretability scan

The first five columns are the ones consumed by ``analysis/meta_analysis.py``
and ``figures/make_figure_parts.py`` (unchanged schema, extended).

Usage
-----
    python analysis/scan_level_stats.py --recon-dir runs/recon_maps \
        --ibw-dir data/esm_ibw --out results/meta_scanlevel.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import igor2.binarywave as bw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.reproduce_benchmark import material_of  # noqa: E402

# IBW channel stack of these Cypher ES DART scans (see data/README.md):
# 0 HeightTrace, 1 HeightRetrace, 2 Amplitude1Trace (ground truth), ...,
# 10 FrequencyTrace, 11 FrequencyRetrace
CH_HEIGHT = (0, 1)
CH_FREQ = (10, 11)
QC_R_THRESHOLD = 0.5


def qc_correlations(ibw_path: Path, gt: np.ndarray):
    arr = bw.load(str(ibw_path))["wave"]["wData"]
    height = (arr[:, :, CH_HEIGHT[0]].astype(float) + arr[:, :, CH_HEIGHT[1]]) / 2
    freq = (arr[:, :, CH_FREQ[0]].astype(float) + arr[:, :, CH_FREQ[1]]) / 2
    r_topo = float(np.corrcoef(gt.ravel(), height.ravel())[0, 1])
    r_freq = float(np.corrcoef(gt.ravel(), freq.ravel())[0, 1])
    return r_topo, r_freq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon-dir", type=Path, default=Path("runs/recon_maps"))
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--out", type=Path, default=Path("results/meta_scanlevel.csv"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(args.recon_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"{args.recon_dir} is empty - run reproduce_benchmark.py "
                         f"with --recon-dir first.")
    qc_cache = {}
    rows = []
    for p in npz_files:
        stem, cmap_name = p.stem.rsplit("__", 1)
        d = np.load(p)
        gt, rec, valid = d["gt"].astype(float), d["recon"].astype(float), d["valid"]
        if stem not in qc_cache:
            ibw = args.ibw_dir / f"{stem}.ibw"
            qc_cache[stem] = qc_correlations(ibw, gt) if ibw.exists() else (np.nan, np.nan)
        r_topo, r_freq = qc_cache[stem]
        g, r = gt, rec[valid]
        gt_med, rec_med = float(np.median(g)), float(np.median(r))
        gt_iqr = float(np.percentile(g, 75) - np.percentile(g, 25))
        rec_iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
        rows.append({
            "scan": stem, "material": material_of(stem), "colormap": cmap_name,
            "gt_med": gt_med, "rec_med": rec_med,
            "gt_iqr": gt_iqr, "rec_iqr": rec_iqr,
            "gt_cv": gt_iqr / gt_med, "rec_cv": rec_iqr / rec_med,
            "med_dev_pct": (rec_med / gt_med - 1) * 100,
            "cv_dev_pct": ((rec_iqr / rec_med) / (gt_iqr / gt_med) - 1) * 100,
            "qc_topo": r_topo, "qc_freq": r_freq,
            "qc_flag": bool(abs(r_topo) > QC_R_THRESHOLD or abs(r_freq) > QC_R_THRESHOLD),
            "valid_ratio": float(valid.mean()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    dev = df.med_dev_pct.abs()
    print(f"{len(df)} cases / {df.scan.nunique()} scans")
    print(f"|scan-median deviation| = {dev.mean():.2f} +/- {dev.std():.2f} % (max {dev.max():.2f} %)")
    print("by material:", df.assign(d=dev).groupby("material").d.mean().round(2).to_dict())
    print(f"heterogeneity: r = {np.corrcoef(df.gt_cv, df.rec_cv)[0, 1]:.3f}, "
          f"|dev| = {df.cv_dev_pct.abs().mean():.1f} %, signed mean = {df.cv_dev_pct.mean():.1f} %")
    one = df[df.colormap == df.colormap.iloc[0]]
    print("QC flags (|r|>0.5):", one.groupby("material").qc_flag.agg(["sum", "count"]).to_dict("index"))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
