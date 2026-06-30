"""Simulated literature-scale meta-analysis (statistics only).

Treats the 26 reconstructed scans as a figure corpus and checks that the
scientific conclusions survive reconstruction. For the ground-truth corpus and
for each colormap-specific reconstructed corpus it computes:

  * the material ranking of scan-median ESM amplitudes (expected LMNO > LICGC > NCM811),
  * pairwise Cliff's delta effect sizes,
  * two-sided Mann-Whitney U test p-values.

Reproduces the numbers behind Fig. 5 / Table 1. Figure rendering lives in
figures/make_figure_parts.py; this module is computation only and prints a summary.

Input: results/meta_scanlevel.csv with columns
    scan, material, colormap, gt_med, rec_med   (medians in metres)

Usage
-----
    python analysis/meta_analysis.py --meta results/meta_scanlevel.csv

Source: migrated from the statistics in scripts/make_fig5.py (logic preserved).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

COLORMAPS = ["copper", "hot", "jet", "viridis"]
MATS = ["LMNO", "LICGC", "NCM811"]
DELTA_PAIRS = [("LMNO", "LICGC"), ("LMNO", "NCM811"), ("LICGC", "NCM811")]


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def corpus_stats(amps_by_material: dict) -> dict:
    """Ranking, pairwise Cliff's delta and Mann-Whitney p for one corpus."""
    medians = {m: float(np.median(v)) for m, v in amps_by_material.items()}
    ranking = " > ".join(sorted(medians, key=lambda m: -medians[m]))
    deltas, pvals = {}, {}
    for a, b in DELTA_PAIRS:
        deltas[f"{a}-{b}"] = cliffs_delta(amps_by_material[a], amps_by_material[b])
        pvals[f"{a}-{b}"] = mannwhitneyu(
            amps_by_material[a], amps_by_material[b], alternative="two-sided").pvalue
    return {"medians_pm": medians, "ranking": ranking,
            "cliffs_delta": deltas, "mannwhitney_p": pvals}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=Path, default=Path("results/meta_scanlevel.csv"))
    args = ap.parse_args()

    meta = pd.read_csv(args.meta)
    meta["material"] = meta.material.replace({"NCM": "NCM811"})
    meta["gt_pm"] = meta.gt_med * 1e12
    meta["rec_pm"] = meta.rec_med * 1e12

    # ground-truth corpus is colormap-invariant: take one colormap slice
    gt_slice = meta[meta.colormap == COLORMAPS[0]]
    gt_amps = {m: gt_slice[gt_slice.material == m].gt_pm.values for m in MATS}

    print("=" * 64)
    print("GROUND-TRUTH CORPUS")
    gt_stats = corpus_stats(gt_amps)
    _print_corpus(gt_stats)

    for cm in COLORMAPS:
        sub = meta[meta.colormap == cm]
        rec_amps = {m: sub[sub.material == m].rec_pm.values for m in MATS}
        print("=" * 64)
        print(f"RECONSTRUCTED CORPUS - {cm}")
        _print_corpus(corpus_stats(rec_amps))

    # consistency verdict
    print("=" * 64)
    rankings = {("ground_truth"): gt_stats["ranking"]}
    for cm in COLORMAPS:
        sub = meta[meta.colormap == cm]
        rec_amps = {m: sub[sub.material == m].rec_pm.values for m in MATS}
        rankings[cm] = corpus_stats(rec_amps)["ranking"]
    all_same = len(set(rankings.values())) == 1
    print(f"Ranking preserved across all corpora: {all_same}  ({gt_stats['ranking']})")


def _print_corpus(stats: dict):
    med = "  ".join(f"{m}={stats['medians_pm'][m]:.0f}pm" for m in MATS)
    print(f"  medians: {med}")
    print(f"  ranking: {stats['ranking']}")
    for pair in stats["cliffs_delta"]:
        print(f"  {pair:14s} delta={stats['cliffs_delta'][pair]:+.2f}  "
              f"p={stats['mannwhitney_p'][pair]:.4f}")


if __name__ == "__main__":
    main()
