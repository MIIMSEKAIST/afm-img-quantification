"""Geometric properties of the benchmark colormaps in CIELAB.

For each colormap the 256-level strip is converted to CIELAB and the path is
decomposed into its lightness (L*) and chroma (a*, b*) components:

    path_dE          total CIELAB path length  (sum of per-step Delta-E)
    path_dL          lightness-only path length (sum of |Delta L*|)
    path_dab         chroma-plane path length   (sum of sqrt(da^2 + db^2))
    luminance_share  path_dL / path_dE  (fraction of the value encoding carried
                     by lightness, which survives JPEG chroma subsampling)
    L_monotonic      True if L* is monotonic along the colormap
    min_step_dE      smallest per-step Delta-E (local color resolution)

These descriptors are used to interpret the colormap-dependent reconstruction
error (Referee 3, comment 2).

Usage
-----
    python benchmark/colormap_analysis.py --out results/colormap_properties.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from colorspacious import cspace_convert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant.render import colormap_strip  # noqa: E402

COLORMAPS = ["copper", "hot", "jet", "viridis"]


def describe(cmap_name, n=256):
    lab = cspace_convert(colormap_strip(cmap_name, n).astype(float), "sRGB255", "CIELab")
    d = np.diff(lab, axis=0)
    dL = np.abs(d[:, 0])
    dab = np.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2)
    dE = np.sqrt(dL ** 2 + dab ** 2)
    L = lab[:, 0]
    return {
        "colormap": cmap_name,
        "path_dE": float(dE.sum()),
        "path_dL": float(dL.sum()),
        "path_dab": float(dab.sum()),
        "luminance_share": float(dL.sum() / dE.sum()),
        "L_min": float(L.min()), "L_max": float(L.max()),
        "L_monotonic": bool(np.all(np.diff(L) <= 1e-9) or np.all(np.diff(L) >= -1e-9)),
        "min_step_dE": float(dE.min()),
        "median_step_dE": float(np.median(dE)),
        "mean_chroma": float(np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/colormap_properties.csv"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([describe(c) for c in COLORMAPS])
    df.to_csv(args.out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
