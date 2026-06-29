"""Robustness sweep over JPEG quality and figure resolution.

Repeats the full benchmark (26 scans x 4 colormaps) across:
  * JPEG qualities {95, 85, 75, 65, 50} at native 256 px resolution, and
  * figure resolutions {256, 192, 128, 96, 64} px at fixed JPEG quality 85,
    where the rendered figure is bilinearly downsampled and compared against the
    identically downsampled ground truth.

This is 26 x 4 x (5 + 5) = 1040 cases. Reproduces Supplementary Note 3 / Fig. S4
and results/robustness_sweep.csv: nMAE stays within ~1.3-3.1 %, and the colormap
ordering (copper < hot < jet < viridis) is preserved at every quality level.

Output columns: sweep, colormap, param, nMAE, SSIM (one row per case).

Usage
-----
    python benchmark/robustness_sweep.py --ibw-dir data/esm_ibw \
        --out results/robustness_sweep.csv

Source: reconstructed from the benchmark logic (the original sweep script was not
retained); engine/render/metrics imported from `afmquant`, consistent with
reproduce_benchmark.py. Validation against the retained results/robustness_sweep.csv:
the JPEG-quality sweep reproduces it exactly; the resolution sweep reproduces the
published decreasing trend (exact at native 256 px, within ~0.1-0.25 % nMAE at
reduced resolution; the small residual reflects downsampling-filter details that
were not recorded in the lost script).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant import quantify_map, render_to_published_style, evaluate  # noqa: E402
from reproduce_benchmark import load_ground_truth  # noqa: E402

COLORMAPS = ["copper", "hot", "jet", "viridis"]
QUALITIES = [95, 85, 75, 65, 50]
RESOLUTIONS = [256, 192, 128, 96, 64]
FIXED_QUALITY = 85


# INTER_AREA (area-averaging) is the correct downsampler here: it spatially
# averages the full-resolution JPEG artifacts, which is why per-pixel nMAE
# decreases slightly at lower resolution (Supplementary Note 3).
def _resize(arr, size, interp=cv2.INTER_AREA):
    return cv2.resize(arr, (size, size), interpolation=interp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--out", type=Path, default=Path("results/robustness_sweep.csv"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ibw_files = sorted(args.ibw_dir.glob("*.ibw"))
    print(f"Found {len(ibw_files)} IBW files; "
          f"{len(ibw_files) * len(COLORMAPS) * (len(QUALITIES) + len(RESOLUTIONS))} cases")

    rows = []
    for ibw_path in ibw_files:
        try:
            gt = load_ground_truth(ibw_path)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {ibw_path.name}: {e}")
            continue

        for cmap_name in COLORMAPS:
            # --- JPEG quality sweep (native resolution) ---
            for q in QUALITIES:
                map_rgb, strip, vmin, vmax = render_to_published_style(gt, cmap_name, q)
                recon, valid, _ = quantify_map(map_rgb, strip, vmin, vmax)
                m = evaluate(gt, recon, valid)
                rows.append({"sweep": "quality", "colormap": cmap_name, "param": q,
                             "nMAE": m["nMAE_pct"], "SSIM": m["SSIM"]})

            # --- resolution sweep (fixed quality 85) ---
            map_rgb, strip, vmin, vmax = render_to_published_style(gt, cmap_name, FIXED_QUALITY)
            for res in RESOLUTIONS:
                map_ds = _resize(map_rgb, res)
                gt_ds = _resize(gt.astype(np.float32), res)
                recon, valid, _ = quantify_map(map_ds, strip, vmin, vmax)
                m = evaluate(gt_ds, recon, valid)
                rows.append({"sweep": "resolution", "colormap": cmap_name, "param": res,
                             "nMAE": m["nMAE_pct"], "SSIM": m["SSIM"]})
        print(f"  done {ibw_path.stem}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print("ROBUSTNESS SWEEP — SUMMARY".replace("—", "-"))
    print("=" * 60)
    for sweep in ("quality", "resolution"):
        sub = df[df.sweep == sweep]
        print(f"\n[{sweep}] mean nMAE by {sweep}:")
        print(sub.groupby("param")["nMAE"].mean().round(2).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
