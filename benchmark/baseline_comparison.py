"""Baseline comparison for the quantification engine (Referee 3, comment 8).

All methods receive the same JPEG-degraded (q85, 4:2:0) map and color-bar strip
as the manuscript benchmark and are scored on ALL pixels (no validity rejection),
so that color spaces with incomparable distance units can be compared fairly.
The manuscript configuration (CIELAB, Delta-E < 20 validity mask) is reported
in addition, scored on its valid pixels only, as "cielab_nn [valid]".

    rgb_nn           nearest neighbour in Euclidean sRGB (simple baseline)
    cielab_nn        nearest neighbour in CIELAB (manuscript method)
    cielab_interp    CIELAB with projection onto the strip polyline
                     (continuous output; removes the 256-level quantization)
    cielab_oracle    CIELAB with the ideal, uncompressed color-bar strip
                     (upper bound: perfect color-bar calibration)

Usage
-----
    python benchmark/baseline_comparison.py --ibw-dir data/esm_ibw \
        --out results/baseline_comparison.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant import quantify_map, render_to_published_style, evaluate  # noqa: E402
from afmquant.render import colormap_strip  # noqa: E402
from reproduce_benchmark import load_ground_truth, list_benchmark_ibw, material_of  # noqa: E402

COLORMAPS = ["copper", "hot", "jet", "viridis"]
JPEG_QUALITY = 85
METHODS = [  # (name, engine kwargs, strip source)
    ("rgb_nn", dict(color_space="rgb", interpolate=False), "compressed"),
    ("cielab_nn", dict(color_space="cielab", interpolate=False), "compressed"),
    ("cielab_interp", dict(color_space="cielab", interpolate=True), "compressed"),
    ("cielab_oracle", dict(color_space="cielab", interpolate=False), "ideal"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--out", type=Path, default=Path("results/baseline_comparison.csv"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ibw_path in list_benchmark_ibw(args.ibw_dir):
        gt = load_ground_truth(ibw_path)
        for cmap_name in COLORMAPS:
            m, s_c, vmin, vmax = render_to_published_style(gt, cmap_name, JPEG_QUALITY)
            s_ideal = colormap_strip(cmap_name, 256)
            base = {"scan": ibw_path.stem, "material": material_of(ibw_path.stem),
                    "colormap": cmap_name}
            for name, kw, strip_src in METHODS:
                strip = s_c if strip_src == "compressed" else s_ideal
                recon, valid, _ = quantify_map(m, strip, vmin, vmax,
                                               distance_threshold=None, **kw)
                rows.append({"method": name, "mask": "all", **base,
                             **evaluate(gt, recon, valid)})
            # manuscript configuration with its validity mask
            recon, valid, qc = quantify_map(m, s_c, vmin, vmax, distance_threshold=20.0)
            rows.append({"method": "cielab_nn", "mask": "valid", **base,
                         "valid_pixel_ratio": qc["valid_pixel_ratio"],
                         **evaluate(gt, recon, valid)})
        print(f"  done {ibw_path.stem}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON - mean over 104 cases")
    print("=" * 60)
    summary = (df.groupby(["method", "mask"])[["nMAE_pct", "nMAE_IQR_pct", "SSIM",
                                                "std_error_signed_pct"]]
               .mean().round(3))
    print(summary.to_string())
    print("\nnMAE (%) by colormap, all pixels:")
    print(df[df["mask"] == "all"].pivot_table(index="colormap", columns="method",
                                              values="nMAE_pct", aggfunc="mean")
          .reindex(COLORMAPS).round(3).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
