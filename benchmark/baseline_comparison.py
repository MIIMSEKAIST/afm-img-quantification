"""Baseline comparison for the quantification engine (Referee 3, comment 8).

All methods receive the same JPEG-degraded (q85, 4:2:0) map and color-bar strip
as the manuscript benchmark and are scored on ALL pixels (no validity rejection),
so that color spaces with incomparable distance units can be compared fairly.
The released configuration (luma-weighted YCbCr, CIELAB Delta-E < 20 to the nearest
strip colour as validity mask) and the submitted one (CIELAB with the same mask)
are reported in addition on their valid pixels, as "ycc_nn [valid]" and
"cielab_nn [valid]" (the valid pixels are the same for both).

    ycc_nn           nearest neighbour in YCbCr with the chroma axes weighted by
                     1/4 (released lookup distance)
    rgb_nn           nearest neighbour in Euclidean sRGB (simple baseline)
    cielab_nn        nearest neighbour in CIELAB (submitted method)
    cielab_interp    CIELAB with projection onto the strip polyline
                     (continuous output; removes the 256-level quantization)
    cielab_oracle    CIELAB with the ideal, uncompressed color-bar strip
                     (upper bound: perfect color-bar calibration)
    ycc_w<w>         YCbCr with chroma weight w (sensitivity of the released
                     distance to its one parameter)

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
import afmquant.engine as engine  # noqa: E402
from afmquant import quantify_map, render_to_published_style, evaluate  # noqa: E402
from afmquant.render import colormap_strip  # noqa: E402
from reproduce_benchmark import load_ground_truth, list_benchmark_ibw, material_of  # noqa: E402

COLORMAPS = ["copper", "hot", "jet", "viridis"]
JPEG_QUALITY = 85
METHODS = [  # (name, engine kwargs, strip source)
    ("ycc_nn", dict(color_space="ycc", interpolate=False), "compressed"),
    ("rgb_nn", dict(color_space="rgb", interpolate=False), "compressed"),
    ("cielab_nn", dict(color_space="cielab", interpolate=False), "compressed"),
    ("cielab_interp", dict(color_space="cielab", interpolate=True), "compressed"),
    ("cielab_oracle", dict(color_space="cielab", interpolate=False), "ideal"),
]
CHROMA_WEIGHTS = [1.0, 0.5, 0.125, 0.0625, 0.03125]     # 0.25 is ycc_nn


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
            default_w = engine.CHROMA_WEIGHT
            for w in CHROMA_WEIGHTS:                          # sensitivity to the chroma weight
                engine.CHROMA_WEIGHT = w
                recon, valid, _ = quantify_map(m, s_c, vmin, vmax, distance_threshold=None, color_space="ycc")
                rows.append({"method": f"ycc_w{w:g}", "mask": "all", **base, **evaluate(gt, recon, valid)})
            engine.CHROMA_WEIGHT = default_w
            # released and submitted configurations with the validity mask (the same valid pixels for both)
            for name, cs in (("ycc_nn", "ycc"), ("cielab_nn", "cielab")):
                recon, valid, qc = quantify_map(m, s_c, vmin, vmax, distance_threshold=20.0,
                                                color_space=cs, mask_space="nearest")
                rows.append({"method": name, "mask": "valid", **base,
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
