"""Codec / chroma-subsampling test (Referee 3, comment 2).

Repeats the 104-case benchmark (26 scans x 4 colormaps) under three encodings of
the rendered map and color bar:

    png        lossless control  -> quantization floor of the 256-level LUT
    jpeg_444   JPEG q85, 4:4:4   -> luma + chroma compression, no subsampling
    jpeg_420   JPEG q85, 4:2:0   -> the manuscript setting (Pillow default)

Decomposes the reconstruction error into (quantization) + (JPEG compression)
+ (chroma subsampling), per colormap.

Usage
-----
    python benchmark/subsampling_sweep.py --ibw-dir data/esm_ibw \
        --out results/subsampling_sweep.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant import quantify_map, render_to_published_style, evaluate  # noqa: E402
from reproduce_benchmark import load_ground_truth, list_benchmark_ibw, material_of  # noqa: E402

COLORMAPS = ["copper", "hot", "jet", "viridis"]
VARIANTS = [  # (name, fmt, subsampling)
    ("png", "png", None),
    ("jpeg_444", "jpeg", 0),
    ("jpeg_420", "jpeg", 2),
]
JPEG_QUALITY = 85


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--out", type=Path, default=Path("results/subsampling_sweep.csv"))
    ap.add_argument("--color-space", default="ycc", choices=["ycc", "cielab", "rgb"],
                    help="lookup distance of the engine (default: released; 'cielab' = submitted engine). The valid "
                         "pixels are the same for every distance (CIELAB Delta-E < 20 to the nearest strip colour)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ibw_path in list_benchmark_ibw(args.ibw_dir):
        gt = load_ground_truth(ibw_path)
        for cmap_name in COLORMAPS:
            for name, fmt, sub in VARIANTS:
                m, s, vmin, vmax = render_to_published_style(
                    gt, cmap_name, JPEG_QUALITY, fmt=fmt, subsampling=sub)
                recon, valid, qc = quantify_map(
                    m, s, vmin, vmax, color_space=args.color_space, mask_space="nearest", distance_threshold=20.0)
                met = evaluate(gt, recon, valid)
                rows.append({"variant": name, "color_space": args.color_space, "scan": ibw_path.stem,
                             "material": material_of(ibw_path.stem),
                             "colormap": cmap_name,
                             "valid_pixel_ratio": qc["valid_pixel_ratio"], **met})
        print(f"  done {ibw_path.stem}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print("CODEC / CHROMA-SUBSAMPLING SWEEP - mean nMAE (%) and SSIM")
    print("=" * 60)
    for metric in ("nMAE_pct", "SSIM"):
        print(f"\n[{metric}]")
        print(df.pivot_table(index="colormap", columns="variant", values=metric,
                             aggfunc="mean").reindex(COLORMAPS)[[v[0] for v in VARIANTS]]
              .round(3).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
