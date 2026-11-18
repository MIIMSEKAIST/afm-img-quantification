"""Structure of the JPEG noise in the benchmark renderings (revision R1, lookup distance).

For every benchmark case (26 scans x 4 colormaps) the map is rendered exactly as in the engine benchmark, once
losslessly and once through the JPEG codec (quality 85, 4:2:0 by default), and the difference between the two is
expressed in the YCbCr coordinates of the codec (ITU-R BT.601, full range). The standard deviations of the luma (Y)
and chroma (Cb, Cr) components show where the codec puts its error: when the chroma error exceeds the luma error, a
lookup distance that trusts luma more than chroma is matched to the noise (afmquant.engine, color_space="ycc").

Usage
-----
    python benchmark/jpeg_noise.py --ibw-dir data/esm_ibw --out results/jpeg_noise.csv [--quality 85] [--subsampling 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant.engine import _RGB2YCC  # noqa: E402
from afmquant.render import encode_decode, get_cmap, normalize_to_unit  # noqa: E402
from reproduce_benchmark import list_benchmark_ibw, load_ground_truth, material_of  # noqa: E402

COLORMAPS = ["copper", "hot", "jet", "viridis"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--out", type=Path, default=Path("results/jpeg_noise.csv"))
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--subsampling", type=int, default=2, help="0 = 4:4:4, 2 = 4:2:0 (Pillow default)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ibw_path in list_benchmark_ibw(args.ibw_dir):
        gt = load_ground_truth(ibw_path)
        vmin, vmax = float(np.nanmin(gt)), float(np.nanmax(gt))
        norm = normalize_to_unit(gt, vmin, vmax)
        for cmap_name in COLORMAPS:
            clean = (get_cmap(cmap_name)(norm)[:, :, :3] * 255).astype(np.uint8)
            jpeg = encode_decode(clean, "jpeg", args.quality, args.subsampling)
            diff = (jpeg.reshape(-1, 3).astype(float) - clean.reshape(-1, 3).astype(float)) @ _RGB2YCC.T
            sd = diff.std(axis=0)
            rows.append({"scan": ibw_path.stem, "material": material_of(ibw_path.stem), "colormap": cmap_name,
                         "quality": args.quality, "subsampling": args.subsampling,
                         "sd_Y": sd[0], "sd_Cb": sd[1], "sd_Cr": sd[2],
                         "chroma_to_luma": float(np.sqrt((sd[1] ** 2 + sd[2] ** 2) / 2) / max(sd[0], 1e-9))})
        print(f"  done {ibw_path.stem}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\nJPEG noise s.d. in YCbCr (code values), mean over scans:")
    print(df.groupby("colormap")[["sd_Y", "sd_Cb", "sd_Cr", "chroma_to_luma"]].mean().reindex(COLORMAPS).round(2).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
