"""Generate supplementary Table S1 and Figures S1, S2.

  * Table S1 — full per-case validation metrics for the 104 benchmark cases.
  * Figure S1 — benchmark rendering example (one scan, four colormaps + JPEG).
  * Figure S2 — reconstruction error (nMAE) by material and colormap (box plot).

Self-contained: regenerates the per-case metrics from the ground-truth IBW via
the validated `afmquant` engine, so it does not depend on any precomputed CSV
schema.

Usage
-----
    IBW_DIR=data/esm_ibw python figures/make_supplementary.py

Source: migrated from scripts/make_supplementary.py (now engine-backed and
parameterized; outputs under figures/parts/supp).
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import igor2.binarywave as bw
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant import quantify_map, render_to_published_style, evaluate  # noqa: E402
from benchmark.reproduce_benchmark import list_benchmark_ibw  # noqa: E402

IBW_DIR = Path(os.environ.get("IBW_DIR", "data/esm_ibw"))
OUT = Path("figures/parts/supp")
OUT.mkdir(parents=True, exist_ok=True)
COLORMAPS = ["copper", "hot", "jet", "viridis"]
CMAP_HEX = {"copper": "#b87333", "hot": "#d62728", "jet": "#1f77b4", "viridis": "#2ca02c"}


def material_of(stem: str) -> str:
    s = stem.upper()
    return ("LICGC" if s.startswith("LICGC") else "LMNO" if s.startswith("LMNO")
            else "NCM811" if s.startswith("NCM") else stem.split("_")[0])


def load_gt(p: Path) -> np.ndarray:
    gt = bw.load(str(p))["wave"]["wData"][:, :, 2].astype(float)
    return np.nan_to_num(gt, nan=np.nanmedian(gt))


def build_metrics(ibw_files):
    rows = []
    for p in ibw_files:
        gt = load_gt(p)
        for cmap_name in COLORMAPS:
            map_rgb, strip, vmin, vmax = render_to_published_style(gt, cmap_name, 85)
            recon, valid, qc = quantify_map(map_rgb, strip, vmin, vmax)
            m = evaluate(gt, recon, valid)
            rows.append({"case_id": f"{p.stem}__{cmap_name}", "material": material_of(p.stem),
                         "colormap": cmap_name, **m, "valid_pixel_ratio": qc["valid_pixel_ratio"]})
    return pd.DataFrame(rows)


def main():
    ibw_files = list_benchmark_ibw(IBW_DIR)
    if not ibw_files:
        raise SystemExit(f"No benchmark .ibw under {IBW_DIR.resolve()} (set IBW_DIR).")

    df = build_metrics(ibw_files)

    # --- Table S1 ---
    cols = ["case_id", "material", "colormap", "nMAE_pct", "mean_error_pct",
            "std_error_pct", "normalized_EMD_pct", "SSIM", "valid_pixel_ratio"]
    df[cols].round(3).to_excel(OUT / "TableS1_validation_cases.xlsx", index=False)
    print(f"Table S1: {len(df)} cases -> TableS1_validation_cases.xlsx")

    # --- Figure S2: nMAE by material x colormap ---
    fig, ax = plt.subplots(figsize=(10, 5))
    materials = ["LMNO", "LICGC", "NCM811"]
    data_list, positions, labels, colors_list, pos = [], [], [], [], 0
    for mat in materials:
        for cmap_name in COLORMAPS:
            sub = df[(df.material == mat) & (df.colormap == cmap_name)]["nMAE_pct"]
            if len(sub):
                data_list.append(sub.values); positions.append(pos)
                labels.append(f"{mat}\n{cmap_name}"); colors_list.append(CMAP_HEX[cmap_name])
                pos += 1
        pos += 0.5
    bp = ax.boxplot(data_list, positions=positions, widths=0.7, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("nMAE (%)"); ax.set_title("Reconstruction error by material and colormap")
    fig.tight_layout(); fig.savefig(OUT / "FigS2_material_colormap_breakdown.png", dpi=300)
    plt.close(fig)
    print("Figure S2 saved")

    # --- Figure S1: rendering example (first scan, four colormaps) ---
    sample = ibw_files[0]
    gt = load_gt(sample)
    norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for j, cmap_name in enumerate(COLORMAPS):
        rgb = (cm.get_cmap(cmap_name)(norm)[:, :, :3] * 255).astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=85); buf.seek(0)
        axes[0, j].imshow(np.array(Image.open(buf)))
        axes[0, j].set_title(f"Rendered + JPEG ({cmap_name})", fontsize=10); axes[0, j].axis("off")
        axes[1, j].imshow(gt, cmap="gray")
        axes[1, j].set_title("Ground truth (same for all)", fontsize=10); axes[1, j].axis("off")
    fig.suptitle(f"Benchmark rendering example: {sample.stem}", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "FigS1_rendering_examples.png", dpi=300)
    plt.close(fig)
    print("Figure S1 saved")
    print(f"\nAll supplementary materials in: {OUT}")


if __name__ == "__main__":
    main()
