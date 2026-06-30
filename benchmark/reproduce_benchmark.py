"""Reproduce the 104-case ground-truth validation benchmark.

For each in-house ESM scan (IBW), render it into a publication-style figure with
each of four colormaps, quantify it back with the engine, and compare against
ground truth. Reproduces Table/Fig. 3 of the manuscript:

    26 ESM scans x 4 colormaps = 104 cases, JPEG quality 85
    -> nMAE 2.09 +/- 0.76 %, SSIM 0.807 +/- 0.105

Usage
-----
    python benchmark/reproduce_benchmark.py \
        --ibw-dir data/esm_ibw --out results/benchmark104.csv

Source: refactored from scripts/gold_validation.py (logic preserved; engine,
render and metrics now imported from the `afmquant` package).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import igor2.binarywave as bw

# allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant import quantify_map, render_to_published_style, evaluate  # noqa: E402

COLORMAPS = ["viridis", "jet", "hot", "copper"]
JPEG_QUALITY = 85
ESM_AMPLITUDE_CHANNEL = 2  # DART ESM amplitude channel in the IBW wave stack


def load_ground_truth(ibw_path: Path) -> np.ndarray:
    """Load the ESM amplitude matrix that serves as ground truth for a scan."""
    data = bw.load(str(ibw_path))
    arr = data["wave"]["wData"]
    gt = arr[:, :, ESM_AMPLITUDE_CHANNEL].astype(float)
    if not np.isfinite(gt).all():
        gt = np.nan_to_num(gt, nan=np.nanmedian(gt))
    return gt


def list_benchmark_ibw(ibw_dir):
    """Return the benchmark IBW files in `ibw_dir`.

    If the manifest data/benchmark_scans.txt is found, restrict to exactly the
    26 listed scans (so extra .ibw files in the directory — e.g. the auxiliary
    single-scan files in the Zenodo deposit — never change the 104-case count).
    Falls back to globbing all *.ibw if no manifest is present.
    """
    ibw_dir = Path(ibw_dir)
    all_ibw = sorted(ibw_dir.glob("*.ibw"))
    repo_root = Path(__file__).resolve().parents[1]
    for manifest in (repo_root / "data" / "benchmark_scans.txt",
                     ibw_dir / "benchmark_scans.txt",
                     ibw_dir.parent / "benchmark_scans.txt"):
        if manifest.exists():
            names = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#")]
            by_name = {p.name: p for p in all_ibw}
            selected = [by_name[n] for n in names if n in by_name]
            missing = [n for n in names if n not in by_name]
            if missing:
                print(f"[manifest] {len(missing)} listed scan(s) not found in "
                      f"{ibw_dir}: {missing[:3]}{'...' if len(missing) > 3 else ''}")
            return selected
    return all_ibw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--out", type=Path, default=Path("results/benchmark104.csv"))
    ap.add_argument("--recon-dir", type=Path, default=None,
                    help="optional dir to dump per-case .npz reconstructions")
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    args = ap.parse_args()

    if args.recon_dir:
        args.recon_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ibw_files = list_benchmark_ibw(args.ibw_dir)
    print(f"Using {len(ibw_files)} benchmark IBW files from {args.ibw_dir}")

    results = []
    for ibw_path in ibw_files:
        try:
            gt = load_ground_truth(ibw_path)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {ibw_path.name}: {e}")
            continue

        for cmap_name in COLORMAPS:
            case_id = f"{ibw_path.stem}__{cmap_name}"
            try:
                map_rgb, strip_rgb, vmin, vmax = render_to_published_style(
                    gt, cmap_name, args.jpeg_quality)
                value_map, valid_mask, qc = quantify_map(map_rgb, strip_rgb, vmin, vmax)
                metrics = evaluate(gt, value_map, valid_mask)
                metrics.update(qc)
                metrics["case_id"] = case_id
                metrics["material"] = ibw_path.stem
                metrics["colormap"] = cmap_name
                results.append(metrics)
                if args.recon_dir:
                    np.savez_compressed(
                        args.recon_dir / f"{case_id}.npz",
                        gt=gt.astype(np.float32),
                        recon=value_map.astype(np.float32),
                        valid=valid_mask)
                print(f"  {case_id}: nMAE={metrics['nMAE_pct']:.2f}%  "
                      f"SSIM={metrics['SSIM']:.3f}  "
                      f"valid={metrics['valid_pixel_ratio']*100:.0f}%")
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {case_id}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print("GROUND-TRUTH VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total cases: {len(df)}")
    print(f"nMAE:  {df['nMAE_pct'].mean():.2f} +/- {df['nMAE_pct'].std():.2f} %")
    print(f"Mean error: {df['mean_error_pct'].mean():.2f} %")
    print(f"Std error:  {df['std_error_pct'].mean():.2f} %")
    print(f"EMD:        {df['normalized_EMD_pct'].mean():.2f} %")
    print(f"SSIM:       {df['SSIM'].mean():.3f} +/- {df['SSIM'].std():.3f}")
    print("\nBy colormap:")
    print(df.groupby("colormap")[["nMAE_pct", "SSIM"]].mean().to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
