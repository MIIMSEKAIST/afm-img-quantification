"""Evaluate the full pipeline on the Tier 2 figures produced by tier2_render.py.

For every figure in the manifest the pipeline (detection -> colour-bar OCR ->
pairing -> quantification) is run; each expected ground-truth panel is matched
to the detected map with the largest IoU (>= 0.3) and scored:

    status      ok | map_not_detected | no_bar_detected | bar_unreadable |
                no_compatible_bar | wrong_scale (read range off by > 25 % of span)
                | low_valid_ratio
    scale_err   max(|vmin_read - vmin|, |vmax_read - vmax|) / (vmax - vmin)
    nMAE_pct    range-normalised MAE of the reconstructed map, after resampling
                the reconstruction to the ground-truth grid (256 x 256)
    nMAE_IQR_pct, median_dev_pct, SSIM, valid_pixel_ratio, unit_ok

Runs with the engine in both matching spaces (``--color-spaces cielab rgb``).

Usage
-----
    python benchmark/tier2_evaluate.py --manifest runs/tier2/manifest.json \
        --weights detection/weights/afm_detect_best.pt --out results/tier2_results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant.pipeline import FigurePipeline  # noqa: E402
from afmquant.ocr import ColorbarOCR  # noqa: E402


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _best_shift(gt, vm, vmask, max_shift=4):
    """Integer (dy, dx) shift of the reconstruction that maximises its
    correlation with ground truth (the detector box is not pixel-exact)."""
    best, best_c = (0, 0), -2.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            s = np.roll(np.roll(vm, dy, axis=0), dx, axis=1)
            m = np.roll(np.roll(vmask, dy, axis=0), dx, axis=1)
            m[:max_shift, :] = m[-max_shift:, :] = m[:, :max_shift] = m[:, -max_shift:] = False
            if m.sum() < 100:
                continue
            c = np.corrcoef(gt[m], s[m])[0, 1]
            if c > best_c:
                best, best_c = (dy, dx), c
    return best, best_c


def pairing_correct(det_panel, bars, pn, gt_panels, W, H):
    """True if the colour bar paired with the detected map is the panel's own
    bar: the paired detected bar is assigned to the nearest true bar (centre
    distance). None if the manifest has no bar boxes or the map was not paired."""
    if det_panel.get("bar_idx") is None or "bar_bbox" not in pn:
        return None
    bx = bars[det_panel["bar_idx"]]["box"]
    c = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
    true_bars = []
    for q in gt_panels:
        b = q.get("bar_bbox")
        if b is not None:
            bb = (b[0] * W, b[1] * H, b[2] * W, b[3] * H)
            if not any(np.allclose(bb, t) for t in true_bars):
                true_bars.append(bb)
    nearest = min(true_bars, key=lambda t: ((t[0] + t[2]) / 2 - c[0]) ** 2 + ((t[1] + t[3]) / 2 - c[1]) ** 2)
    own = (pn["bar_bbox"][0] * W, pn["bar_bbox"][1] * H, pn["bar_bbox"][2] * W, pn["bar_bbox"][3] * H)
    return bool(np.allclose(nearest, own))


def score_panel(gt, value_map, valid, vmin_true, vmax_true, align=True):
    """Resample the reconstruction to the GT grid (and align it by an integer
    shift of up to 4 px) and compute the metrics."""
    H, W = gt.shape
    interp = cv2.INTER_AREA if value_map.shape[0] >= H else cv2.INTER_LINEAR
    vm = cv2.resize(value_map.astype(np.float32), (W, H), interpolation=interp)
    vmask = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if vmask.sum() < 0.05 * vmask.size:
        return None
    shift, corr = (0, 0), np.nan
    if align:
        shift, corr = _best_shift(gt, vm, vmask)
        vm = np.roll(np.roll(vm, shift[0], axis=0), shift[1], axis=1)
        vmask = np.roll(np.roll(vmask, shift[0], axis=0), shift[1], axis=1)
        k = 4
        vmask[:k, :] = vmask[-k:, :] = vmask[:, :k] = vmask[:, -k:] = False
    rng = gt.max() - gt.min() + 1e-12
    iqr = np.subtract(*np.percentile(gt, [75, 25])) + 1e-12
    g, r = gt[vmask], vm[vmask]
    mae = float(np.abs(g - r).mean())
    gn = (gt - gt.min()) / rng
    rn = np.clip((vm - gt.min()) / rng, -0.5, 1.5)
    return {"nMAE_pct": mae / rng * 100, "nMAE_IQR_pct": mae / iqr * 100,
            "median_dev_pct": (np.median(r) / np.median(g) - 1) * 100,
            "median_dev_range_pct": (np.median(r) - np.median(g)) / rng * 100,   # for zero-centred data
            "mean_dev_pct": (r.mean() / g.mean() - 1) * 100,
            "mean_dev_range_pct": (r.mean() - g.mean()) / rng * 100,
            "SSIM": float(ssim(gn, rn, data_range=1.0)),
            "valid_pixel_ratio_gtgrid": float(vmask.mean()),
            "align_shift_px": f"{shift[0]},{shift[1]}", "align_corr": float(corr)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("runs/tier2/manifest.json"))
    ap.add_argument("--weights", type=Path, default=Path("detection/weights/afm_detect_best.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/tier2_results.csv"))
    ap.add_argument("--color-spaces", nargs="*", default=["cielab", "rgb"])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--overlay-dir", type=Path, default=None)
    ap.add_argument("--pairing-mode", default="nearest", choices=["nearest", "evidence"])
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.variants:
        manifest = [m for m in manifest if m["variant"] in args.variants]
    ocr = ColorbarOCR()
    # detection + colour-bar reading are done once; only the lookup is repeated per colour space
    pipe0 = FigurePipeline(args.weights, color_space=args.color_spaces[0], ocr=ocr, pairing_mode=args.pairing_mode)
    from afmquant.engine import quantify_map  # noqa: E402
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from run_figure import draw_overlay  # noqa: E402

    rows = []
    for k, m in enumerate(manifest):
        base = {"variant": m["variant"], "case_id": m["case_id"], "scan": m["scan"],
                "material": m["material"], "colormap": m["colormap"]}
        if m.get("image") is None:
            for pn in m["panels"]:
                rows.append({**base, "panel": pn["name"], "color_space": "-", "status": "extraction_failed"})
            continue
        image = np.array(Image.open(m["image"]).convert("RGB"))
        H, W = image.shape[:2]
        res0 = pipe0.process(image, figure_id=m["case_id"])
        if args.overlay_dir:
            Image.fromarray(draw_overlay(image, res0)).save(
                args.overlay_dir / f"{m['variant']}__{m['case_id']}.png")
        for cs in args.color_spaces:
            res = res0
            if cs != args.color_spaces[0]:
                # re-quantify the same crops with the other matching space
                res = dict(res0)
                res["panels"] = []
                for p in res0["panels"]:
                    q = dict(p)
                    if p["value_map"] is not None:
                        mx0, my0, mx1, my1 = p["map_crop_box"]
                        bar = res0["bars"][p["bar_idx"]]
                        vm, valid, qc = quantify_map(image[my0:my1, mx0:mx1], bar["strip"], bar["vmin"],
                                                     bar["vmax"], color_space=cs, mask_space="cielab")
                        q["value_map"], q["valid_mask"] = vm, valid
                        q["qc"] = {**p["qc"], **{k: v for k, v in qc.items()}}
                        q["status"] = "ok" if qc["valid_pixel_ratio"] >= 0.5 else "low_valid_ratio"
                    res["panels"].append(q)
            for pn in m["panels"]:
                gt = np.load(pn["gt_file"]).astype(float)
                bx = [pn["bbox"][0] * W, pn["bbox"][1] * H, pn["bbox"][2] * W, pn["bbox"][3] * H]
                best, best_iou = None, 0.0
                for p in res["panels"]:
                    v = iou(bx, p["map_box"])
                    if v > best_iou:
                        best, best_iou = p, v
                rec = {**base, "panel": pn["name"], "panel_cmap": pn["cmap"], "color_space": cs,
                       "vmin_true": pn["vmin"], "vmax_true": pn["vmax"], "n_maps_detected": len(res["maps"]),
                       "n_bars_detected": len(res["bars"]), "match_iou": best_iou}
                if best is None or best_iou < 0.3:
                    rec["status"] = "map_not_detected"
                    rows.append(rec); continue
                rec["det_status"] = best["status"]
                rec["pairing_ok"] = pairing_correct(best, res["bars"], pn, m["panels"], W, H)
                if best["status"] != "ok":
                    rec["status"] = best["status"]
                    rows.append(rec); continue
                q = best["qc"]
                span = pn["vmax"] - pn["vmin"]
                scale_err = max(abs(q["vmin"] - pn["vmin"]), abs(q["vmax"] - pn["vmax"])) / span
                rec.update({"vmin_read": q["vmin"], "vmax_read": q["vmax"], "scale_err": scale_err,
                            "unit_read": q.get("unit"), "unit_ok": (q.get("unit") or "").lower() == pn["unit"].lower(),
                            "valid_pixel_ratio": q["valid_pixel_ratio"], "n_anchors": q.get("n_anchors")})
                if scale_err > 0.25:
                    rec["status"] = "wrong_scale"
                    rows.append(rec); continue
                sc = score_panel(gt, best["value_map"], best["valid_mask"], pn["vmin"], pn["vmax"])
                if sc is None:
                    rec["status"] = "low_valid_ratio"
                else:
                    rec.update(sc); rec["status"] = "ok"
                rows.append(rec)
        if k % 20 == 0:
            print(f"  {k}/{len(manifest)} {m['variant']} {m['case_id']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n" + "=" * 64)
    print("TIER 2 END-TO-END SUMMARY (per colour space)")
    print("=" * 64)
    for cs in args.color_spaces:
        d = df[df.color_space == cs]
        ok = d[d.status == "ok"]
        print(f"\n[{cs}] panels={len(d)}  ok={len(ok)} ({len(ok)/max(1,len(d)):.1%})")
        print("status:", d.status.value_counts().to_dict())
        if len(ok):
            print(ok.groupby(["variant", "panel"])[["nMAE_pct", "nMAE_IQR_pct", "median_dev_pct", "SSIM", "scale_err"]]
                  .agg(["mean"]).round(3).to_string())
        if "pairing_ok" in d:
            pr = d.dropna(subset=["pairing_ok"])
            print("pairing correct (paired panels):", pr.assign(pairing_ok=pr.pairing_ok.astype(float)).groupby("variant").pairing_ok.mean().round(3).to_dict())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
