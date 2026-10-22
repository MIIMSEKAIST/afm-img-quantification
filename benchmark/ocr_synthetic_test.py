"""Unit-level test of the colour-bar reading stage on synthetic colour bars.

Generates matplotlib colour bars with known scale (random colormap, orientation,
value range and magnitude, number of ticks, font size, unit label, offset
exponent, DPI, JPEG quality), runs the released colour-bar reader

    afmquant.ocr.read_colorbar  (gradient region -> strip -> OCR -> scale fit)

and scores the recovered endpoints against the truth. A case is a *success*
when both endpoints are within ``tol`` (default 2 %) of the true span and the
orientation is correct. Reproduces the OCR-stage table of the revised SI.

Usage
-----
    python benchmark/ocr_synthetic_test.py --n 300 --out results/ocr_synthetic_test.csv
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant.ocr import ColorbarOCR, read_colorbar  # noqa: E402

CMAPS = ["copper", "hot", "jet", "viridis", "gray", "plasma", "inferno", "RdBu", "coolwarm", "YlGnBu"]
UNITS = [None, "nm", "pm", "mV", "pA", "deg", "N/m", "a.u.", "µm"]


def nice_range(rng):
    """Random but realistic (vmin, vmax) pair."""
    kind = rng.choice(["zero_to", "sym", "offset", "decimal"])
    base = float(rng.choice([1, 2, 2.5, 4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100,
                             120, 150, 200, 250, 300, 400, 500, 800, 1000, 1269, 1500, 2000]))
    if kind == "zero_to":
        return 0.0, base
    if kind == "sym":
        return -base, base
    if kind == "offset":
        lo = float(rng.choice([base * 0.3, base * 0.5, 42, 83, 300]))
        return lo, lo + base
    return 0.0, float(rng.choice([0.2, 0.5, 0.8, 1.5]))


def make_case(seed):
    rng = np.random.default_rng(seed)
    cmap = str(rng.choice(CMAPS))
    vertical = rng.random() < 0.7
    vmin, vmax = nice_range(rng)
    exponent = int(rng.choice([0, 0, 0, 0, -3, -9, -12, 3, 6]))   # offset text cases
    vmin_t, vmax_t = vmin * 10.0 ** exponent, vmax * 10.0 ** exponent
    nticks = int(rng.integers(2, 7))
    fontsize = int(rng.integers(6, 11))
    unit = UNITS[int(rng.integers(0, len(UNITS)))]
    dpi = int(rng.choice([100, 150, 200, 300]))
    jpeg_q = int(rng.choice([60, 75, 85, 95]))
    length_in = float(rng.uniform(1.4, 3.2))
    thick_in = float(rng.uniform(0.12, 0.3))

    fig = plt.figure(figsize=((thick_in * 5, length_in) if vertical else (length_in, thick_in * 5)), dpi=dpi)
    ax = fig.add_axes([0.35, 0.05, 0.25, 0.9] if vertical else [0.05, 0.4, 0.9, 0.25])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin_t, vmax_t))
    cb = fig.colorbar(sm, cax=ax, orientation="vertical" if vertical else "horizontal")
    cb.set_ticks(np.linspace(vmin_t, vmax_t, nticks))
    cb.ax.tick_params(labelsize=fontsize)
    if unit:
        cb.set_label(unit if rng.random() < 0.5 else f"Amplitude ({unit})", fontsize=fontsize)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    jb = io.BytesIO()
    Image.fromarray(img).save(jb, format="JPEG", quality=jpeg_q)
    jb.seek(0)
    img = np.array(Image.open(jb).convert("RGB"))
    return img, {"seed": seed, "cmap": cmap, "vertical": vertical, "vmin": vmin_t, "vmax": vmax_t,
                 "exponent": exponent, "nticks": nticks, "fontsize": fontsize, "unit": unit,
                 "dpi": dpi, "jpeg_q": jpeg_q, "img_h": img.shape[0], "img_w": img.shape[1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=Path("results/ocr_synthetic_test.csv"))
    ap.add_argument("--dump-fail", type=Path, default=None, help="dir to save failed images")
    ap.add_argument("--reader", default="v2", choices=["v1", "v2"], help="label-column reader (v2) or the submitted reader (v1)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.dump_fail:
        args.dump_fail.mkdir(parents=True, exist_ok=True)

    ocr = ColorbarOCR()
    rows = []
    for seed in range(args.n):
        img, meta = make_case(seed)
        if args.reader == "v2":
            from afmquant.reader2 import read_colorbar_v2
            H, W = img.shape[:2]
            res = read_colorbar_v2(img, ocr, (0, 0, W, H), (0, 0, W, H), [], [], unit_mode="lexicon")
        else:
            res = read_colorbar(img, ocr)
        rec = {**meta, "ok": bool(res["ok"]), "reason": res.get("reason"),
               "orient_ok": (res.get("orientation") == "vertical") == meta["vertical"]
               if res.get("orientation") else False,
               "n_anchors": res.get("n_anchors", 0), "unit_read": res.get("unit"),
               "ocr_texts": "|".join(t for t, _ in res.get("ocr_items", [])),
               "err_lo_rel": np.nan, "err_hi_rel": np.nan, "success": False}
        if res["ok"]:
            span = meta["vmax"] - meta["vmin"]
            rec["err_lo_rel"] = abs(res["vmin"] - meta["vmin"]) / span
            rec["err_hi_rel"] = abs(res["vmax"] - meta["vmax"]) / span
            rec["success"] = bool(rec["orient_ok"] and rec["err_lo_rel"] <= args.tol
                                  and rec["err_hi_rel"] <= args.tol)
            rec["success_5pct"] = bool(rec["orient_ok"] and rec["err_lo_rel"] <= 0.05
                                       and rec["err_hi_rel"] <= 0.05)
            rec["unit_ok"] = (meta["unit"] is None) or (res.get("unit") == meta["unit"])
        rows.append(rec)
        if args.dump_fail and not rec["success"]:
            Image.fromarray(img).save(args.dump_fail / f"case{seed:04d}_{meta['cmap']}.png")
        if seed % 25 == 0:
            print(f"  {seed}/{args.n}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n" + "=" * 60)
    print(f"OCR SYNTHETIC TEST  n={len(df)}  tol={args.tol:.0%}")
    print("=" * 60)
    print(f"scale read (ok): {df.ok.mean():.1%} | orientation ok: {df.orient_ok.mean():.1%} | "
          f"SUCCESS (both ends within tol): {df.success.mean():.1%} | within 5%: "
          f"{df.get('success_5pct', pd.Series(False, index=df.index)).fillna(False).mean():.1%} | "
          f"unit ok: {df.get('unit_ok', pd.Series(False, index=df.index)).fillna(False).mean():.1%}")
    ok = df[df.ok]
    if len(ok):
        errs = np.r_[ok.err_lo_rel, ok.err_hi_rel]
        print(f"endpoint error (ok cases): median {np.nanmedian(errs):.2%}, "
              f"95th {np.nanpercentile(errs, 95):.2%}, max {np.nanmax(errs):.2%}")
    for col in ("vertical", "exponent", "jpeg_q", "fontsize", "nticks", "cmap"):
        print(f"\nsuccess by {col}:")
        print(df.groupby(col).success.mean().round(3).to_string())
    print("\nfailure reasons:")
    print(df[~df.success].reason.fillna("endpoint error > tol").value_counts().head(10).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
