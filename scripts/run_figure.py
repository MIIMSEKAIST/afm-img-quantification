"""Run the full pipeline on figure image(s): detection -> colour-bar OCR ->
pairing -> quantification -> QC. Writes, per figure, a JSON summary, an
overlay PNG (boxes, pairing, status) and one .npz per quantified panel.

Usage
-----
    python scripts/run_figure.py --images path/to/figs --weights detection/weights/afm_detect_best.pt \
        --out out/pipeline [--color-space cielab|rgb]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant.pipeline import FigurePipeline  # noqa: E402

COL = {"map": (0, 200, 0), "bar": (220, 0, 220), "ok": (0, 200, 0), "fail": (0, 0, 230), "held": (0, 140, 255)}   # held = orange


def iter_images(p: Path):
    """Image files in a directory (case-insensitive extensions, no duplicates on
    case-insensitive file systems)."""
    if p.is_dir():
        seen = set()
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg") and f.resolve() not in seen:
                seen.add(f.resolve())
                yield f
    else:
        yield p


def draw_overlay(image_rgb, result):
    img = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    for b in result["bars"]:
        x0, y0, x1, y1 = [int(v) for v in b["box"]]
        cv2.rectangle(img, (x0, y0), (x1, y1), COL["bar"], 2)
        txt = f"{b['vmin']:.3g}..{b['vmax']:.3g} {b.get('unit') or ''}" if b.get("ok") else f"bar: {b.get('reason')}"
        cv2.putText(img, txt, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL["bar"], 1, cv2.LINE_AA)
    for i, p in enumerate(result["panels"]):
        x0, y0, x1, y1 = [int(v) for v in p["map_box"]]
        ok = p["status"] == "ok"
        colour = COL["ok"] if ok else (COL["held"] if p["status"] in ("no_exclusive_bar", "own_bar_unreadable") else COL["fail"])
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 2)
        label = f"#{i} {p['status']}"
        if ok:
            label += f" valid={p['qc']['valid_pixel_ratio']:.2f}"
        cv2.putText(img, label, (x0 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        if p["bar_idx"] is not None:
            bx0, by0, bx1, by1 = [int(v) for v in result["bars"][p["bar_idx"]]["box"]]
            cv2.line(img, ((x0 + x1) // 2, (y0 + y1) // 2), ((bx0 + bx1) // 2, (by0 + by1) // 2),
                     COL["bar"], 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("detection/weights/afm_detect_best.pt"))
    ap.add_argument("--out", type=Path, default=Path("out/pipeline"))
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--color-space", default="cielab", choices=["cielab", "rgb"])
    ap.add_argument("--no-npz", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipe = FigurePipeline(args.weights, conf=args.conf, color_space=args.color_space)
    summary = []
    for img_path in iter_images(args.images):
        image = np.array(Image.open(img_path).convert("RGB"))
        res = pipe.process(image, figure_id=img_path.stem)
        rec = {"figure": img_path.name, "n_maps": len(res["maps"]), "n_bars": len(res["bars"]),
               "n_bars_readable": sum(1 for b in res["bars"] if b.get("ok")),
               "panels": []}
        for i, p in enumerate(res["panels"]):
            prec = {k: p[k] for k in ("map_box", "map_conf", "bar_idx", "compat", "status")}
            prec["qc"] = {k: v for k, v in p["qc"].items() if not isinstance(v, np.ndarray)}
            rec["panels"].append(prec)
            if p["value_map"] is not None and not args.no_npz:
                np.savez_compressed(args.out / f"{img_path.stem}_panel{i}.npz",
                                    value_map=p["value_map"].astype(np.float32), valid=p["valid_mask"])
        Image.fromarray(draw_overlay(image, res)).save(args.out / f"{img_path.stem}_overlay.png")
        (args.out / f"{img_path.stem}.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
        summary.append(rec)
        ok = sum(1 for p in rec["panels"] if p["status"] == "ok")
        print(f"{img_path.name}: maps={rec['n_maps']} bars={rec['n_bars']} readable={rec['n_bars_readable']} ok_panels={ok}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
