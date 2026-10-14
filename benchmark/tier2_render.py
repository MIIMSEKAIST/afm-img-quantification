"""Tier 2 benchmark: render the ground-truth ESM scans as realistic, complete
publication figures (axes, tick labels, colour bar with tick labels and unit,
scale-bar overlay, resampling to the figure resolution, JPEG encoding, optional
PDF embedding + pdffigures2 extraction) for an end-to-end test of the whole
pipeline: detection -> colour-bar OCR -> pairing -> quantification.

Figure variants (one realism factor changed at a time relative to ``default``):

    default      single panel, dpi 150, 5 tick labels + unit, scale bar, JPEG q85
    no_overlay   no scale bar / text overlay on the map
    clip_2_98    colour range clipped to the 2nd-98th percentile (saturation)
    ticks_2      only the two end labels on the colour bar
    ticks_7      seven tick labels
    hbar         horizontal colour bar below the map
    dpi_72       low-resolution figure (map down-sampled)
    dpi_300      high-resolution figure (map up-sampled, interpolated)
    font_10      larger fonts
    jpeg_q60     stronger JPEG compression
    png          lossless figure file
    multipanel   2x2 composite: ESM amplitude (a), topography (b, 'gray' colormap
                 with its own colour bar), histogram (c) and text panel (d)
    pdf_extract  figure embedded in a PDF page with a caption and recovered
                 with pdffigures2 (re-rasterised), then JPEG q75 (re-compression)

``default`` is rendered for all 26 scans x 4 colormaps; every other variant for
the 26 scans with the colormap assigned round-robin (26 figures each).

Outputs (under --out): <variant>/<case>.{jpg|png}, ground-truth .npy per panel,
and manifest.json listing, for every figure, the expected map panels (relative
bounding boxes, true vmin/vmax/unit, GT file).

Usage
-----
    python benchmark/tier2_render.py --ibw-dir data/esm_ibw --out runs/tier2
    python benchmark/tier2_render.py ... --pdffigures2 tools/pdffigures2/pdffigures2.jar
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # embed TrueType text so pdffigures2 can read captions
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from PIL import Image
import igor2.binarywave as bw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.reproduce_benchmark import load_ground_truth, list_benchmark_ibw, material_of  # noqa: E402

COLORMAPS = ["copper", "hot", "jet", "viridis"]
UNIT = "pm"
DEFAULT = dict(dpi=150, fontsize=8, nticks=5, scalebar=True, clip=None, jpeg_q=85,
               horizontal=False, layout="single", pdf=False)
VARIANTS = {
    "default": {},
    "no_overlay": dict(scalebar=False),
    "clip_2_98": dict(clip=(2, 98)),
    "ticks_2": dict(nticks=2),
    "ticks_7": dict(nticks=7),
    "hbar": dict(horizontal=True),
    "dpi_72": dict(dpi=72),
    "dpi_300": dict(dpi=300),
    "font_10": dict(fontsize=10),
    "jpeg_q60": dict(jpeg_q=60),
    "png": dict(jpeg_q=None),
    "multipanel": dict(layout="2x2"),
    "pdf_extract": dict(pdf=True),
}


def scan_size_um(stem):
    m = re.search(r"_(\d+)um", stem)
    return float(m.group(1)) if m else 4.0


def load_height(ibw_path):
    arr = bw.load(str(ibw_path))["wave"]["wData"]
    h = (arr[:, :, 0].astype(float) + arr[:, :, 1]) / 2
    h = h - np.nanmedian(h)
    return np.nan_to_num(h, nan=0.0) * 1e9          # nm


def add_map(ax, data, cmap, size_um, p, unit, label_prefix=""):
    """imshow + colour bar + optional scale bar; returns (vmin, vmax, cbar)."""
    vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    if p["clip"]:
        vmin, vmax = [float(v) for v in np.percentile(data, p["clip"])]
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=(0, size_um, 0, size_um),
                   interpolation="antialiased")
    ax.set_xlabel("x (µm)", fontsize=p["fontsize"]); ax.set_ylabel("y (µm)", fontsize=p["fontsize"])
    ax.tick_params(labelsize=p["fontsize"] - 1)
    if p["scalebar"]:
        ax.plot([0.08 * size_um, 0.08 * size_um + 1.0], [0.08 * size_um] * 2, color="white", lw=3,
                solid_capstyle="butt")
        ax.text(0.08 * size_um + 0.5, 0.11 * size_um, "1 µm", color="white", ha="center",
                va="bottom", fontsize=p["fontsize"], fontweight="bold")
        ax.text(0.03, 0.97, label_prefix or "ESM", transform=ax.transAxes, color="white",
                ha="left", va="top", fontsize=p["fontsize"], fontweight="bold")
    cb = plt.colorbar(im, ax=ax, orientation="horizontal" if p["horizontal"] else "vertical",
                      fraction=0.08 if p["horizontal"] else 0.046, pad=0.18 if p["horizontal"] else 0.03)
    cb.locator = MaxNLocator(nbins=p["nticks"] - 1, min_n_ticks=2) if p["nticks"] > 2 else \
        matplotlib.ticker.FixedLocator([vmin, vmax])
    cb.update_ticks()
    cb.set_label(f"Amplitude ({unit})" if unit == UNIT else f"Height ({unit})", fontsize=p["fontsize"])
    cb.ax.tick_params(labelsize=p["fontsize"] - 1)
    return vmin, vmax, cb


def axes_rel_bbox(ax, fig, dpi):
    """Axes box (data area) as fractions of the saved canvas."""
    fig.canvas.draw()
    bb = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    W, H = fig.get_size_inches() * dpi
    return [bb.x0 / W, 1 - bb.y1 / H, bb.x1 / W, 1 - bb.y0 / H]      # x0, y0, x1, y1 (top-left origin)


def render_case(stem, gt_pm, height_nm, cmap, p, size_um):
    """Render one figure; returns (png_bytes, panels) where panels lists the
    GT panels with relative bboxes and true scale."""
    plt.rcParams.update({"font.size": p["fontsize"]})
    panels = []
    if p["layout"] == "single":
        fig, ax = plt.subplots(figsize=(3.4, 3.0) if not p["horizontal"] else (3.0, 3.6))
        vmin, vmax, _ = add_map(ax, gt_pm, cmap, size_um, p, UNIT)
        fig.tight_layout()
        panels.append(dict(name="esm", gt="esm", cmap=cmap, vmin=vmin, vmax=vmax, unit=UNIT,
                           bbox=axes_rel_bbox(ax, fig, p["dpi"])))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.0))
        vmin, vmax, _ = add_map(axes[0, 0], gt_pm, cmap, size_um, p, UNIT, label_prefix="(a)")
        panels.append(dict(name="esm", gt="esm", cmap=cmap, vmin=vmin, vmax=vmax, unit=UNIT))
        hvmin, hvmax, _ = add_map(axes[0, 1], height_nm, "gray", size_um, p, "nm", label_prefix="(b)")
        panels.append(dict(name="topo", gt="height", cmap="gray", vmin=hvmin, vmax=hvmax, unit="nm"))
        axes[1, 0].hist(gt_pm.ravel(), bins=60, color="#555"); axes[1, 0].set_xlabel("Amplitude (pm)")
        axes[1, 0].set_ylabel("Counts"); axes[1, 0].set_title("(c)", loc="left", fontsize=p["fontsize"])
        axes[1, 1].axis("off")
        axes[1, 1].text(0.0, 0.9, "(d)  Sample: " + material_of(stem) + "\nDART-ESM, Vac = 1 V\n"
                        f"scan {size_um:g} µm, 256 x 256 px", fontsize=p["fontsize"], va="top")
        fig.tight_layout()
        panels[0]["bbox"] = axes_rel_bbox(axes[0, 0], fig, p["dpi"])
        panels[1]["bbox"] = axes_rel_bbox(axes[0, 1], fig, p["dpi"])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=p["dpi"], facecolor="white")
    plt.close(fig)
    return buf.getvalue(), panels


def encode(png_bytes, jpeg_q):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if jpeg_q is None:
        return np.array(img), "png"
    b = io.BytesIO(); img.save(b, format="JPEG", quality=int(jpeg_q)); b.seek(0)
    return np.array(Image.open(b).convert("RGB")), "jpg"


def run_pdffigures2(jar, pdf_path, out_dir, dpi=150):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-jar", str(jar), "-m", str(out_dir) + "/", "-d", str(out_dir) + "/",
           "-i", str(dpi), str(pdf_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--out", type=Path, default=Path("runs/tier2"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--pdffigures2", type=Path, default=Path("tools/pdffigures2/pdffigures2.jar"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scans = []
    for p in list_benchmark_ibw(args.ibw_dir):
        gt = load_ground_truth(p) * 1e12                      # pm
        scans.append((p.stem, gt, load_height(p), scan_size_um(p.stem)))
    gt_dir = args.out / "gt"; gt_dir.mkdir(exist_ok=True)
    for stem, gt, h, _ in scans:
        np.save(gt_dir / f"{stem}__esm.npy", gt.astype(np.float32))
        np.save(gt_dir / f"{stem}__height.npy", h.astype(np.float32))

    manifest = []
    for vname in args.variants:
        p = {**DEFAULT, **VARIANTS[vname]}
        vdir = args.out / vname; vdir.mkdir(exist_ok=True)
        cases = [(s, c) for s in scans for c in COLORMAPS] if vname == "default" else \
                [(s, COLORMAPS[i % 4]) for i, s in enumerate(scans)]
        pdf_pages = []
        for (stem, gt, h, size_um), cmap in cases:
            case_id = f"{stem}__{cmap}"
            png, panels = render_case(stem, gt, h, cmap, p, size_um)
            if p["pdf"]:
                pdf_pages.append((case_id, png, panels))
                continue
            img, ext = encode(png, p["jpeg_q"])
            fpath = vdir / f"{case_id}.{ext}"
            Image.fromarray(img).save(fpath, quality=95 if ext == "jpg" else None) if ext == "png" \
                else Image.fromarray(img).save(fpath, quality=100, subsampling=0)
            for pn in panels:
                pn["gt_file"] = str(gt_dir / f"{stem}__{pn['gt']}.npy")
            manifest.append(dict(variant=vname, case_id=case_id, scan=stem, material=material_of(stem),
                                 colormap=cmap, image=str(fpath), params=p, panels=panels))
        if p["pdf"] and pdf_pages:
            pdf_path = vdir / f"{vname}.pdf"
            with PdfPages(pdf_path) as pdf:
                for n, (case_id, png, panels) in enumerate(pdf_pages, start=1):
                    img = Image.open(io.BytesIO(png))
                    w_in, h_in = img.size[0] / p["dpi"], img.size[1] / p["dpi"]
                    fig = plt.figure(figsize=(max(w_in, 4.5) + 0.6, h_in + 1.4))
                    axi = fig.add_axes([0.3 / fig.get_figwidth(), 1.0 / fig.get_figheight(),
                                        w_in / fig.get_figwidth(), h_in / fig.get_figheight()])
                    axi.imshow(np.array(img)); axi.axis("off")
                    fig.text(0.05, 0.35 / fig.get_figheight(),
                             f"Figure {n}. ESM amplitude map of {material_of(case_id.split('__')[0])} "
                             f"rendered with the {case_id.split('__')[1]} colormap (test figure {n}).",
                             fontsize=8, wrap=True)
                    pdf.savefig(fig, dpi=p["dpi"]); plt.close(fig)
            ext_dir = vdir / "pdffigures2"
            if ext_dir.exists():
                shutil.rmtree(ext_dir)
            run_pdffigures2(args.pdffigures2, pdf_path, ext_dir, dpi=p["dpi"])
            doc = None
            for n, (case_id, png, panels) in enumerate(pdf_pages, start=1):
                hits = sorted(ext_dir.glob(f"{vname}-Figure{n}-*.png"))
                if hits:
                    img = np.array(Image.open(hits[0]).convert("RGB"))
                    source = "pdffigures2"
                else:
                    # pdffigures2 pairs captions with figures using article-layout
                    # heuristics and returns nothing for these single-figure pages;
                    # fall back to re-rasterising the page region (same DPI).
                    import fitz  # PyMuPDF
                    if doc is None:
                        doc = fitz.open(str(pdf_path))
                    page = doc[n - 1]
                    pix = page.get_pixmap(dpi=p["dpi"])
                    page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                    im0 = Image.open(io.BytesIO(png))
                    w_in, h_in = im0.size[0] / p["dpi"], im0.size[1] / p["dpi"]
                    x0, y0 = int(0.3 * p["dpi"]) - 4, int(0.4 * p["dpi"]) - 4
                    x1, y1 = int((0.3 + w_in) * p["dpi"]) + 4, int((0.4 + h_in) * p["dpi"]) + 4
                    img = np.ascontiguousarray(page_img[max(0, y0):y1, max(0, x0):x1])
                    source = "pymupdf_raster"
                b = io.BytesIO(); Image.fromarray(img).save(b, format="JPEG", quality=75); b.seek(0)
                fpath = vdir / f"{case_id}.jpg"
                Image.open(b).save(fpath, quality=100, subsampling=0)
                stem = case_id.split("__")[0]
                for pn in panels:
                    pn["gt_file"] = str(gt_dir / f"{stem}__{pn['gt']}.npy")
                    pn["bbox_approx"] = True            # extractor crops the page region
                manifest.append(dict(variant=vname, case_id=case_id, scan=stem, material=material_of(stem),
                                     colormap=case_id.split("__")[1], image=str(fpath), params=p, panels=panels,
                                     pdf_source=source))
        print(f"  {vname}: {len(cases)} figures")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {len(manifest)} figures -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
