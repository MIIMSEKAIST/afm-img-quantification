"""Tier 3 benchmark: run the full pipeline on the authors' own published ESM
figures and compare the reconstructions with the raw data (Referee 3, #1d).

For every publisher PDF ``<pdfs>/<paper_key>.pdf``:

1. Figure extraction with pdffigures2, exactly as for the literature corpus
   (``collection/step2_extract.py``: FigureExtractorBatchCli, default 150 dpi,
   PNG; figure number and caption kept). Pages on which pdffigures2 finds no
   figure (or every page, if it fails) fall back to PyMuPDF: raster images and
   vector drawings of the page are clustered into figure regions (with their
   tick labels) and rendered at the same dpi. Already extracted figure images
   can be supplied instead in ``<pdfs>/<paper_key>/*.png|jpg|tif``.
2. The full pipeline (``FigurePipeline``: detection -> colour-bar OCR ->
   pairing -> quantification) on every figure; every detected panel is kept
   with its status.
3. Raw candidates: every 2-D layer of every raw file found (recursively) under
   ``<raw>/<paper_key>/`` and the ``--extra-raw`` folders, in four processing
   variants (raw, plane, flat0, flat1; see ``tier3_match.py``).
4. Automatic matching of every quantified panel to (file, channel, variant,
   orientation [8 dihedral], field of view [full scan or zoomed crop]); see
   ``tier3_match.match_panel``. The hints CSV (optional, may be partial)
   restricts the candidates of a figure to the files/channels it names.
5. Agreement metrics for matched panels, computed as in Tier 2
   (``tier2_evaluate.score_panel``: reconstruction resampled to the raw grid,
   integer-shift alignment, nMAE % of range, nMAE/IQR, median deviation % of
   range, SSIM, Pearson r). Units: the colour-bar unit read by OCR and the raw
   channel unit are converted to SI; with compatible units the agreement is
   absolute, otherwise (unknown/incompatible units) a least-squares gain/offset
   is fitted first ("shape" metrics) and the panel is flagged. Shape, offset-
   matched and clip-aware variants are always reported as extra columns, as
   are the read colour range vs. raw min/max, 1st/99th percentiles and (for
   Asylum files) the AR display range, and the clipped pixel fractions.

Outputs (``--out``)
-------------------
    tier3_panels.csv          one row per detected panel (status, read scale, match, metrics, flags)
    tier3_summary.json        counts by status / match status, medians of the metrics, per paper
    tier3_matches.json        top hypotheses of every match (for checking ambiguous cases)
    overlays/                 per figure: boxes, status, read scale, pairing lines, matched file
    contact_sheets/           per paper: published crop | reconstruction | matched raw (same colormap
                              and range) | difference, one row per matched panel
    figures/                  extracted figure images (+ pdffigures2 JSON)
    recon/                    per panel .npz: value map, valid mask, matched raw region (bar unit)

Usage
-----
    python benchmark/tier3_evaluate.py --pdfs D:/AFM_Database_SH/revision/tier3/pdfs \
        --raw D:/AFM_Database_SH/revision/tier3/raw \
        --hints D:/AFM_Database_SH/revision/tier3/collection_template.csv \
        --weights D:/AFM_Database_SH/runs/detect/runs/afm_detect/v13/weights/best.pt \
        --out D:/AFM_Database_SH/revision/tier3/results

(all of the above are the defaults). Options: ``--papers KEY ...`` to process
a subset, ``--extra-raw DIR ...`` for raw folders shared by all papers,
``--no-crop-search`` to match full scans only, ``--dpi`` for the figure
rasterisation (150 = corpus setting).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afmquant.pipeline import FigurePipeline  # noqa: E402
from afmquant.ocr import ColorbarOCR  # noqa: E402
from benchmark.tier2_evaluate import score_panel, _best_shift  # noqa: E402
from benchmark import tier3_match as tm  # noqa: E402

ROOT = Path("D:/AFM_Database_SH")
DEFAULTS = dict(pdfs=ROOT / "revision/tier3/pdfs", raw=ROOT / "revision/tier3/raw",
                hints=ROOT / "revision/tier3/collection_template.csv", out=ROOT / "revision/tier3/results",
                weights=ROOT / "runs/detect/runs/afm_detect/v13/weights/best.pt",
                pdffigures2=ROOT / "tools/pdffigures2/pdffigures2.jar")
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
NL = chr(10)
_FIG_RE = re.compile(r"^\s*(?:Supplementary\s+|Suppl\.\s*)?(Fig(?:ure)?s?\.?|FIG\.?|Figure)\s*(S?\s?\d+[A-Za-z]?)", re.I)


def log(*a):
    print(*a, flush=True)


# ===================================================== figure extraction
def norm_fig(s):
    """'Figure 3', 'Fig. S2b', '3', 's2' -> '3', 'S2' (figure-level id)."""
    if s is None:
        return ""
    m = re.search(r"(S?)\s*(\d+)", str(s).strip(), re.I)
    return (m.group(1).upper() + m.group(2)) if m else str(s).strip()


def run_pdffigures2(jar, pdf, work, dpi=150, java="java", heap="4g", timeout=300):
    """pdffigures2 with the corpus settings (collection/step2_extract.py).
    Returns (figures, error) where figures are dicts from its JSON."""
    img_dir, json_dir = work / "images", work / "json"
    if work.exists():
        shutil.rmtree(work)
    img_dir.mkdir(parents=True)
    json_dir.mkdir(parents=True)
    cmd = [java, f"-Xmx{heap}", "-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider", "-cp", str(jar),
           "org.allenai.pdffigures2.FigureExtractorBatchCli", str(pdf), "-s", str(work / "stat_file.json"),
           "-m", str(img_dir) + "/", "-d", str(json_dir) + "/", "-i", str(int(dpi))]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return [], "java not found"
    except subprocess.TimeoutExpired:
        return [], f"timeout ({timeout}s)"
    if r.returncode != 0:
        return [], f"exit {r.returncode}: {(r.stderr or '').strip()[-300:]}"
    figs = []
    for jf in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        data = data.get("figures", []) if isinstance(data, dict) else data
        for f in data or []:
            if str(f.get("figType", "Figure")) != "Figure":
                continue
            p = Path(str(f.get("renderURL", "")))
            if not p.is_absolute():
                p = (img_dir / p.name) if (img_dir / p.name).exists() else p
            if not p.exists():
                hits = sorted(img_dir.glob(f"*-Figure{f.get('name')}-*.png"))
                p = hits[0] if hits else None
            if p is None or not Path(p).exists():
                continue
            figs.append({"name": str(f.get("name", "")), "caption": f.get("caption") or "",
                         "page": int(f.get("page", -1)), "region": f.get("regionBoundary"),
                         "image_path": Path(p), "source": "pdffigures2"})
    return figs, None


def _merge_rects(rects, gap):
    """Greedy union of rectangles closer than ``gap`` (PDF points)."""
    import fitz  # PyMuPDF
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        while rects:
            a = rects.pop()
            grown = fitz.Rect(a.x0 - gap, a.y0 - gap, a.x1 + gap, a.y1 + gap)
            keep = []
            for b in rects:
                if grown.intersects(b):
                    a |= b
                    grown = fitz.Rect(a.x0 - gap, a.y0 - gap, a.x1 + gap, a.y1 + gap)
                    changed = True
                else:
                    keep.append(b)
            rects = keep
            out.append(a)
        rects = out
    return rects


def pymupdf_figures(pdf, work, dpi=150, covered=None, min_side_pt=60.0, min_side_covered_pt=100.0):
    """Fallback figure finder: cluster the raster images and vector drawings of
    each page into figure regions, add the short text spans around them (tick
    and colour-bar labels), render each region at ``dpi`` and attach the
    nearest 'Fig. N' caption on the page.

    ``covered`` maps page -> list of regions (PDF points) already extracted by
    pdffigures2; clusters overlapping them are skipped and, on such pages,
    only large clusters (>= ``min_side_covered_pt``) are kept."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf))
    covered = covered or {}
    out = []
    work.mkdir(parents=True, exist_ok=True)
    for pno, page in enumerate(doc):
        cov = [fitz.Rect(r) for r in covered.get(pno, [])]
        min_side = min_side_covered_pt if cov else min_side_pt
        W, H = page.rect.width, page.rect.height
        rects = []
        try:
            rects += [fitz.Rect(i["bbox"]) for i in page.get_image_info()]
        except Exception:  # noqa: BLE001
            pass
        try:
            for d in page.get_drawings():
                r = fitz.Rect(d["rect"])
                if r.width > 0.8 * W and r.height < 3:      # rules / page frames
                    continue
                if r.width * r.height > 0.85 * W * H:        # page background
                    continue
                rects.append(r)
        except Exception:  # noqa: BLE001
            pass
        rects = [r & page.rect for r in rects if not (r & page.rect).is_empty]
        clusters = [c for c in _merge_rects(rects, gap=6.0) if c.width >= min_side and c.height >= min_side]
        clusters = [c for c in clusters if not any((c & r).get_area() > 0.3 * c.get_area() for r in cov)]
        if not clusters:
            continue
        blocks = page.get_text("dict").get("blocks", [])
        spans, captions = [], []
        for b in blocks:
            if b.get("type") != 0:
                continue
            text = " ".join(s["text"] for ln in b.get("lines", []) for s in ln.get("spans", [])).strip()
            m = _FIG_RE.match(text)
            if m:
                captions.append((fitz.Rect(b["bbox"]), norm_fig(m.group(2)), text))
                continue
            for ln in b.get("lines", []):
                for s in ln.get("spans", []):
                    if s["text"].strip() and len(s["text"].strip()) <= 30:
                        spans.append(fitz.Rect(s["bbox"]))
        for k, c in enumerate(clusters):
            grown = fitz.Rect(c.x0 - 28, c.y0 - 16, c.x1 + 40, c.y1 + 22)
            for s in spans:
                if grown.contains(s) or (grown.intersects(s) and (s & grown).get_area() > 0.6 * s.get_area()):
                    c |= s
            c = fitz.Rect(c.x0 - 3, c.y0 - 3, c.x1 + 3, c.y1 + 3) & page.rect
            # caption: nearest "Fig. N" block below (else above) the region
            name, caption, best = "", "", None
            for r, nm, text in captions:
                d = (r.y0 - c.y1) if r.y0 >= c.y1 - 5 else (c.y0 - r.y1) + 200
                if d >= -5 and (best is None or d < best):
                    best, name, caption = d, nm, text
            pix = page.get_pixmap(dpi=dpi, clip=c)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
            path = work / f"page{pno + 1:03d}_region{k + 1}.png"
            Image.fromarray(np.ascontiguousarray(img)).save(path)
            out.append({"name": name, "fallback_name": f"p{pno + 1}-{k + 1}", "caption": caption, "page": pno,
                        "region": [c.x0, c.y0, c.x1, c.y1], "image_path": path, "source": "pymupdf"})
    return out


def extract_figures(pdf, work, jar, dpi, java):
    """pdffigures2 first; PyMuPDF for pages without a pdffigures2 figure."""
    figs, err = run_pdffigures2(jar, pdf, work / "pdffigures2", dpi=dpi, java=java)
    if err:
        log(f"    pdffigures2 failed ({err}); PyMuPDF fallback for all pages")
    covered = {}
    for f in figs:
        b = f.get("region") or {}
        if isinstance(b, dict) and {"x1", "y1", "x2", "y2"} <= set(b):
            covered.setdefault(f["page"], []).append((b["x1"], b["y1"], b["x2"], b["y2"]))
        else:
            covered.setdefault(f["page"], []).append((0, 0, 1e5, 1e5))
    fb = pymupdf_figures(pdf, work / "pymupdf", dpi=dpi, covered=covered)
    log(f"    pdffigures2: {len(figs)} figure(s); PyMuPDF fallback: {len(fb)} further region(s)")
    figs = figs + fb
    # figure id from the figure number (pdffigures2 name / caption); unique labels
    seen = {}
    for f in figs:
        f["fig_id"] = norm_fig(f["name"]) if f["name"] else ""
        base = f["fig_id"] or f.get("fallback_name") or "x"
        seen[base] = seen.get(base, 0) + 1
        f["label"] = re.sub(r"[^A-Za-z0-9_.-]", "_", base if seen[base] == 1 else f"{base}_{seen[base]}")
    return figs, err


def figures_from_folder(folder):
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMG_EXT:
            m = re.search(r"(?:fig(?:ure)?)?[ _-]*(S?\d+)", p.stem, re.I)
            name = norm_fig(m.group(1)) if m else p.stem
            out.append({"name": name, "label": p.stem, "fig_id": name, "caption": "", "page": -1,
                        "region": None, "image_path": p, "source": "image_file"})
    return out


# =================================================================== hints
def raw_root_for(raw_dir, paper):
    """<raw>/<paper_key>; else the longest <raw>/<key> whose key prefixes the
    paper key (e.g. 'Chen_2025_JPCC' for the SI PDF 'Chen_2025_JPCC_SI')."""
    raw_dir = Path(raw_dir)
    if (raw_dir / paper).is_dir():
        return raw_dir / paper
    if not raw_dir.is_dir():
        return None
    cands = [d for d in raw_dir.iterdir() if d.is_dir() and len(paper) > len(d.name)
             and paper.startswith(d.name) and paper[len(d.name)] in "_-. "]
    return max(cands, key=lambda d: len(d.name)) if cands else None


def load_hints(path):
    """Hint rows by lower-case paper key. Excel on Korean Windows saves "CSV"
    as cp949 unless "CSV UTF-8" is chosen, so both encodings are accepted."""
    if path is None or not Path(path).exists():
        return {}
    df, err = None, None
    for enc in ("utf-8-sig", "cp949", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception as e:  # noqa: BLE001
            err = e
    if df is None:
        log(f"  hints CSV not readable ({err}); ignored")
        return {}
    df.columns = [str(c).strip().lower() for c in df.columns]
    out = {}
    for _, r in df.iterrows():
        key = str(r.get("paper_key", "")).strip()
        if key:
            out.setdefault(key.lower(), []).append({k: str(v).strip() for k, v in r.items()})
    return out


def _split_list(s):
    return [x.strip() for x in re.split(r"[;,\n]", s or "") if x.strip()]


def hint_mask(bank, rows):
    """Candidates allowed by the hint rows of one figure (None = no restriction)
    and a note on what was applied."""
    rows = [r for r in rows if r.get("raw_file") or r.get("channel")]
    if not rows:
        return None, ""
    lib = bank.lib
    allowed = np.zeros(len(bank.cands), bool)
    notes = []
    for r in rows:
        files = [Path(f.replace("\\", "/")).name.lower() for f in _split_list(r.get("raw_file"))]
        chans = [c.lower().replace(" ", "") for c in _split_list(r.get("channel"))]
        lay_ok = []
        for L in lib.layers:
            fn = L.file.name.lower()
            f_ok = not files or any(f in (fn, L.file.stem.lower()) or fn.startswith(f) for f in files)
            c_ok = not chans or any(c == L.channel.lower() or c in L.channel.lower() for c in chans)
            if f_ok and c_ok:
                lay_ok.append(L.lid)
        if not lay_ok:
            notes.append(f"hint not found (file={r.get('raw_file')!r}, channel={r.get('channel')!r})")
            continue
        allowed |= np.isin(bank.lids, lay_ok)
    if not allowed.any():
        return None, "; ".join(notes + ["no hinted candidate found -> all raw files searched"])
    return allowed, "; ".join(notes) if notes else "candidates restricted by hints"


# =============================================================== panels
def reading_order(panels):
    """1-based reading-order position of each panel (rows top to bottom,
    left to right inside a row)."""
    if not panels:
        return []
    boxes = [p["map_box"] for p in panels]
    hs = np.median([b[3] - b[1] for b in boxes])
    idx = sorted(range(len(boxes)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    rows, cur, cy = [], [], None
    for i in idx:
        y = (boxes[i][1] + boxes[i][3]) / 2
        if cy is not None and abs(y - cy) > 0.5 * hs:
            rows.append(cur)
            cur = []
        cur.append(i)
        cy = y if cy is None else (cy * (len(cur) - 1) + y) / len(cur)
    rows.append(cur)
    pos = [0] * len(boxes)
    k = 1
    for row in rows:
        for i in sorted(row, key=lambda i: boxes[i][0]):
            pos[i] = k
            k += 1
    return pos


# ============================================================== metrics
def aligned(gt, vm, valid):
    """Reconstruction on the raw grid with the Tier 2 alignment (score_panel)."""
    H, W = gt.shape
    interp = cv2.INTER_AREA if vm.shape[0] >= H else cv2.INTER_LINEAR
    v = cv2.resize(vm.astype(np.float32), (W, H), interpolation=interp)
    m = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if m.sum() < 0.05 * m.size:
        return v, m, (0, 0), np.nan
    shift, corr = _best_shift(gt, v, m)
    v = np.roll(np.roll(v, shift[0], axis=0), shift[1], axis=1)
    m = np.roll(np.roll(m, shift[0], axis=0), shift[1], axis=1)
    k = 4
    m[:k, :] = m[-k:, :] = m[:, :k] = m[:, -k:] = False
    return v, m, shift, corr


def _score(gt, vm, valid, vmin, vmax, suffix=""):
    sc = score_panel(gt, vm, valid, vmin, vmax)
    if sc is None:
        return {}
    out = {"nMAE_pct": sc["nMAE_pct"], "nMAE_IQR_pct": sc["nMAE_IQR_pct"],
           "median_dev_range_pct": sc["median_dev_range_pct"],
           "abs_median_dev_range_pct": abs(sc["median_dev_range_pct"]),
           "mean_dev_range_pct": sc["mean_dev_range_pct"], "SSIM": sc["SSIM"], "pearson_r": sc["align_corr"],
           "align_shift_px": sc["align_shift_px"], "valid_pixel_ratio_rawgrid": sc["valid_pixel_ratio_gtgrid"]}
    if suffix:     # alternative references: only the metrics that depend on the reference values
        return {k + suffix: out[k] for k in ("nMAE_pct", "nMAE_IQR_pct", "median_dev_range_pct",
                                             "abs_median_dev_range_pct", "mean_dev_range_pct", "SSIM")}
    return out


def compute_metrics(region_si, raw_unit, vm, valid, vmin, vmax, unit_read, meta):
    """Agreement between the reconstruction (bar unit) and the matched raw region (SI)."""
    fb, bb = tm.parse_unit(unit_read)
    if bb is None:
        unit_status = "unknown_bar_unit"
    elif raw_unit is None:
        unit_status = "unknown_raw_unit"
    elif bb != raw_unit:
        unit_status = "incompatible"
    else:
        unit_status = "compatible"
    compatible = unit_status == "compatible"
    gt = (region_si / fb) if compatible else region_si.astype(np.float64)
    rec = {"unit_status": unit_status, "metric_basis": "absolute" if compatible else "linear_fit",
           "unit_read_si": f"{fb:g} {bb}" if bb else None}
    # read colour range vs. the displayed raw region
    p = np.percentile(gt, [0, 0.5, 1, 99, 99.5, 100])
    rec.update({"raw_min": p[0], "raw_max": p[5], "raw_p1": p[2], "raw_p99": p[3],
                "raw_unit_shown": tm.unit_label(fb, bb) if compatible else (raw_unit or "?")})
    flags = []
    # gain/offset (shape agreement): least squares on block-averaged maps (~32 blocks on the long side),
    # using blocks free of invalid and saturated pixels. A pixel-level fit is biased low (regression
    # dilution: the raw pixel noise is not resolved by the figure; self-test gain/true 0.45-0.93 at pixel
    # level vs 0.91-1.07 with 8x8 blocks).
    v, m, shift, corr = aligned(gt, vm, valid)
    span = vmax - vmin
    unsat = m & (v > vmin + 0.005 * span) & (v < vmax - 0.005 * span)
    use = unsat if unsat.sum() > 0.3 * m.sum() else m
    gain, offset = np.nan, np.nan
    H, W = gt.shape
    k = max(1, int(round(max(H, W) / 32)))
    if k > 1 and H // k >= 4 and W // k >= 4:
        def blk(x):
            return cv2.resize(x.astype(np.float32), (W // k, H // k), interpolation=cv2.INTER_AREA)
        cover = blk(use.astype(np.float32))
        mb = cover > 0.95
        xs = (blk(np.where(use, gt, 0.0)) / np.maximum(cover, 1e-6))[mb]
        ys = (blk(np.where(use, v, 0.0)) / np.maximum(cover, 1e-6))[mb]
    else:
        mb, xs, ys = None, gt[use], v[use]
    if len(xs) < 20 and use.sum() > 50:                       # too few clean blocks: pixel-level fit
        xs, ys = gt[use], v[use]
    if len(xs) >= 20 and np.ptp(xs) > 0:
        gain, offset = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), 1)
    rec.update({"fit_gain": gain, "fit_offset": offset})
    if m.sum() > 50:
        iq_r = np.subtract(*np.percentile(gt[m], [75, 25]))
        rec["iqr_ratio"] = float(np.subtract(*np.percentile(v[m], [75, 25])) / iq_r) if iq_r > 0 else np.nan
    gt_fit = gain * gt + offset if np.isfinite(gain) else None
    shown = gt if compatible else gt_fit
    if shown is not None:
        rng_shown = np.ptp(shown) + 1e-30
        rec.update({"span_ratio": span / rng_shown,
                    "span_ratio_p1p99": span / (np.subtract(*np.percentile(shown, [99, 1])) + 1e-30),
                    "clip_below_frac": float((shown < vmin - 1e-3 * span).mean()),
                    "clip_above_frac": float((shown > vmax + 1e-3 * span).mean())})
        lo, hi = shown.min(), shown.max()
        p1, p99 = np.percentile(shown, [1, 99])
        tol = 0.02 * (hi - lo)
        ar_lo, ar_hi = meta.get("ar_display_lo"), meta.get("ar_display_hi")
        if compatible and ar_lo is not None:
            rec.update({"ar_display_lo": ar_lo / fb, "ar_display_hi": ar_hi / fb})
        if abs(vmin - lo) <= tol and abs(vmax - hi) <= tol:
            rtype = "full"
        elif abs(vmin - p1) <= tol and abs(vmax - p99) <= tol:
            rtype = "p1-p99"
        elif compatible and ar_lo is not None and abs(vmin - ar_lo / fb) <= tol and abs(vmax - ar_hi / fb) <= tol:
            rtype = "ar_display"
        elif rec["clip_below_frac"] + rec["clip_above_frac"] > 0.005:
            rtype = "clipped"
        else:
            rtype = "wider"
        rec["range_type"] = rtype
        if rec["clip_below_frac"] + rec["clip_above_frac"] > 0.005:
            flags.append("clipped_display")
    # primary metrics (Tier 2 definitions)
    if compatible:
        rec.update(_score(gt, vm, valid, vmin, vmax))
        if gt_fit is not None:
            rec.update(_score(gt_fit, vm, valid, vmin, vmax, "_fit"))
        off = float(np.median(v[m]) - np.median(gt[m])) if m.sum() > 50 else 0.0
        sc_off = _score(gt + off, vm, valid, vmin, vmax)
        rec.update({"offset_applied": off, "nMAE_pct_offset": sc_off.get("nMAE_pct"),
                    "median_dev_range_pct_offset": sc_off.get("median_dev_range_pct")})
        if np.isfinite(gain) and (gain > 3 or gain < 1 / 3):
            flags.append("gain_off(unit/calibration?)")
        if rec.get("span_ratio") and (rec["span_ratio"] > 30 or rec["span_ratio"] < 1 / 30):
            flags.append("span_mismatch(prefix?)")
    elif gt_fit is not None:
        full = _score(gt_fit, vm, valid, vmin, vmax)
        rec.update(full)                                           # primary = shape metrics
        rec.update({k + "_fit": v for k, v in full.items() if k in ("nMAE_pct", "nMAE_IQR_pct", "median_dev_range_pct",
                                                                   "abs_median_dev_range_pct", "mean_dev_range_pct", "SSIM")})
        flags.append(unit_status)
    clip_basis = gt if compatible else gt_fit
    if clip_basis is not None:
        sc_c = _score(np.clip(clip_basis, vmin, vmax), vm, valid, vmin, vmax)
        rec["nMAE_pct_clipaware"] = sc_c.get("nMAE_pct")
    return rec, flags, gt if compatible else gt_fit


# ======================================================== visualisation
def draw_match_overlay(image, res, rows_by_idx, title):
    """Pipeline overlay (scripts/run_figure.draw_overlay: boxes, status, read
    scale, pairing lines) + the matched raw file per panel."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_figure import draw_overlay  # noqa: E402
    img = cv2.cvtColor(draw_overlay(image, res), cv2.COLOR_RGB2BGR)
    for i, p in enumerate(res["panels"]):
        r = rows_by_idx.get(i, {})
        x0, y0, x1, y1 = [int(v) for v in p["map_box"]]
        ms = r.get("match_status")
        if ms in (None, "not_quantified"):
            txt, col = f"pos {r.get('panel_pos', '?')}", (0, 0, 230)
        else:
            txt = f"pos {r.get('panel_pos')}: {Path(str(r.get('match_file') or '-')).name[:22]} " \
                  f"{r.get('match_channel') or ''}{'*' if r.get('ambiguous_channel') else ''} " \
                  f"s={r.get('match_score', np.nan):.2f}{' crop' if r.get('match_fov') == 'crop' else ''}"
            col = (0, 160, 0) if ms == "matched" and not r.get("ambiguous_file") else \
                ((0, 140, 255) if ms in ("matched", "weak") else (0, 0, 230))
        scale = max(0.35, min(0.55, (x1 - x0) / 700))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + 6 + tw, y0 + 8 + th), (255, 255, 255), -1)
        cv2.putText(img, txt, (x0 + 4, y0 + 5 + th), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    pad = np.full((26, img.shape[1], 3), 255, np.uint8)
    title += "  (s = match score; * = channel ambiguous, e.g. trace/retrace)"
    cv2.putText(pad, title[:170], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(np.vstack([pad, img]), cv2.COLOR_BGR2RGB)


def contact_sheets(paper, items, out_dir, per_page=6):
    """One row per matched panel: published crop | reconstruction | matched raw
    (both with the figure's own colour bar as colormap and the read range) |
    difference; the header names file, channel, variant, orientation, score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    paths = []
    for page in range(0, len(items), per_page):
        chunk = items[page:page + per_page]
        n = len(chunk)
        fig = plt.figure(figsize=(11, 3.25 * n + 0.4))
        gs = fig.add_gridspec(2 * n, 4, height_ratios=[0.42, 2.6] * n, hspace=0.12, wspace=0.08,
                              left=0.02, right=0.96, top=1 - 0.4 / (3.25 * n + 0.4), bottom=0.01)
        for k, it in enumerate(chunk):
            r = it["row"]
            cmap = ListedColormap(it["strip"][::-1] / 255.0) if it.get("strip") is not None else "viridis"
            vmin, vmax = r["vmin_read"], r["vmax_read"]
            axh = fig.add_subplot(gs[2 * k, :])
            axh.axis("off")
            basis = "" if r.get("metric_basis") == "absolute" else " [gain/offset fitted: units unknown/incompatible]"
            crop = f" {100 * r['crop_frac']:.0f}% crop" if r.get("match_fov") == "crop" else ""
            head = (f"Fig. {r['figure']}  pos {r['panel_pos']} (panel #{r['panel_idx']}, {r['status']})  ->  "
                    f"{r['match_file']}  |  {r['match_channel']}  |  {r['match_variant']}, {r['match_orientation']}, "
                    f"{r['match_fov']}{crop}  |  {r.get('match_status')}" + NL +
                    f"score {r['match_score']:.3f} (margin {r.get('margin', np.nan):.3f} to "
                    f"{Path(str(r.get('runner_up_file') or '-')).name}; channel margin {r.get('channel_margin', np.nan):.3f})"
                    f"  |  r {r.get('pearson_r', np.nan):.3f}  |  nMAE {r.get('nMAE_pct', np.nan):.2f}%  |  "
                    f"median dev {r.get('median_dev_range_pct', np.nan):+.2f}%  |  range {vmin:.4g} to {vmax:.4g} "
                    f"{r.get('unit_read') or '?'}{basis}" + (NL + f"flags: {r['flags']}" if r.get("flags") else ""))
            axh.text(0.0, 0.05, head, fontsize=7.2, va="bottom", ha="left", transform=axh.transAxes)
            axs = [fig.add_subplot(gs[2 * k + 1, j]) for j in range(4)]
            axs[0].imshow(it["crop_rgb"])
            axs[1].imshow(it["recon"], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            axs[2].imshow(it["raw"], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            diff = np.where(it["mask"], it["recon"] - it["raw"], np.nan)
            lim = 0.1 * (vmax - vmin)
            im = axs[3].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
            for ax, t in zip(axs, ("published crop", "reconstruction", "matched raw (same colormap/range)",
                                   "recon - raw (+/-10 % of range)")):
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(t, fontsize=7, pad=2)
        fig.suptitle(f"Tier 3 matches - {paper} (sheet {page // per_page + 1})", fontsize=10, y=0.999)
        path = out_dir / f"{paper}_contact_{page // per_page + 1}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


# ================================================================= main
def _fmt_box(b):
    return ",".join(str(int(round(v))) for v in b) if b is not None else ""


def evaluate_paper(paper, pdf_or_dir, args, pipe, params, hints):
    out = args.out
    t0 = time.time()
    fig_dir = out / "figures" / paper
    fig_dir.mkdir(parents=True, exist_ok=True)
    if pdf_or_dir.is_dir():
        figs, err = figures_from_folder(pdf_or_dir), None
    else:
        figs, err = extract_figures(pdf_or_dir, fig_dir / "_work", args.pdffigures2, args.dpi, args.java)
    log(f"  {len(figs)} figure(s) extracted ({time.time() - t0:.0f}s)")
    # raw library (paper folder + shared folders)
    own = raw_root_for(args.raw, paper)
    roots = [r for r in [own] + list(args.extra_raw or []) if r is not None and Path(r).exists()]
    if own is not None and own.name != paper:
        log(f"  raw folder {own} used for {paper} (prefix match)")
    lib = tm.RawLibrary(roots, log=lambda *a: log(*a)).scan() if roots else None
    bank = tm.CandidateBank(lib, log=lambda *a: log(*a)) if lib is not None and lib.layers else None
    if bank is None:
        log(f"  no raw data found under {args.raw / paper} -> panels are not matched")
    hint_rows = hints.get(paper.lower()) or (hints.get(own.name.lower(), []) if own is not None else [])
    used_hint_figs = set()
    rows, details, sheet_items = [], [], []
    for fi, f in enumerate(figs):
        img = np.array(Image.open(f["image_path"]).convert("RGB"))
        label = f["label"]
        saved = fig_dir / f"Fig{label}.png"
        if Path(f["image_path"]).resolve() != saved.resolve():
            Image.fromarray(img).save(saved)
        res = pipe.process(img, figure_id=f"{paper}:{label}")
        n_ok = sum(1 for p in res["panels"] if p["status"] == "ok")
        log(f"  Fig. {label} ({f['source']}, {img.shape[1]}x{img.shape[0]}): {len(res['panels'])} map(s), "
            f"{len(res['bars'])} bar(s), {n_ok} quantified")
        if not res["panels"]:
            continue
        fig_hints = [h for h in hint_rows if norm_fig(h.get("figure")) == f["fig_id"]]
        if fig_hints:
            used_hint_figs.add(f["fig_id"])
        allowed, hint_note = hint_mask(bank, fig_hints) if (bank is not None and fig_hints) else (None, "")
        pos = reading_order(res["panels"])
        rows_by_idx = {}
        for i, p in enumerate(res["panels"]):
            q = p.get("qc") or {}
            bar = res["bars"][p["bar_idx"]] if p.get("bar_idx") is not None else None
            row = {"paper_key": paper, "figure": label, "page": f["page"] + 1 if f["page"] >= 0 else None,
                   "figure_source": f["source"],
                   "figure_image": str(saved), "caption": (f.get("caption") or "")[:200],
                   "panel_idx": i, "panel_pos": pos[i], "map_box": _fmt_box(p["map_box"]), "status": p["status"],
                   "bar_idx": p.get("bar_idx"), "bar_orientation": q.get("bar_orientation"),
                   "vmin_read": q.get("vmin"), "vmax_read": q.get("vmax"), "unit_read": q.get("unit"),
                   "valid_pixel_ratio": q.get("valid_pixel_ratio"), "n_anchors": q.get("n_anchors"),
                   "bar_reason": None if bar is None or bar.get("ok") else bar.get("reason"),
                   "hint_raw_file": "; ".join(h.get("raw_file", "") for h in fig_hints if h.get("raw_file")),
                   "hint_channel": "; ".join(h.get("channel", "") for h in fig_hints if h.get("channel")),
                   "hint_postprocessing": "; ".join(h.get("postprocessing", "") for h in fig_hints
                                                    if h.get("postprocessing")),
                   "hint_note": hint_note}
            flags = []
            if p.get("value_map") is None:
                row["match_status"] = "not_quantified"
            elif bank is None:
                row["match_status"] = "no_raw"
            else:
                vm, valid = p["value_map"].astype(np.float64), p["valid_mask"].astype(bool)
                fb, bb = tm.parse_unit(q.get("unit"))
                span_si = (q["vmax"] - q["vmin"]) * fb if fb else None
                tmatch = time.time()
                mres = tm.match_panel(bank, vm, valid, params, allowed=allowed, unit=(fb, bb), span_si=span_si)
                b = mres["best"]
                row["t_match_s"] = round(time.time() - tmatch, 2)
                row["n_candidates"] = int(len(bank.cands) if allowed is None else allowed.sum())
                row["crop_searched"] = mres["crop_searched"]
                if b is None:
                    row["match_status"] = "unmatched"
                else:
                    L = lib.layers[b["lid"]]
                    # near-tied processing variants: the one with the best absolute agreement
                    # (only possible with compatible units; otherwise the plain-r order is kept)
                    variant = b["variant"]
                    if len(b["variant_ties"]) > 1 and bb is not None and bb == L.unit:
                        best_v = None
                        for v in b["variant_ties"]:
                            reg = tm.region_of(lib, b["lid"], v, b["orientation"], b["crop"])
                            sc = score_panel(reg / fb, vm, valid, q["vmin"], q["vmax"])
                            if sc is not None and (best_v is None or sc["nMAE_pct"] < best_v[0]):
                                best_v = (sc["nMAE_pct"], v)
                        if best_v is not None:
                            variant = best_v[1]
                    rf, rc = mres["runner_file"], mres["runner_channel"]
                    margin = b["score_ident"] - rf["score"] if rf else np.nan
                    cmargin = b["score_ident"] - rc["score"] if rc else np.nan
                    score, contrast = b["score_ident"], mres["contrast"]
                    status = "matched" if (score >= params.min_score and contrast >= params.min_contrast) else \
                        ("weak" if score >= params.min_score or contrast >= params.min_contrast else "unmatched")
                    fov = "full" if (b["crop"] is None or b["crop_frac"] >= 0.95) else "crop"
                    row.update({"match_status": status, "match_file": L.rel, "match_path": str(L.file),
                                "match_channel": L.channel, "match_variant": variant,
                                "variant_ties": ",".join(b["variant_ties"]),
                                "match_orientation": b["orientation"],
                                "match_orientation_desc": tm.IBW_ORIENT_DESC.get(b["orientation"]) if L.fmt == "ibw" else "",
                                "match_fov": fov, "crop_box": _fmt_box(b["crop"]) if b["crop"] else "",
                                "crop_frac": b["crop_frac"], "raw_shape": f"{L.shape[0]}x{L.shape[1]}",
                                "match_score": score, "match_r_plain": b["r_plain"], "match_null": mres["null"],
                                "match_contrast": contrast,
                                "runner_up_file": rf["file"] if rf else None,
                                "runner_up_channel": rf["channel"] if rf else None,
                                "runner_up_score": rf["score"] if rf else None, "margin": margin,
                                "ambiguous_file": bool(np.isfinite(margin) and margin < params.ambiguous_margin),
                                "runner_up_channel_same_file": rc["channel"] if rc else None,
                                "channel_margin": cmargin,
                                "ambiguous_channel": bool(np.isfinite(cmargin) and cmargin < params.channel_margin),
                                "raw_unit": L.unit, "raw_unit_source": L.unit_source,
                                "ar_colormap": L.meta.get("ar_colormap")})
                    if row["ambiguous_file"]:
                        flags.append("ambiguous_file")
                    if row["ambiguous_channel"]:
                        flags.append("ambiguous_channel")
                    if fov == "crop":
                        flags.append("crop_match")
                    if status != "matched":
                        flags.append(f"match_{status}")
                    if p["status"] != "ok":
                        flags.append(f"panel_{p['status']}")
                    region = tm.region_of(lib, b["lid"], variant, b["orientation"], b["crop"])
                    # the AR display range in the note refers to the unprocessed channel
                    met, mflags, gt_shown = compute_metrics(region, L.unit, vm, valid, q["vmin"], q["vmax"],
                                                            q.get("unit"), L.meta if variant == "raw" else {})
                    row.update(met)
                    flags += mflags
                    if row["ambiguous_channel"] and rc is not None:
                        # metrics if the runner-up channel (typically trace <-> retrace) were the shown one
                        L2 = lib.layers[rc["lid"]]
                        reg2 = tm.region_of(lib, rc["lid"], rc["variant"], rc["orientation"], rc["crop"])
                        m2, _, _ = compute_metrics(reg2, L2.unit, vm, valid, q["vmin"], q["vmax"], q.get("unit"), L2.meta)
                        row.update({"alt_channel": f"{L2.channel} ({rc['variant']})",
                                    "alt_channel_nMAE_pct": m2.get("nMAE_pct"),
                                    "alt_channel_median_dev_range_pct": m2.get("median_dev_range_pct")})
                    np.savez_compressed(out / "recon" / f"{paper}__Fig{label}__p{i}.npz",
                                        value_map=vm.astype(np.float32), valid=valid,
                                        raw_region=(gt_shown if gt_shown is not None else region).astype(np.float32))
                    details.append({"paper_key": paper, "figure": label, "panel_idx": i, "panel_pos": pos[i],
                                    "best": {k: b[k] for k in ("file", "channel", "variant", "orientation", "mode",
                                                               "crop", "crop_frac", "score", "r_plain")},
                                    "null": mres["null"], "contrast": contrast,
                                    "best_full_score": mres["best_full_score"],
                                    "best_crop_score": mres["best_crop_score"],
                                    "top": [{k: h[k] for k in ("file", "channel", "variant", "orientation", "mode",
                                                               "crop_frac", "score", "r_plain", "coarse")}
                                            for h in mres["table"][:10]]})
                    if gt_shown is not None:
                        v_al, m_al, _, _ = aligned(gt_shown, vm, valid)
                        mx0, my0, mx1, my1 = p["map_crop_box"]
                        sheet_items.append({"row": row, "crop_rgb": img[my0:my1, mx0:mx1],
                                            "recon": v_al, "raw": gt_shown, "mask": m_al,
                                            "strip": bar.get("strip") if bar is not None else None})
            row["flags"] = ";".join(flags)
            rows.append(row)
            rows_by_idx[i] = row
        title = f"{paper}  Fig. {label}  ({f['source']})"
        Image.fromarray(draw_match_overlay(img, res, rows_by_idx, title)).save(out / "overlays" / f"{paper}__Fig{label}.png")
    unused = sorted({norm_fig(h.get("figure")) for h in hint_rows if h.get("figure")} - used_hint_figs)
    sheets = contact_sheets(paper, sheet_items, out / "contact_sheets") if sheet_items else []
    info = {"pdf": str(pdf_or_dir), "n_figures": len(figs), "extraction_error": err,
            "figure_sources": pd.Series([f["source"] for f in figs]).value_counts().to_dict() if figs else {},
            "raw_roots": [str(r) for r in roots],
            "raw_files": len({str(L.file) for L in lib.layers}) if lib else 0,
            "raw_layers": len(lib.layers) if lib else 0, "candidates": len(bank.cands) if bank else 0,
            "raw_skipped": lib.skipped if lib else {}, "hint_figures_not_found": unused,
            "contact_sheets": [str(s) for s in sheets], "runtime_s": round(time.time() - t0, 1)}
    if unused:
        log(f"  hints name figure(s) {unused} that were not extracted")
    return rows, details, info


METRICS = ["nMAE_pct", "nMAE_IQR_pct", "abs_median_dev_range_pct", "SSIM", "pearson_r", "match_score",
           "nMAE_pct_fit", "nMAE_pct_offset", "nMAE_pct_clipaware"]
CSV_ORDER = [
    # panel and pipeline result
    "paper_key", "figure", "page", "panel_pos", "panel_idx", "status", "match_status", "flags",
    "vmin_read", "vmax_read", "unit_read", "valid_pixel_ratio", "n_anchors", "bar_orientation", "bar_reason",
    # match
    "match_file", "match_channel", "match_variant", "match_orientation", "match_orientation_desc", "match_fov",
    "crop_box", "crop_frac", "raw_shape", "match_score", "match_contrast", "match_null", "match_r_plain",
    "runner_up_file", "runner_up_channel", "runner_up_score", "margin", "ambiguous_file",
    "runner_up_channel_same_file", "channel_margin", "ambiguous_channel", "variant_ties",
    # units and colour range
    "raw_unit", "raw_unit_source", "unit_status", "metric_basis", "unit_read_si", "raw_unit_shown",
    "raw_min", "raw_max", "raw_p1", "raw_p99", "span_ratio", "span_ratio_p1p99", "range_type",
    "clip_below_frac", "clip_above_frac", "ar_display_lo", "ar_display_hi", "ar_colormap",
    # agreement (primary basis = absolute units if compatible, else gain/offset fitted)
    "nMAE_pct", "nMAE_IQR_pct", "median_dev_range_pct", "abs_median_dev_range_pct", "mean_dev_range_pct",
    "SSIM", "pearson_r", "align_shift_px", "valid_pixel_ratio_rawgrid",
    "fit_gain", "fit_offset", "iqr_ratio", "nMAE_pct_fit", "nMAE_IQR_pct_fit", "median_dev_range_pct_fit",
    "abs_median_dev_range_pct_fit", "mean_dev_range_pct_fit", "SSIM_fit",
    "offset_applied", "nMAE_pct_offset", "median_dev_range_pct_offset", "nMAE_pct_clipaware",
    "alt_channel", "alt_channel_nMAE_pct", "alt_channel_median_dev_range_pct",
    # provenance
    "figure_source", "figure_image", "caption", "map_box", "bar_idx", "match_path", "n_candidates",
    "crop_searched", "t_match_s", "hint_raw_file", "hint_channel", "hint_postprocessing", "hint_note"]


def summarize(df, infos, args, runtime):
    def block(d):
        out = {"n_panels": int(len(d)),
               "status": d["status"].value_counts().to_dict() if len(d) else {},
               "match_status": d["match_status"].value_counts().to_dict() if "match_status" in d and len(d) else {}}
        m = d[(d.get("match_status") == "matched") & (d["status"] == "ok")] if len(d) and "match_status" in d else d.iloc[0:0]
        out["n_matched_ok"] = int(len(m))
        if len(m):
            out["metric_basis"] = m["metric_basis"].value_counts().to_dict() if "metric_basis" in m else {}
            out["medians_matched"] = {k: float(m[k].median()) for k in METRICS if k in m and m[k].notna().any()}
            ab = m[m["metric_basis"] == "absolute"] if "metric_basis" in m else m.iloc[0:0]
            if len(ab):
                out["medians_absolute_units"] = {k: float(ab[k].median()) for k in METRICS
                                                 if k in ab and ab[k].notna().any()}
            for flag in ("ambiguous_file", "ambiguous_channel"):
                if flag in m:
                    out[f"n_{flag}"] = int(m[flag].fillna(False).astype(bool).sum())
            out["n_crop_matches"] = int((m.get("match_fov") == "crop").sum()) if "match_fov" in m else 0
        return out
    summ = {"settings": {k: str(v) for k, v in vars(args).items()}, "runtime_s": round(runtime, 1),
            "overall": block(df) if len(df) else {"n_panels": 0},
            "papers": {}}
    for paper, info in infos.items():
        d = df[df["paper_key"] == paper] if len(df) else df
        summ["papers"][paper] = {**info, **(block(d) if len(d) else {"n_panels": 0})}
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdfs", type=Path, default=DEFAULTS["pdfs"], help="folder with <paper_key>.pdf")
    ap.add_argument("--raw", type=Path, default=DEFAULTS["raw"], help="folder with <paper_key>/ raw folders")
    ap.add_argument("--hints", type=Path, default=DEFAULTS["hints"], help="optional collection CSV")
    ap.add_argument("--extra-raw", type=Path, nargs="*", default=[], help="raw folders searched for every paper")
    ap.add_argument("--weights", type=Path, default=DEFAULTS["weights"])
    ap.add_argument("--out", type=Path, default=DEFAULTS["out"])
    ap.add_argument("--papers", nargs="*", default=None, help="process only these paper keys")
    ap.add_argument("--pdffigures2", type=Path, default=DEFAULTS["pdffigures2"])
    ap.add_argument("--java", default="java")
    ap.add_argument("--dpi", type=int, default=150, help="figure rasterisation dpi (corpus: 150)")
    ap.add_argument("--conf", type=float, default=0.5, help="detector confidence (pipeline default)")
    ap.add_argument("--color-space", default="cielab", choices=["cielab", "rgb"])
    ap.add_argument("--no-crop-search", action="store_true", help="match full scans only")
    ap.add_argument("--ambiguous-margin", type=float, default=0.05,
                    help="score margin to the best other FILE below which a match is flagged ambiguous")
    ap.add_argument("--channel-margin", type=float, default=0.03,
                    help="score margin to the best other channel of the same file (trace/retrace ...)")
    args = ap.parse_args()
    t_start = time.time()
    papers = {}
    if args.pdfs.exists():
        for p in sorted(args.pdfs.iterdir()):
            if p.suffix.lower() == ".pdf":
                papers[p.stem] = p
            elif p.is_dir() and any(q.suffix.lower() in IMG_EXT for q in p.iterdir()):
                papers.setdefault(p.name, p)
    if args.papers:
        papers = {k: v for k, v in papers.items() if k in set(args.papers)}
    if not papers:
        log(f"no <paper_key>.pdf (or <paper_key>/ folder of figure images) found in {args.pdfs}")
        return
    missing_raw = [k for k in papers if raw_root_for(args.raw, k) is None]
    if missing_raw and not args.extra_raw:
        log(f"warning: no raw folder {args.raw}/<paper_key> for {missing_raw}: "
            "their panels are quantified but not matched")
    for sub in ("figures", "overlays", "contact_sheets", "recon"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    hints = load_hints(args.hints)
    params = tm.MatchParams(ambiguous_margin=args.ambiguous_margin, channel_margin=args.channel_margin)
    if args.no_crop_search:
        params.crop_trigger, params.crop_skip_contrast = -1.0, -1.0
    pipe = FigurePipeline(args.weights, conf=args.conf, color_space=args.color_space, ocr=ColorbarOCR())
    all_rows, all_details, infos = [], [], {}
    for paper, src in papers.items():
        log(f"\n=== {paper} ({src.name}) ===")
        rows, details, info = evaluate_paper(paper, src, args, pipe, params, hints)
        all_rows += rows
        all_details += details
        infos[paper] = info
    df = pd.DataFrame(all_rows)
    first = [c for c in CSV_ORDER if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(args.out / "tier3_panels.csv", index=False, encoding="utf-8-sig")
    (args.out / "tier3_matches.json").write_text(json.dumps(all_details, indent=1, default=float), encoding="utf-8")
    runtime = time.time() - t_start
    summ = summarize(df, infos, args, runtime)
    (args.out / "tier3_summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")

    log("\n" + "=" * 72)
    log("TIER 3 SUMMARY")
    log("=" * 72)
    if len(df):
        cols = ["paper_key", "figure", "panel_pos", "status", "match_status", "match_file", "match_channel",
                "match_variant", "match_orientation", "match_fov", "match_score", "margin", "nMAE_pct",
                "median_dev_range_pct", "metric_basis", "flags"]
        log(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        ov = summ["overall"]
        log(f"\npanels: {ov['n_panels']}  status: {ov['status']}  match: {ov.get('match_status')}")
        if ov.get("medians_matched"):
            log("medians (matched, ok): " + ", ".join(f"{k}={v:.3f}" for k, v in ov["medians_matched"].items()))
    log(f"\nruntime {runtime:.0f}s. Saved: {args.out / 'tier3_panels.csv'}, tier3_summary.json, overlays/, contact_sheets/")


if __name__ == "__main__":
    main()
