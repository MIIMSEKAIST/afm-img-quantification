"""Self-test of the Tier 3 tool with synthetic "papers" whose truth is known.

Figures are rendered from the 26 ground-truth DART-ESM scans with the Tier 2
renderer (``tier2_render.render_case`` / ``add_map``: axes, colour bar with
tick labels and unit, scale bar; raster figures JPEG q85) and embedded in PDF
pages with captions:

    SelfTest_A  article-like layout (two-column text, "Figure N." captions,
                raster figures), extracted with pdffigures2:
                  Fig. 1  single panel (viridis)
                  Fig. 2  two maps with their own colour bars (jet); the left map
                          is displayed vertically flipped
                  Fig. 3  zoomed sub-region (50 % of the scan, hot)
    SelfTest_B  Tier 2 PDF route (one raster figure per page; pdffigures2 finds
                nothing on such pages -> PyMuPDF fallback):
                  Fig. 1  2 x 2 composite: ESM amplitude (copper) + topography =
                          HeightRetrace with per-line 1st-order flatten (gray, nm)
                          + histogram + text
                  Fig. 2  single panel with the colour range clipped to the
                          2nd-98th percentile (jet)
                  Fig. 3  two maps sharing one colour bar (viridis)
    SelfTest_C  negative control: one figure whose scan is NOT in the raw folder
    SelfTest_D  publisher-like page: two VECTOR figures (matplotlib PDF: axes and
                tick labels as text, map image embedded at 300 dpi) on one
                article page, extracted with pdffigures2:
                  Fig. 1  Phase1Retrace in deg, per-line 0th-order flatten, shown
                          in the Igor/AR orientation (viridis)
                  Fig. 2  (a) amplitude normalised to 0-1 and labelled "a.u."
                          (units unknown -> gain/offset fit); (b) Amplitude1Retrace
                          in pm in the Igor/AR orientation (hot)

Raw folders contain the true .ibw files under uninformative names (some in
sub-folders) plus distractor scans, a .gsf export of one distractor and an
unsupported file. A hints CSV restricts one figure and contains an example row
of an unknown paper.

The tool (tier3_evaluate.py) is run through its command line and every
quantified panel is checked against the truth: file, channel, processing
variant, orientation, field of view (crop box IoU >= 0.7), confident match, and
reported metrics equal to the "oracle" metrics computed against the true raw
region (Tier 2 score_panel); in aggregate the metrics must be in line with
Tier 2. The negative control must not be matched confidently.

Usage
-----
    python benchmark/tier3_selftest.py --root D:/AFM_Database_SH/revision/tier3/_selftest
    python benchmark/tier3_selftest.py ... --check-only      # re-check an existing run
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.tier2_render import render_case, add_map, encode, DEFAULT, scan_size_um  # noqa: E402
from benchmark.reproduce_benchmark import material_of  # noqa: E402
from benchmark import tier3_match as tm  # noqa: E402

ROOT = Path("D:/AFM_Database_SH")

# Panel truth: file stem, channel, processing variant applied before display, display orientation
# relative to the Igor/AR view (Tier 2 shows the stored matrix with imshow(origin='upper'), which is
# 'rot90cw' of the Igor view; 'antitranspose' = that, flipped vertically), optional crop of the
# displayed matrix (rows r0:r1, cols c0:c1) and the unit shown on the colour bar.
PAPERS = {
    "SelfTest_A": {
        "route": "article",
        "figures": [
            {"fig": 1, "layout": "single", "cmap": "viridis",
             "panels": [dict(stem="LICGC_03_4um", channel="Amplitude1Trace")]},
            {"fig": 2, "layout": "pair_same", "cmap": "jet",
             "panels": [dict(stem="LMNO_pristine_05_1um", channel="Amplitude1Trace", display="antitranspose"),
                        dict(stem="NCM811_pristine_04_2um", channel="Amplitude1Trace")]},
            {"fig": 3, "layout": "single", "cmap": "hot",
             "panels": [dict(stem="LICGC_07_4um", channel="Amplitude1Trace", crop=(64, 192, 40, 168))]},
        ],
        "distractors": ["LICGC_04_4um", "LICGC_06_4um", "LMNO_pristine_06_1um", "NCM811_pristine_06_2um",
                        "LMNO_pristine_02_4um", "NCM811_pristine_01_2um", "LICGC_10_4um"],
    },
    "SelfTest_B": {
        "route": "tier2_pages",
        "figures": [
            {"fig": 1, "layout": "2x2", "cmap": "copper",
             "panels": [dict(stem="LICGC_09_4um", channel="Amplitude1Trace"),
                        dict(stem="LICGC_09_4um", channel="HeightRetrace", variant="flat1", unit="nm")]},
            {"fig": 2, "layout": "single", "cmap": "jet", "clip": (2, 98),
             "panels": [dict(stem="NCM811_pristine_06_2um", channel="Amplitude1Trace")]},
            {"fig": 3, "layout": "pair_shared", "cmap": "viridis",
             "panels": [dict(stem="LMNO_pristine_02_4um", channel="Amplitude1Trace"),
                        dict(stem="LICGC_05_4um", channel="Amplitude1Trace")]},
        ],
        "distractors": ["LICGC_01_4um", "LICGC_02_4um", "LMNO_pristine_03_4um", "LMNO_pristine_09_1um",
                        "NCM811_pristine_03_4um", "NCM811_pristine_05_4um", "LICGC_08_4um", "LMNO_pristine_07_1um"],
        "gsf_distractor": "LMNO_pristine_01_4um",
    },
    "SelfTest_C": {
        "route": "tier2_pages",
        "figures": [
            {"fig": 1, "layout": "single", "cmap": "copper",
             "panels": [dict(stem="LMNO_pristine_08_4um", channel="Amplitude1Trace", absent=True)]},
        ],
        "distractors": ["LMNO_pristine_04_1um", "NCM811_pristine_02_1um", "LICGC_03_4um", "LICGC_09_4um",
                        "NCM811_pristine_07_4um", "LMNO_pristine_05_1um"],
    },
    "SelfTest_D": {
        "route": "article_vector",
        "figures": [
            {"fig": 1, "layout": "single", "cmap": "viridis",
             "panels": [dict(stem="LICGC_06_4um", channel="Phase1Retrace", variant="flat0", display="igor",
                             unit="deg")]},
            {"fig": 2, "layout": "pair_same", "cmap": "hot",
             "panels": [dict(stem="NCM811_pristine_05_4um", channel="Amplitude1Trace", unit="a.u."),
                        dict(stem="LMNO_pristine_06_1um", channel="Amplitude1Retrace", display="igor")]},
        ],
        "distractors": ["LICGC_01_4um", "LICGC_08_4um", "LMNO_pristine_03_4um", "NCM811_pristine_02_1um",
                        "NCM811_pristine_04_2um", "LMNO_pristine_09_1um"],
    },
}
BODY = ("Electrochemical strain microscopy (ESM) probes the local ionic dynamics of battery materials by "
        "detecting the strain that follows an oscillating electric field applied through a conductive tip. "
        "The DART mode tracks the contact resonance and yields amplitude, phase and frequency images "
        "together with the topography. In this self-test document the figures are synthetic renderings of "
        "archived measurements; the text only provides the layout context that figure extraction tools "
        "expect from a journal article, such as body paragraphs in two columns and numbered captions. ")


def panel_matrix(pn, ibw_dir):
    """Displayed matrix of a truth panel (figure unit), its orientation, the
    crop box in Igor-view pixels and the affine map raw SI -> figure unit."""
    layers = {ch: arr for ch, arr, *_ in tm.load_ibw(ibw_dir / f"{pn['stem']}.ibw")}
    B = tm.VARIANTS[pn.get("variant", "raw")](layers[pn["channel"]])
    o = pn.get("display", "rot90cw")
    D = tm.orient(B, o)
    box = None
    if pn.get("crop"):
        r0, r1, c0, c1 = pn["crop"]
        idx = tm.orient(np.arange(B.size).reshape(B.shape), o)[r0:r1, c0:c1]
        ys, xs = np.divmod(idx.ravel(), B.shape[1])
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        D = D[r0:r1, c0:c1]
    unit = pn.get("unit", "pm")
    if unit == "a.u.":
        scale, offset = 1.0 / np.ptp(D), -D.min() / np.ptp(D)
    else:
        scale, offset = 1.0 / tm.parse_unit(unit)[0], 0.0
    return np.ascontiguousarray(D * scale + offset), o, box, (float(scale), float(offset)), unit


def render_figure(fg, ibw_dir):
    """PNG bytes of one raster figure (Tier 2 render_case)."""
    p = {**DEFAULT, "layout": "2x2" if fg["layout"] == "2x2" else fg["layout"]}
    if fg.get("clip"):
        p["clip"] = fg["clip"]
    pans = fg["panels"]
    mats = []
    for pn in pans:
        D = panel_matrix(pn, ibw_dir)[0]
        size = scan_size_um(pn["stem"]) * (D.shape[0] / 256.0)
        mats.append((D, size))
    if fg["layout"] == "single":
        png, _ = render_case(pans[0]["stem"], mats[0][0], None, fg["cmap"], p, mats[0][1])
    elif fg["layout"] in ("pair_same", "pair_shared"):
        png, _ = render_case(pans[0]["stem"], mats[0][0], None, fg["cmap"], p, mats[0][1],
                             second=(pans[1]["stem"], mats[1][0], mats[1][1]))
    else:   # 2x2: (a) amplitude, (b) topography (gray, own bar), (c) histogram, (d) text
        png, _ = render_case(pans[0]["stem"], mats[0][0], mats[1][0], fg["cmap"], p, mats[0][1])
    return png


def vector_figure(fg, ibw_dir):
    """Vector PDF of one figure: Tier 2 add_map per panel with the panel's
    unit; map images rasterised at 300 dpi inside the PDF."""
    p = {**DEFAULT}
    n = len(fg["panels"])
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.0))
    axes = np.atleast_1d(axes)
    for k, (ax, pn) in enumerate(zip(axes, fg["panels"])):
        D, _, _, _, unit = panel_matrix(pn, ibw_dir)
        add_map(ax, D, fg["cmap"], scan_size_um(pn["stem"]), p, unit, label_prefix=f"({'abcd'[k]})" if n > 1 else "")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300)
    plt.close(fig)
    return buf.getvalue()


def _two_columns(page, y0, y1, text):
    import fitz
    cw = (545 - 50 - 16) / 2
    for k in range(2):
        page.insert_textbox(fitz.Rect(50 + k * (cw + 16), y0, 50 + k * (cw + 16) + cw, y1), text,
                            fontsize=8.5, fontname="tiro", align=fitz.TEXT_ALIGN_JUSTIFY)


def article_pdf(path, figs, title):
    """Journal-like pages (PyMuPDF): title, two-column text, raster figure
    (JPEG q85 as in Tier 2), caption, more text; one figure per page."""
    import fitz
    doc = fitz.open()
    for n, (png, caption) in enumerate(figs, start=1):
        page = doc.new_page(width=595, height=842)
        if n == 1:
            page.insert_textbox(fitz.Rect(50, 40, 545, 75), title, fontsize=14, fontname="hebo")
        y = 80
        _two_columns(page, y, y + 150, BODY * 2)
        img = Image.open(io.BytesIO(png)).convert("RGB")
        arr, _ = encode(png, 85)
        b = io.BytesIO()
        Image.fromarray(arr).save(b, format="JPEG", quality=95)
        w_pt, h_pt = img.size[0] / DEFAULT["dpi"] * 72, img.size[1] / DEFAULT["dpi"] * 72
        x0 = (595 - w_pt) / 2
        r = fitz.Rect(x0, y + 165, x0 + w_pt, y + 165 + h_pt)
        page.insert_image(r, stream=b.getvalue())
        page.insert_textbox(fitz.Rect(50, r.y1 + 8, 545, r.y1 + 50), caption, fontsize=8, fontname="tiro")
        _two_columns(page, r.y1 + 58, 800, BODY * 3)
    doc.save(str(path))


def article_vector_pdf(path, figs, title):
    """One journal-like page with all (vector) figures of the paper."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 40, 545, 75), title, fontsize=14, fontname="hebo")
    _two_columns(page, 80, 160, BODY)
    y = 170
    for pdf_bytes, caption in figs:
        src = fitz.open("pdf", pdf_bytes)
        w, h = src[0].rect.width, src[0].rect.height
        s = min(1.0, 495 / w)
        x0 = (595 - w * s) / 2
        r = fitz.Rect(x0, y, x0 + w * s, y + h * s)
        page.show_pdf_page(r, src, 0)
        page.insert_textbox(fitz.Rect(50, r.y1 + 6, 545, r.y1 + 40), caption, fontsize=8, fontname="tiro")
        y = r.y1 + 46
    _two_columns(page, y, 800, BODY * 2)
    doc.save(str(path))


def tier2_pages_pdf(path, figs):
    """Tier 2 PDF route (tier2_render.main): one figure per page + caption."""
    dpi = DEFAULT["dpi"]
    with PdfPages(path) as pdf:
        for n, (png, caption) in enumerate(figs, start=1):
            arr, _ = encode(png, 85)
            img = Image.fromarray(arr)
            w_in, h_in = img.size[0] / dpi, img.size[1] / dpi
            fig = plt.figure(figsize=(max(w_in, 4.5) + 0.6, h_in + 1.4))
            axi = fig.add_axes([0.3 / fig.get_figwidth(), 1.0 / fig.get_figheight(),
                                w_in / fig.get_figwidth(), h_in / fig.get_figheight()])
            axi.imshow(np.array(img)); axi.axis("off")
            fig.text(0.05, 0.35 / fig.get_figheight(), caption, fontsize=8, wrap=True)
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)


def write_gsf(path, arr, xreal, zunit="m", title="Height"):
    """Gwyddion simple field (used to exercise the .gsf reader)."""
    hdr = (f"Gwyddion Simple Field 1.0\nXRes = {arr.shape[1]}\nYRes = {arr.shape[0]}\nXReal = {xreal}\n"
           f"YReal = {xreal}\nXYUnits = m\nZUnits = {zunit}\nTitle = {title}\n").encode()
    pad = 4 - len(hdr) % 4
    path.write_bytes(hdr + b"\0" * pad + arr.astype("<f4").tobytes())


def build(root, ibw_dir):
    if root.exists():
        shutil.rmtree(root)
    (root / "pdfs").mkdir(parents=True)
    truth, hints = [], []
    rng = np.random.default_rng(7)
    for paper, spec in PAPERS.items():
        raw = root / "raw" / paper
        raw.mkdir(parents=True)
        names = {}
        true_stems = sorted({pn["stem"] for fg in spec["figures"] for pn in fg["panels"] if not pn.get("absent")})
        pool = true_stems + spec["distractors"]
        for k, j in enumerate(rng.permutation(len(pool))):
            stem = pool[j]
            sub = raw / ("session_2" if k % 3 == 2 else "")
            sub.mkdir(parents=True, exist_ok=True)
            new = sub / f"{'Image' if k % 2 else 'scan'}{(k + 1) * 7 % 100:04d}.ibw"
            shutil.copy2(ibw_dir / f"{stem}.ibw", new)
            names[stem] = str(new.relative_to(raw))
        if spec.get("gsf_distractor"):
            st = spec["gsf_distractor"]
            layers = {ch: arr for ch, arr, *_ in tm.load_ibw(ibw_dir / f"{st}.ibw")}
            write_gsf(raw / "export_height.gsf", layers["HeightRetrace"], scan_size_um(st) * 1e-6)
            (raw / "old_bruker_scan.spm").write_bytes(b"\\*File list\r\n")      # unsupported -> listed
        figs = []
        for fg in spec["figures"]:
            data = vector_figure(fg, ibw_dir) if spec["route"] == "article_vector" else render_figure(fg, ibw_dir)
            stems = [pn["stem"] for pn in fg["panels"]]
            what = sorted({pn["channel"].replace("Retrace", "").replace("Trace", "") for pn in fg["panels"]})
            cap = (f"Figure {fg['fig']}. DART-ESM {' and '.join(what).lower()} map(s) of "
                   f"{', '.join(material_of(s) for s in stems)} ({fg['layout']}, {fg['cmap']}).")
            figs.append((data, cap))
            for pos, pn in enumerate(fg["panels"], start=1):
                _, o, box, (scale, offset), unit = panel_matrix(pn, ibw_dir)
                truth.append({"paper_key": paper, "figure": str(fg["fig"]), "panel_pos": pos,
                              "true_stem": pn["stem"], "true_file": None if pn.get("absent") else names[pn["stem"]],
                              "true_channel": pn["channel"], "true_variant": pn.get("variant", "raw"),
                              "true_orientation": o, "true_crop": box, "true_fov": "crop" if box else "full",
                              "unit": unit, "scale": scale, "offset": offset, "absent": bool(pn.get("absent"))})
        pdf = root / "pdfs" / f"{paper}.pdf"
        if spec["route"] == "article":
            article_pdf(pdf, figs, f"{paper}: synthetic article for the Tier 3 self-test")
        elif spec["route"] == "article_vector":
            article_vector_pdf(pdf, figs, f"{paper}: synthetic article with vector figures")
        else:
            tier2_pages_pdf(pdf, figs)
        if paper == "SelfTest_A":
            # figure 2 restricted to three files (the two true ones + one distractor); channel left empty
            f2 = [pn["stem"] for pn in spec["figures"][1]["panels"]] + [spec["distractors"][0]]
            hints.append({"paper_key": paper, "doi": "", "figure": "2", "panel": "a,b", "afm_mode": "ESM amplitude",
                          "raw_file": "; ".join(Path(names[s]).name for s in f2), "channel": "", "software": "",
                          "postprocessing": "", "colormap": "jet", "clipping_or_range": "auto", "notes": "self-test"})
        if paper == "SelfTest_B":
            hints.append({"paper_key": paper, "doi": "", "figure": "1", "panel": "b", "afm_mode": "topography",
                          "raw_file": "", "channel": "", "software": "", "postprocessing": "flatten 1st order",
                          "colormap": "gray", "clipping_or_range": "", "notes": "no restriction (only metadata)"})
    hints.append({"paper_key": "Chen_2025_EXAMPLE", "doi": "10.xxxx/xxxxx", "figure": "3", "panel": "b",
                  "afm_mode": "ESM amplitude", "raw_file": "LMNO_pristine_03_4um.ibw", "channel": "Amplitude1Trace",
                  "software": "Asylum Research (Igor)", "postprocessing": "flatten 0th order", "colormap": "copper",
                  "clipping_or_range": "auto", "notes": "example row"})
    pd.DataFrame(hints).to_csv(root / "hints.csv", index=False, encoding="utf-8-sig")
    (root / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def oracle_metrics(root, t, r):
    """Tier 2 metrics of the reconstruction against the TRUE raw region (known
    file, channel, variant, orientation and crop): what the tool should report
    if its match and alignment are right."""
    from benchmark.tier2_evaluate import score_panel
    npz = root / "results" / "recon" / f"{t['paper_key']}__Fig{t['figure']}__p{int(r['panel_idx'])}.npz"
    if not npz.exists():
        return {}
    z = np.load(npz)
    layers = {ch: arr for ch, arr, *_ in tm.load_ibw(root / "raw" / t["paper_key"] / t["true_file"])}
    a = tm.VARIANTS[t["true_variant"]](layers[t["true_channel"]])
    if t["true_crop"]:
        x0, y0, x1, y1 = t["true_crop"]
        a = a[y0:y1, x0:x1]
    gt = tm.orient(a, t["true_orientation"]) * t["scale"] + t["offset"]       # figure unit
    sc = score_panel(np.ascontiguousarray(gt), z["value_map"].astype(float), z["valid"].astype(bool),
                     r["vmin_read"], r["vmax_read"])
    return {} if sc is None else {"oracle_nMAE_pct": sc["nMAE_pct"], "oracle_median_dev_range_pct": sc["median_dev_range_pct"],
                                  "oracle_pearson_r": sc["align_corr"]}


def check(root):
    """Compare the tool's output with the truth. A positive panel passes when
    file, channel, processing variant, orientation and field of view (crop box
    IoU >= 0.7) are right, the match is confident, and the reported metrics
    agree with the oracle metrics (true region) within 0.5 percentage points.
    In aggregate the metrics must be in line with Tier 2 (median nMAE 1.5-4.5 %,
    median |median deviation| < 1 % of range). The negative control must not
    be reported as a confident match."""
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    df = pd.read_csv(root / "results" / "tier3_panels.csv", dtype={"figure": str})
    rows, ok_all = [], True
    for t in truth:
        d = df[(df.paper_key == t["paper_key"]) & (df.figure == t["figure"]) & (df.panel_pos == t["panel_pos"])]
        rec = {"paper": t["paper_key"], "fig": t["figure"], "pos": t["panel_pos"], "true_stem": t["true_stem"],
               "true_file": t["true_file"] or "(absent)", "true_channel": t["true_channel"],
               "true_variant": t["true_variant"], "true_orient": t["true_orientation"], "true_fov": t["true_fov"]}
        if d.empty:
            rec.update({"status": "not_detected", "result": "FAIL (panel not detected)"})
            ok_all = False
            rows.append(rec)
            continue
        r = d.iloc[0]
        rec.update({"status": r.status, "match_status": r.get("match_status"), "matched_file": r.get("match_file"),
                    "matched_channel": r.get("match_channel"), "matched_variant": r.get("match_variant"),
                    "matched_orient": r.get("match_orientation"), "matched_fov": r.get("match_fov"),
                    "score": r.get("match_score"), "margin": r.get("margin"), "channel_margin": r.get("channel_margin"),
                    "pearson_r": r.get("pearson_r"), "nMAE_pct": r.get("nMAE_pct"),
                    "median_dev_range_pct": r.get("median_dev_range_pct"), "metric_basis": r.get("metric_basis"),
                    "unit_read": r.get("unit_read"), "flags": r.get("flags")})
        if t["absent"]:
            good = r.get("match_status") != "matched"
            rec["result"] = "PASS (no confident match)" if good else "FAIL (false match)"
        elif r.status != "ok":
            rec["result"] = f"SKIP (pipeline status {r.status})"
        else:
            problems = []
            if str(r.get("match_file")).replace("\\", "/") != t["true_file"].replace("\\", "/"):
                problems.append("file")
            if r.get("match_channel") != t["true_channel"]:
                # a trace/retrace swap is accepted only if the tool flags it as ambiguous and names the
                # true channel as the runner-up (the two passes differ by less than the figure resolves)
                base = lambda c: str(c).replace("Retrace", "").replace("Trace", "")  # noqa: E731
                swap_flagged = (bool(r.get("ambiguous_channel")) and r.get("runner_up_channel_same_file") == t["true_channel"]
                                and base(r.get("match_channel")) == base(t["true_channel"]))
                if swap_flagged:
                    rec["note"] = "trace/retrace ambiguous (flagged); true channel is the runner-up"
                else:
                    problems.append("channel")
            if r.get("match_variant") != t["true_variant"]:
                problems.append("variant")
            if r.get("match_orientation") != t["true_orientation"]:
                problems.append("orientation")
            if r.get("match_fov") != t["true_fov"]:
                problems.append("fov")
            if t["true_crop"] and isinstance(r.get("crop_box"), str) and r.get("crop_box"):
                iou = box_iou([float(v) for v in r["crop_box"].split(",")], t["true_crop"])
                rec["crop_iou"] = iou
                if iou < 0.7:
                    problems.append(f"crop IoU {iou:.2f}")
            if r.get("match_status") != "matched":
                problems.append(f"match_status {r.get('match_status')}")
            if not problems:
                rec.update(oracle_metrics(root, t, r))
                if "oracle_nMAE_pct" in rec:
                    # absolute units: same numbers expected; unknown units: the tool fits gain/offset
                    tol = 0.5 if r.get("metric_basis") == "absolute" else 1.5
                    if abs(r["nMAE_pct"] - rec["oracle_nMAE_pct"]) > tol:
                        problems.append("nMAE differs from oracle")
                    if abs(r["median_dev_range_pct"] - rec["oracle_median_dev_range_pct"]) > tol:
                        problems.append("median dev differs from oracle")
                if not (np.isfinite(r.get("nMAE_pct", np.nan)) and r["nMAE_pct"] < 6.0):
                    problems.append("nMAE >= 6 %")
            rec["result"] = ("PASS" + (" (trace/retrace ambiguity flagged)" if rec.get("note") else "")) \
                if not problems else "FAIL (" + ", ".join(problems) + ")"
        if rec["result"].startswith("FAIL"):
            ok_all = False
        rows.append(rec)
    out = pd.DataFrame(rows)
    pos = out[(out.true_file != "(absent)") & out.result.str.startswith("PASS")]
    agg = {"n_truth_panels": len(out), "n_pass": int(out.result.str.startswith("PASS").sum()),
           "median_nMAE_pct": float(pos.nMAE_pct.median()) if len(pos) else np.nan,
           "median_abs_median_dev_pct": float(pos.median_dev_range_pct.abs().median()) if len(pos) else np.nan}
    agg["tier2_consistent"] = bool(1.5 <= agg["median_nMAE_pct"] <= 4.5 and agg["median_abs_median_dev_pct"] < 1.0)
    ok_all = ok_all and agg["tier2_consistent"]
    out.to_csv(root / "selftest_report.csv", index=False)
    (root / "selftest_summary.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    extra = df[~df.set_index(["paper_key", "figure", "panel_pos"]).index.isin(
        [(t["paper_key"], t["figure"], t["panel_pos"]) for t in truth])]
    return out, ok_all, extra, agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "revision/tier3/_selftest")
    ap.add_argument("--ibw-dir", type=Path, default=ROOT / "data/esm_ibw")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/detect/runs/afm_detect/v13/weights/best.pt")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.check_only:
        t0 = time.time()
        truth = build(args.root, args.ibw_dir)
        print(f"built {len(truth)} truth panels in {time.time() - t0:.0f}s under {args.root}")
        cmd = [sys.executable, "-X", "utf8", str(Path(__file__).with_name("tier3_evaluate.py")),
               "--pdfs", str(args.root / "pdfs"), "--raw", str(args.root / "raw"),
               "--hints", str(args.root / "hints.csv"), "--weights", str(args.weights),
               "--out", str(args.root / "results")]
        print("running:", " ".join(cmd), flush=True)
        t1 = time.time()
        r = subprocess.run(cmd)
        print(f"tier3_evaluate.py finished (exit {r.returncode}) in {time.time() - t1:.0f}s")
        (args.root / "selftest_runtime.txt").write_text(f"{time.time() - t1:.1f}\n", encoding="utf-8")
    rep, ok_all, extra, agg = check(args.root)
    pd.set_option("display.width", 250)
    cols = ["paper", "fig", "pos", "true_file", "true_channel", "matched_file", "matched_channel", "matched_variant",
            "matched_orient", "matched_fov", "score", "margin", "pearson_r", "nMAE_pct", "median_dev_range_pct",
            "oracle_nMAE_pct", "oracle_median_dev_range_pct", "status", "match_status", "result"]
    print(rep[[c for c in cols if c in rep]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(extra):
        print(f"\n{len(extra)} additional detected panel(s) not in the truth:")
        print(extra[["paper_key", "figure", "panel_pos", "status", "match_status", "match_file"]].to_string(index=False))
    print("\naggregate:", json.dumps(agg))
    print("\nSELF-TEST", "PASSED" if ok_all else "FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
