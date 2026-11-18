"""Bar-level evaluation of the colour-bar reader against reader_dev/bar_truth.csv plus the 'pipeline-confirmed' bars
(bars used for quantification in corpus figures whose read scales the first-pass rating judged correct; README.md).

    python eval_reader.py --reader v1|v2 [--only S0911,S0979] [--out eval_v1.csv]

Per bar: readable, range error = max(|vmin_read - vmin|, |vmax_read - vmax|) / span, within tolerance, unit ok.
Summary for the truth-table bars (hard cases), the confirmed bars (regression) and all."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
from afmquant.pipeline import FigurePipeline  # noqa: E402
from afmquant.ocr import ColorbarOCR, canonical_unit  # noqa: E402

YS = ROOT / "revision" / "detection" / "yield_sample"
DEV = ROOT / "revision" / "reader_dev"
W = ROOT / "runs" / "detect" / "runs" / "afm_detect" / "v13" / "weights" / "best.pt"
MU = "\u00b5"


def norm_unit(u):
    return (u or "").replace("\u03bc", MU).strip().lower()


def confirmed_bars():
    """Bars whose reading the first-pass rating judged correct: figures with B_scale_ok == y, bars used by quantified panels."""
    d = pd.read_excel(YS / "verification_reviewed_20261021.xlsx", sheet_name="verify", dtype=str).fillna("")
    d["n_panels_ok"] = pd.to_numeric(d.n_panels_ok)
    rows = []
    for sid in d[(d.n_panels_ok > 0) & (d.B_scale_ok == "y")].sample_id:
        j = json.load(open(YS / "out" / f"{sid}.json", encoding="utf-8"))
        for bi in sorted({p["bar_idx"] for p in j["panels"] if p["status"] == "ok"}):
            b = j["bars"][bi]
            rows.append(dict(sid=sid, bar=bi, vmin=b["vmin"], vmax=b["vmax"], unit=canonical_unit(b.get("unit")) or "",
                             tol=0.10, source="pipeline-confirmed", note=""))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="v1", choices=["v1", "v2"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--debug", action="store_true", help="print OCR items and anchors of every truth-table bar")
    a = ap.parse_args()
    truth = pd.read_csv(REPO / "data" / "corpus_sample" / "bar_truth.csv", dtype={"unit": str, "note": str}, encoding="utf-8").fillna({"unit": "", "note": ""})
    conf = confirmed_bars()
    conf = conf[~conf.set_index(["sid", "bar"]).index.isin(truth.set_index(["sid", "bar"]).index)]
    allb = pd.concat([truth, conf], ignore_index=True)
    if a.only:
        allb = allb[allb.sid.isin(a.only.split(","))]
    pipe = FigurePipeline(W, ocr=ColorbarOCR(), unit_mode="lexicon", reader=a.reader)
    out, cache = [], {}
    for r in allb.itertuples():
        if r.sid not in cache:
            img = np.array(Image.open(YS / "imgs" / f"{r.sid}.png").convert("RGB"))
            maps, bars_det = pipe.detect(img)
            cache[r.sid] = pipe.read_bars(img, bars_det, maps=maps)
        bars = cache[r.sid]
        bi = int(r.bar)
        base = dict(sid=r.sid, bar=bi, source=r.source)
        if bi >= len(bars):
            out.append({**base, "readable": False, "err": np.nan, "ok": False, "unit_ok": np.nan, "read": "(no such bar)"}); continue
        b = bars[bi]
        if a.debug:
            print(f"  DEBUG {r.sid} b{bi}: column={b.get('label_column')} border={b.get('touches_border')} anchors={b.get('anchors')} "
                  f"unit_raw={b.get('unit_raw')} passes={b.get('n_passes')} reason={b.get('reason')}\n         labels={b.get('label_debug')}\n         geom={b.get('geom_debug')}")
        if r.source == "not_a_bar":
            out.append({**base, "readable": bool(b.get("ok")), "err": np.nan, "ok": not b.get("ok"), "unit_ok": np.nan,
                        "read": (f"{b.get('vmin'):.4g}..{b.get('vmax'):.4g} {b.get('unit')} (should be rejected)" if b.get("ok") else "unread (correct)")})
            continue
        if not b.get("ok"):
            out.append({**base, "readable": False, "err": np.nan, "ok": False, "unit_ok": np.nan, "read": f"unread: {b.get('reason')}"}); continue
        vmin_t, vmax_t = float(r.vmin), float(r.vmax)
        span = vmax_t - vmin_t
        err = max(abs(b["vmin"] - vmin_t), abs(b["vmax"] - vmax_t)) / span
        u_t = norm_unit(r.unit); u_r = norm_unit(canonical_unit(b.get("unit")))
        out.append({**base, "readable": True, "err": err, "ok": err <= float(r.tol), "unit_ok": (u_r == u_t) if u_t else np.nan,
                    "read": f"{b['vmin']:.4g}..{b['vmax']:.4g} {b.get('unit')}  [truth {r.vmin}..{r.vmax} {r.unit}]"})
    t = pd.DataFrame(out)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 80)
    print(t[["sid", "bar", "source", "readable", "err", "ok", "unit_ok", "read"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for name, sub in [("TRUTH-TABLE (hard cases)", t[t.source != "pipeline-confirmed"]),
                      ("CONFIRMED (regression)", t[t.source == "pipeline-confirmed"]), ("ALL", t)]:
        real, nab = sub[sub.source != "not_a_bar"], sub[sub.source == "not_a_bar"]
        line = (f"\n[{a.reader}] {name}: bars {len(real)} | readable {int(real.readable.sum())} | within tol {int(real.ok.sum())} "
                f"({100 * real.ok.mean():.0f}%) | unit ok {int(real.unit_ok.dropna().astype(bool).sum())}/{int(real.unit_ok.notna().sum())}")
        if len(nab):
            line += f" | not-a-bar rejected {int(nab.ok.sum())}/{len(nab)}"
        print(line)
    if a.out:
        t.to_csv(a.out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
