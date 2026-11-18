"""Figure-level pairing/scale evaluation of a pipeline configuration on the 88 corpus figures that the submitted
pipeline quantified (first-pass labels: B_pairing_ok, B_scale_ok; README.md), plus the 30 AFM figures with a colour bar that the
submitted pipeline detected but did not quantify (do they become quantified?).

    python eval_pairing_v2.py --reader v2 --pairing evidence --unit lexicon [--out results.csv]

Reports: correct-pairing figures still quantified / panels kept; wrong-pairing figures still quantified / panels
kept; held panels by reason; newly quantified figures among the 30 unquantified AFM figures."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
from afmquant.pipeline import FigurePipeline  # noqa: E402
from afmquant.ocr import ColorbarOCR  # noqa: E402

YS = ROOT / "revision" / "detection" / "yield_sample"
W = ROOT / "runs" / "detect" / "runs" / "afm_detect" / "v13" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="v2", choices=["v1", "v2"])
    ap.add_argument("--pairing", default="evidence", choices=["nearest", "evidence"])
    ap.add_argument("--unit", default="lexicon", choices=["raw", "lexicon"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    d = pd.read_excel(YS / "verification_reviewed_20261021.xlsx", sheet_name="verify", dtype=str).fillna("")
    for c in ["A_n_hbars", "A_n_vbars", "B_tp", "n_panels_ok"]:
        d[c] = pd.to_numeric(d[c])
    quantified = d[d.n_panels_ok > 0]
    unq_afm = d[(d.A_has_afm == "y") & ((d.A_n_hbars + d.A_n_vbars) > 0) & (d.B_tp > 0) & (d.n_panels_ok == 0)]
    pipe = FigurePipeline(W, ocr=ColorbarOCR(), reader=a.reader, pairing_mode=a.pairing, unit_mode=a.unit)
    rows = []
    for grp, sub in (("quantified", quantified), ("unquantified_afm", unq_afm)):
        for r in sub.itertuples():
            img = np.array(Image.open(YS / "imgs" / f"{r.sample_id}.png").convert("RGB"))
            res = pipe.process(img, figure_id=r.sample_id)
            st = [p["status"] for p in res["panels"]]
            rows.append(dict(sid=r.sample_id, group=grp, pair_ok=r.B_pairing_ok, scale_ok=r.B_scale_ok, n_before=int(r.n_panels_ok),
                             n_ok=st.count("ok"), n_held=st.count("no_exclusive_bar"), n_own_unread=st.count("own_bar_unreadable"),
                             n_bar_unread=st.count("bar_unreadable"), n_no_bar=st.count("no_bar_detected") + st.count("no_compatible_bar"),
                             bars_ok=sum(1 for b in res["bars"] if b.get("ok")), bars=len(res["bars"]),
                             units="|".join(str(p["qc"].get("unit")) for p in res["panels"] if p["status"] == "ok")))
    t = pd.DataFrame(rows)
    if a.out:
        t.to_csv(a.out, index=False, encoding="utf-8-sig")
    q = t[t.group == "quantified"]
    ok, bad = q[q.pair_ok == "y"], q[q.pair_ok == "n"]
    both = q[(q.pair_ok == "y") & (q.scale_ok == "y")]
    print(f"[{a.reader}/{a.pairing}/{a.unit}] correct-pairing figures {len(ok)}: still quantified {int((ok.n_ok > 0).sum())}, panels {int(ok.n_before.sum())} -> {int(ok.n_ok.sum())}")
    print(f"   wrong-pairing figures {len(bad)}: still quantified {int((bad.n_ok > 0).sum())}, panels {int(bad.n_before.sum())} -> {int(bad.n_ok.sum())}")
    print(f"   fully correct figures {len(both)}: still quantified {int((both.n_ok > 0).sum())}")
    print(f"   held: no_exclusive_bar {int(q.n_held.sum())}, own_bar_unreadable {int(q.n_own_unread.sum())} | bars readable {int(q.bars_ok.sum())}/{int(q.bars.sum())}")
    u = t[t.group == "unquantified_afm"]
    print(f"   30 unquantified AFM figures: now quantified {int((u.n_ok > 0).sum())} (panels {int(u.n_ok.sum())}); bars readable {int(u.bars_ok.sum())}/{int(u.bars.sum())}; "
          f"held own_unread {int(u.n_own_unread.sum())}, bar_unread {int(u.n_bar_unread.sum())}, no_bar {int(u.n_no_bar.sum())}")


if __name__ == "__main__":
    main()
