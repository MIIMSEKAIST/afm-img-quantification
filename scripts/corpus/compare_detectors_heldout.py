"""Compare detectors on the human-verified corpus figures that were NOT used to train v15 (figure-level, no OCR).

    python compare_detectors_heldout.py [--weights v13 v15]

Sets (from verification_reviewed_20261021.xlsx minus datasets/afm_detection_v2/added_from_corpus_sample.csv):
  * map-free figures not used as negatives  -> false-box rate (figures with >= 1 box) and boxes per figure
  * figures with maps not used as positives -> figures with >= 1 box; boxes vs human map count (over/under)
Also the number of colour bars detected vs the human bar count on the figures with maps.
Writes revision/detection/v15/heldout_corpus_compare.json."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
from afmquant.pipeline import FigurePipeline  # noqa: E402

YS = ROOT / "revision" / "detection" / "yield_sample"
RUNS = ROOT / "runs" / "detect" / "runs" / "afm_detect"
USED = ROOT / "datasets" / "afm_detection_v2" / "added_from_corpus_sample.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="*", default=["v13", "v15"])
    ap.add_argument("--conf", nargs="*", type=float, default=[0.5])
    a = ap.parse_args()
    d = pd.read_excel(YS / "verification_reviewed_20261021.xlsx", sheet_name="verify", dtype=str).fillna("")
    for c in ["A_n_maps", "A_n_hbars", "A_n_vbars", "B_tp", "B_fp", "n_maps"]:
        d[c] = pd.to_numeric(d[c])
    used = set(pd.read_csv(USED).sample_id)
    d = d[~d.sample_id.isin(used)]
    nomap = d[d.A_has_map == "n"]; withmap = d[d.A_has_map == "y"]
    print(f"held-out corpus figures: {len(d)} (map-free {len(nomap)}, with maps {len(withmap)}, true maps {int(withmap.A_n_maps.sum())})")
    out = {}
    for name0 in a.weights:
      for conf in a.conf:
        name = f"{name0}@{conf}"
        pipe = FigurePipeline(RUNS / name0 / "weights" / "best.pt", conf=conf)
        rows = []
        for r in d.itertuples():
            img = np.array(Image.open(YS / "imgs" / f"{r.sample_id}.png").convert("RGB"))
            maps, bars = pipe.detect(img)
            rows.append(dict(sample_id=r.sample_id, has_map=r.A_has_map, n_true=int(r.A_n_maps), n_bars_true=int(r.A_n_hbars + r.A_n_vbars),
                             n_boxes=len(maps), n_bars=len(bars)))
        t = pd.DataFrame(rows)
        nm = t[t.has_map == "n"]; wm = t[t.has_map == "y"]
        res = {"map_free_figures": int(len(nm)), "map_free_with_boxes": int((nm.n_boxes > 0).sum()), "map_free_boxes_total": int(nm.n_boxes.sum()),
               "with_maps_figures": int(len(wm)), "with_maps_detected": int((wm.n_boxes > 0).sum()),
               "boxes_total": int(wm.n_boxes.sum()), "true_maps_total": int(wm.n_true.sum()),
               "boxes_le_true": int((wm.n_boxes <= wm.n_true).sum()), "boxes_eq_true": int((wm.n_boxes == wm.n_true).sum()),
               "bars_total": int(wm.n_bars.sum()), "true_bars_total": int(wm.n_bars_true.sum()), "bars_eq_true": int((wm.n_bars == wm.n_bars_true).sum())}
        out[name] = res
        print(f"[{name}] map-free: {res['map_free_with_boxes']}/{res['map_free_figures']} figures with false boxes ({100 * res['map_free_with_boxes'] / max(1, res['map_free_figures']):.1f}%), "
              f"{res['map_free_boxes_total']} boxes | with maps: detected {res['with_maps_detected']}/{res['with_maps_figures']}, boxes {res['boxes_total']} vs true maps {res['true_maps_total']}, "
              f"count equal {res['boxes_eq_true']} | bars {res['bars_total']} vs true {res['true_bars_total']}, count equal {res['bars_eq_true']}")
        t.to_csv(ROOT / "revision" / "detection" / "v15" / f"heldout_corpus_{name0}_{conf}.csv", index=False)
    (ROOT / "revision" / "detection" / "v15" / "heldout_corpus_compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
