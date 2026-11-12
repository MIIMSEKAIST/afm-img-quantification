"""Run the pipeline on the fresh held-out corpus sample (yield_sample_v2) in a given configuration, resumably.

    python run_sample_v2.py --config baseline      # submitted pipeline (nearest pairing, raw units)
    python run_sample_v2.py --config evidence      # revised pipeline (evidence pairing, lexicon units)

Outputs under yield_sample_v2/out_<config>/ (<sid>.json, <sid>_overlay.png) and <config>_results.csv.
Figures whose JSON already exists are skipped, so an interrupted run can be resumed."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
from afmquant.pipeline import FigurePipeline  # noqa: E402
from run_figure import draw_overlay  # noqa: E402

RUNS = ROOT / "runs" / "detect" / "runs" / "afm_detect"
CONFIGS = {"baseline": dict(pairing_mode="nearest", unit_mode="raw", reader="v1", end_labels="fit", crop="box", weights=RUNS / "v13" / "weights" / "best.pt"),
           "evidence": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v1", end_labels="fit", crop="box", weights=RUNS / "v13" / "weights" / "best.pt"),
           "final": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", conf=0.25, end_labels="fit", crop="box", weights=RUNS / "v15" / "weights" / "best.pt"),
           "final2": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", conf=0.25, end_labels="fit", crop="box", weights=RUNS / "v15" / "weights" / "best.pt"),
           "final3": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", conf=0.25, end_labels="fit", crop="box", weights=RUNS / "v15" / "weights" / "best.pt"),
           "final4": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", conf=0.25, end_labels="fit", crop="box", weights=RUNS / "v15" / "weights" / "best.pt"),
           "final5": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", conf=0.25, end_labels="fit", crop="box", weights=RUNS / "v15" / "weights" / "best.pt"),
           "final6": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", conf=0.25, end_labels="fit", crop="box", weights=RUNS / "v15" / "weights" / "best.pt"),
           "final7": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", conf=0.25, end_labels="fit", crop="box", weights=RUNS / "v15" / "weights" / "best.pt"),
           "final8": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", conf=0.25, end_labels="verbatim", crop="panel",
                          weights=RUNS / "v15" / "weights" / "best.pt"),   # released: end labels verbatim, crop trimmed to the panel
           "final_v13": dict(pairing_mode="evidence", unit_mode="lexicon", reader="v2", end_labels="fit", crop="box", weights=RUNS / "v13" / "weights" / "best.pt")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--sample-dir", type=Path, default=ROOT / "revision" / "detection" / "yield_sample_v2")
    a = ap.parse_args()
    out = a.sample_dir / f"out_{a.config}"; out.mkdir(exist_ok=True)
    man = pd.read_csv(a.sample_dir / "sample_manifest.csv")
    cfg = dict(CONFIGS[a.config]); weights = cfg.pop("weights")
    pipe = FigurePipeline(weights, **cfg)
    for k, r in man.iterrows():
        jf = out / f"{r.sample_id}.json"
        if jf.exists():
            continue
        try:
            image = np.array(Image.open(a.sample_dir / "imgs" / f"{r.sample_id}.png").convert("RGB"))
        except Exception as e:                       # a corpus render that is not a readable image (e.g. T0592)
            jf.write_text(json.dumps({"maps": [], "bars": [], "panels": [], "error": f"unreadable_image: {e}"}), encoding="utf-8")
            print(f"  {r.sample_id}: unreadable image, skipped", flush=True)
            continue
        res = pipe.process(image, figure_id=r.sample_id)
        Image.fromarray(draw_overlay(image, res)).save(out / f"{r.sample_id}_overlay.png")
        jf.write_text(json.dumps({
            "maps": res["maps"],
            "bars": [{k2: v for k2, v in b.items() if k2 in ("box", "ok", "vmin", "vmax", "unit", "unit_raw", "reason", "orientation")} for b in res["bars"]],
            "panels": [{"map_box": p["map_box"], "map_conf": p["map_conf"], "bar_idx": p["bar_idx"], "compat": p["compat"],
                        "status": p["status"], "evidence": p.get("pairing_evidence"),
                        "unit": (p.get("qc") or {}).get("unit")} for p in res["panels"]]}, indent=1, default=float), encoding="utf-8")
        if k % 100 == 0:
            print(f"  {k}/{len(man)}", flush=True)
    rows = []
    for r in man.itertuples():
        j = json.loads((out / f"{r.sample_id}.json").read_text(encoding="utf-8"))
        st = [p["status"] for p in j["panels"]]
        rows.append(dict(sample_id=r.sample_id, paper_id=r.paper_id, caption_afm_kw=bool(r.caption_afm_kw), n_maps=len(j["maps"]),
                         unreadable=bool(j.get("error")),
                         n_bars=len(j["bars"]), n_bars_readable=sum(1 for b in j["bars"] if b.get("ok")), n_panels_ok=st.count("ok"),
                         n_held=st.count("no_exclusive_bar"), n_own_unread=st.count("own_bar_unreadable"),
                         units="|".join(str(p.get("unit", (j["bars"][p["bar_idx"]].get("unit") if p.get("bar_idx") is not None else None)))
                                        for p in j["panels"] if p["status"] == "ok")))
    df = pd.DataFrame(rows); df.to_csv(a.sample_dir / f"{a.config}_results.csv", index=False)
    print(f"{a.config}: {len(df)} figures | detected {(df.n_maps > 0).sum()} | quantified figures {(df.n_panels_ok > 0).sum()} | "
          f"panels ok {df.n_panels_ok.sum()} | held {df.n_held.sum()} + own-bar-unreadable {df.n_own_unread.sum()}")


if __name__ == "__main__":
    main()
