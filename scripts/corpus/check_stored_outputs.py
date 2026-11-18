"""How closely a configuration of the released code reproduces the stored outputs of a pipeline state.

The outputs of the earlier pipeline states (``out_<config>/`` of a sample; ``pipeline_output_*`` in the Zenodo deposit)
were produced with the code of their time. This script gives the stored detector boxes of each figure to the released
code, run with the options of a configuration of ``run_sample.py``, and compares the colour-bar readings, the pairing
and the status of every panel with the stored ones. The detector is not run, so that the comparison isolates the
stages after detection.

    python check_stored_outputs.py --config baseline [--sample-dir <sample>] [--n 0]

Result for the initial pipeline (``baseline``) on the second sample, 18 November 2026, on the 109 figures that have
panels and colour-bar candidates in the stored output (230 colour-bar candidates, 378 panels):
    released code with the options of ``baseline``: 95 figures identical; in 14 figures at least one colour bar is
    read differently (other end values, or read where it had not been read, and the reverse), because the strip
    localisation and the number parsing that both readers share were revised after the initial runs (commits of 27
    and 28 October 2026);
    code of the initial runs (commit 823566c, run with its defaults): 109 figures identical.
The options of the released code therefore reproduce the initial pipeline approximately. Its exact code is in the
history of the repository, and its outputs on every sample are in the Zenodo deposit.
"""
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image

from _paths import ROOT  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
from afmquant.pipeline import FigurePipeline, CLASS_MAP, CLASS_BAR  # noqa: E402
from run_sample import CONFIGS  # noqa: E402


class FromStoredBoxes(FigurePipeline):
    """The pipeline with the detections of a stored output in place of the detector."""
    current = None

    def detect(self, image_rgb):
        j = self.current
        maps = [{"xyxy": [float(v) for v in m["xyxy"]], "conf": float(m["conf"]), "cls": CLASS_MAP} for m in j["maps"]]
        bars = [{"xyxy": [float(v) for v in b["box"]], "conf": 1.0, "cls": CLASS_BAR} for b in j["bars"]]
        return maps, bars


def close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def differences(old, new):
    d = []
    for i, (b0, b1) in enumerate(zip(old["bars"], new["bars"])):
        if bool(b0.get("ok")) != bool(b1.get("ok")):
            d.append(f"bar {i} read {b0.get('ok')} -> {b1.get('ok')}")
        elif b0.get("ok") and not (close(b0["vmin"], b1["vmin"]) and close(b0["vmax"], b1["vmax"])
                                   and (b0.get("unit") or None) == (b1.get("unit") or None)):
            d.append(f"bar {i} scale {b0['vmin']:.4g}..{b0['vmax']:.4g} {b0.get('unit')!r} -> {b1['vmin']:.4g}..{b1['vmax']:.4g} {b1.get('unit')!r}")
    for i, (p0, p1) in enumerate(zip(old["panels"], new["panels"])):
        if p0["status"] != p1["status"]:
            d.append(f"panel {i} status {p0['status']} -> {p1['status']}")
        if p0["bar_idx"] != p1["bar_idx"]:
            d.append(f"panel {i} bar {p0['bar_idx']} -> {p1['bar_idx']}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--sample-dir", type=Path, default=ROOT / "revision" / "detection" / "yield_sample_v2")
    ap.add_argument("--n", type=int, default=0, help="number of figures (spread over the sample); 0 = all")
    a = ap.parse_args()
    stored = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((a.sample_dir / f"out_{a.config}").glob("*.json"))}
    cand = [k for k, v in stored.items() if v["panels"] and v["bars"]]          # figures with panels and colour-bar candidates
    sel = cand[::max(1, len(cand) // a.n)][:a.n] if a.n else cand
    cfg = dict(CONFIGS[a.config]); weights = cfg.pop("weights")
    pipe = FromStoredBoxes(weights, **cfg)
    same, diffs = 0, []
    tot = {"figures": 0, "bars": 0, "panels": 0}
    for sid in sel:
        old = stored[sid]
        pipe.current = old
        new = pipe.process(np.array(Image.open(a.sample_dir / "imgs" / f"{sid}.png").convert("RGB")), figure_id=sid)
        assert len(new["bars"]) == len(old["bars"]) and len(new["panels"]) == len(old["panels"])
        d = differences(old, new)
        tot["figures"] += 1; tot["bars"] += len(old["bars"]); tot["panels"] += len(old["panels"])
        if d:
            diffs.append((sid, d))
        else:
            same += 1
    print(f"{a.config}: compared {tot}")
    print(f"figures with identical readings, pairing and statuses: {same} of {tot['figures']}")
    for sid, d in diffs:
        print("  ", sid, "; ".join(d[:6]))


if __name__ == "__main__":
    main()
