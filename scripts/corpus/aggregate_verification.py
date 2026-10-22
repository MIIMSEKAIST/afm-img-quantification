"""Aggregate the human ratings of the fresh held-out sample (verification_v2.xlsx): before/after comparison of the
submitted (baseline) and revised (evidence) pipelines on the same figures.

Reports, per configuration: figures quantified, panels quantified, figures with all pairings correct, figures with
all read scales correct, figures fully correct (both), each with a figure-level bootstrap 95% CI; for the revised
pipeline also the held panels and how many a rater judged wrongly held. Writes notes/검증_v2_결과.md and
results/verification_v2_summary.json.

Usage: python aggregate_verification_v2.py [--sheet verification_v2.xlsx]"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)

S = ROOT / "revision" / "detection" / "yield_sample_v2"
NOTES = ROOT / "revision" / "notes"
RES = ROOT / "revision" / "results"


def ci(vals, B=4000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(vals, len(vals)).mean() for _ in range(B)]
    return tuple(np.round(np.percentile(bs, [2.5, 97.5]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=S / "verification_v2.xlsx")
    ap.add_argument("--new", default="evidence", help="configuration compared with the baseline (results CSV prefix)")
    a = ap.parse_args()
    v = pd.read_excel(a.sheet, sheet_name="verify", dtype=str).fillna("").set_index("sample_id")
    base = pd.read_csv(S / "baseline_results.csv", keep_default_na=False).set_index("sample_id")
    evid = pd.read_csv(S / f"{a.new}_results.csv", keep_default_na=False).set_index("sample_id")
    n_fig = len(base)
    out = {"figures": int(n_fig), "unreadable": int(base.get("unreadable", pd.Series(dtype=bool)).fillna(False).sum())}
    for name, r in (("baseline", base), ("evidence", evid)):
        q = r[r.n_panels_ok > 0]
        pre = "base_" if name == "baseline" else "evid_"
        rated = v.loc[[s for s in q.index if s in v.index]]
        pair = rated[pre + "pairing_ok"].str.strip().str.lower(); scale = rated[pre + "scale_ok"].str.strip().str.lower()
        missing = int(((pair == "") | (scale == "")).sum())
        both = (pair == "y") & (scale == "y")
        d = {"figures_quantified": int(len(q)), "panels_quantified": int(q.n_panels_ok.sum()),
             "figures_rated": int(len(rated)), "ratings_missing": missing,
             "pairing_correct": int((pair == "y").sum()), "scale_correct": int((scale == "y").sum()), "fully_correct": int(both.sum()),
             "pairing_correct_frac": float((pair == "y").mean()) if len(rated) else np.nan, "pairing_ci": ci(pair == "y"),
             "scale_correct_frac": float((scale == "y").mean()) if len(rated) else np.nan, "scale_ci": ci(scale == "y"),
             "fully_correct_frac": float(both.mean()) if len(rated) else np.nan, "fully_ci": ci(both)}
        if name == "evidence":
            held = r.n_held.sum() + (r.n_own_unread.sum() if "n_own_unread" in r else 0)
            wrong_held = pd.to_numeric(v.get("evid_held_wrongly", pd.Series(dtype=str)).replace("", np.nan), errors="coerce").fillna(0)
            d.update({"panels_held": int(held), "panels_held_wrongly": int(wrong_held.sum()),
                      "figures_with_held": int((r.n_held > 0).sum())})
        out[name] = d
    # figures quantified by the baseline but not by the revised pipeline: were they correct?
    both_idx = [s for s in base.index if base.loc[s, "n_panels_ok"] > 0 and evid.loc[s, "n_panels_ok"] == 0 and s in v.index]
    if both_idx:
        bb = v.loc[both_idx]
        out["dropped_by_evidence"] = {"figures": len(both_idx),
                                     "were_fully_correct": int(((bb.base_pairing_ok.str.lower() == "y") & (bb.base_scale_ok.str.lower() == "y")).sum())}
    RES.mkdir(exist_ok=True)
    (RES / "verification_v2_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    b, e = out["baseline"], out["evidence"]
    md = ["# 새 표본(1,000장) 사람 검증 결과 — 개선 전/후", "",
          f"그림 {out['figures']}장 (읽을 수 없는 이미지 {out['unreadable']}장). 채점 누락: 기준 {b['ratings_missing']}, 개선 {e['ratings_missing']}", "",
          "| | 기준(제출본) | 개선 |", "|---|---|---|",
          f"| 정량화된 그림 | {b['figures_quantified']} | {e['figures_quantified']} |",
          f"| 정량화된 패널 | {b['panels_quantified']} | {e['panels_quantified']} (보류 {e['panels_held']}) |",
          f"| 짝 모두 정확 | {b['pairing_correct']}/{b['figures_rated']} ({100*b['pairing_correct_frac']:.0f}%, CI {b['pairing_ci']}) | {e['pairing_correct']}/{e['figures_rated']} ({100*e['pairing_correct_frac']:.0f}%, CI {e['pairing_ci']}) |",
          f"| 범위 모두 정확 | {b['scale_correct']}/{b['figures_rated']} ({100*b['scale_correct_frac']:.0f}%) | {e['scale_correct']}/{e['figures_rated']} ({100*e['scale_correct_frac']:.0f}%) |",
          f"| **둘 다 정확** | **{b['fully_correct']}/{b['figures_rated']} ({100*b['fully_correct_frac']:.0f}%, CI {b['fully_ci']})** | **{e['fully_correct']}/{e['figures_rated']} ({100*e['fully_correct_frac']:.0f}%, CI {e['fully_ci']})** |",
          f"| 잘못 보류된 패널 (사람 판정) | — | {e['panels_held_wrongly']} / {e['panels_held']} |"]
    if "dropped_by_evidence" in out:
        dd = out["dropped_by_evidence"]
        md.append(f"| 기준만 정량화한 그림 | {dd['figures']} (그중 완전 정확 {dd['were_fully_correct']}) | — |")
    (NOTES / "검증_v2_결과.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
