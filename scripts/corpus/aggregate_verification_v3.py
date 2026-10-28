"""Aggregate the human rating of the third corpus sample (verification_v3.xlsx): the released pipeline only, on a
sample that guided no revision (held-out test of the released pipeline). Writes results/verification_v3_summary.json
and notes/검증_v3_결과.md.

Reports: figures quantified, panels quantified, figures rated, figures with all pairings correct, with all read
scales correct, fully correct (each with a figure-level bootstrap 95% CI), withheld panels (whole sample and in the
rated figures) and those the rater judged wrongly withheld, and the caption-keyword share.

Usage: python aggregate_verification_v3.py [--sheet verification_v3.xlsx] [--config final6]"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)

S = ROOT / "revision" / "detection" / "yield_sample_v3"
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
    ap.add_argument("--sheet", type=Path, default=S / "verification_v3.xlsx")
    ap.add_argument("--config", default="final6")
    a = ap.parse_args()
    sheet = a.sheet if a.sheet.is_absolute() else S / a.sheet
    v = pd.read_excel(sheet, sheet_name="verify", dtype=str).fillna("").set_index("sample_id")
    r = pd.read_csv(S / f"{a.config}_results.csv", keep_default_na=False).set_index("sample_id")
    man = pd.read_csv(S / "sample_manifest.csv").set_index("sample_id")
    q = r[r.n_panels_ok > 0]
    rated = v.loc[[s for s in q.index if s in v.index]]
    pair = rated.pairing_ok.str.strip().str.lower(); scale = rated.scale_ok.str.strip().str.lower()
    both = (pair == "y") & (scale == "y")
    held_per_fig = r.n_held + r.n_own_unread
    wrong_held = pd.to_numeric(v.get("held_wrongly", pd.Series(dtype=str)).replace("", np.nan), errors="coerce").fillna(0)
    out = {"figures": int(len(r)), "unreadable": int(r.get("unreadable", pd.Series(dtype=bool)).fillna(False).sum()),
           "sheet": sheet.name, "config": a.config, "caption_afm_kw_frac": float(man.caption_afm_kw.mean()),
           "detected": int((r.n_maps > 0).sum()),
           "released": {"figures_quantified": int(len(q)), "panels_quantified": int(q.n_panels_ok.sum()),
                        "figures_rated": int(len(rated)), "ratings_missing": int(((pair == "") | (scale == "")).sum()),
                        "pairing_correct": int((pair == "y").sum()), "scale_correct": int((scale == "y").sum()),
                        "fully_correct": int(both.sum()),
                        "pairing_correct_frac": float((pair == "y").mean()) if len(rated) else np.nan, "pairing_ci": ci(pair == "y"),
                        "scale_correct_frac": float((scale == "y").mean()) if len(rated) else np.nan, "scale_ci": ci(scale == "y"),
                        "fully_correct_frac": float(both.mean()) if len(rated) else np.nan, "fully_ci": ci(both),
                        "panels_held": int(held_per_fig.sum()),
                        "panels_held_in_rated": int(held_per_fig.loc[[s for s in v.index if s in r.index]].sum()),
                        "panels_held_wrongly": int(wrong_held.sum()), "figures_with_held": int((r.n_held > 0).sum())}}
    RES.mkdir(exist_ok=True)
    (RES / "verification_v3_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    e = out["released"]
    md = ["# 세 번째 표본(1,000장) 사람 검증 결과 — 최종 파이프라인(held-out)", "",
          f"그림 {out['figures']}장 (읽을 수 없는 이미지 {out['unreadable']}장), 맵 검출 {out['detected']}장, 캡션 AFM 키워드 {100 * out['caption_afm_kw_frac']:.1f}%. "
          f"시트 {sheet.name}. 채점 누락 {e['ratings_missing']}", "",
          "| | 최종 파이프라인 |", "|---|---|",
          f"| 정량화된 그림 | {e['figures_quantified']} |",
          f"| 정량화된 패널 | {e['panels_quantified']} (보류 {e['panels_held']}, 그중 평가 그림 안 {e['panels_held_in_rated']}) |",
          f"| 짝 모두 정확 | {e['pairing_correct']}/{e['figures_rated']} ({100 * e['pairing_correct_frac']:.0f}%, CI {e['pairing_ci']}) |",
          f"| 범위 모두 정확 | {e['scale_correct']}/{e['figures_rated']} ({100 * e['scale_correct_frac']:.0f}%) |",
          f"| **둘 다 정확** | **{e['fully_correct']}/{e['figures_rated']} ({100 * e['fully_correct_frac']:.0f}%, CI {e['fully_ci']})** |",
          f"| 잘못 보류된 패널 (사람 판정) | {e['panels_held_wrongly']} / {e['panels_held_in_rated']} |"]
    (NOTES / "검증_v3_결과.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
