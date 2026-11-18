"""Aggregate the first-pass ratings of the second corpus sample (verification_v2.xlsx; verdicts produced with an AI
coding assistant, see README.md): before/after comparison of the
submitted (baseline) and revised (evidence) pipelines on the same figures.

Reports, per configuration: figures quantified, panels quantified, figures with all pairings correct, figures with
all read scales correct, figures fully correct (both), each with a figure-level bootstrap 95% CI; for the revised
pipeline also the withheld panels (in the whole sample and in the rated figures, which are the only ones the rater
saw) and how many of the latter the rater judged wrongly withheld. Writes notes/검증_v2_결과.md and
results/verification_v2_summary.json.

A later run of the revised pipeline whose changed figures were re-rated (make_rerating_sheet.py) is aggregated with
--sheet <re-rating workbook> --new final5; the earlier rating, which guided that final round of revisions, is kept
in the summary as "released_heldout" with --before-sheet <earlier workbook> --before final4.

Usage: python aggregate_verification_v2.py [--sheet verification_v2.xlsx] [--new final4]
                                           [--before-sheet <xlsx> --before final4]"""
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


def read_sheet(path):
    return pd.read_excel(path, sheet_name="verify", dtype=str).fillna("").set_index("sample_id")


def block(v, r, pre, held=False):
    """Figure-level summary of one configuration (results r) against the ratings of sheet v (columns pre + ...)."""
    q = r[r.n_panels_ok > 0]
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
    if held:
        held_per_fig = r.n_held + (r.n_own_unread if "n_own_unread" in r else 0)
        in_sheet = [s for s in v.index if s in r.index]
        wrong_held = pd.to_numeric(v.get("evid_held_wrongly", pd.Series(dtype=str)).replace("", np.nan), errors="coerce").fillna(0)
        d.update({"panels_held": int(held_per_fig.sum()), "panels_held_in_rated": int(held_per_fig.loc[in_sheet].sum()),
                  "panels_held_wrongly": int(wrong_held.sum()), "figures_with_held": int((r.n_held > 0).sum())})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=S / "verification_v2.xlsx")
    ap.add_argument("--new", default="evidence", help="configuration compared with the baseline (results CSV prefix)")
    ap.add_argument("--before-sheet", type=Path, default=None, help="earlier rating workbook, kept as the held-out rating of --before")
    ap.add_argument("--before", default=None, help="configuration of the earlier rating (results CSV prefix)")
    a = ap.parse_args()
    sheet = a.sheet if a.sheet.is_absolute() else S / a.sheet
    v = read_sheet(sheet)
    base = pd.read_csv(S / "baseline_results.csv", keep_default_na=False).set_index("sample_id")
    evid = pd.read_csv(S / f"{a.new}_results.csv", keep_default_na=False).set_index("sample_id")
    out = {"figures": int(len(base)), "unreadable": int(base.get("unreadable", pd.Series(dtype=bool)).fillna(False).sum()),
           "sheet": sheet.name, "baseline": block(v, base, "base_"), "evidence": dict(block(v, evid, "evid_", held=True), config=a.new)}
    h = None
    if a.before_sheet or a.before:
        assert a.before_sheet and a.before, "--before-sheet and --before go together"
        bsheet = a.before_sheet if a.before_sheet.is_absolute() else S / a.before_sheet
        rb = pd.read_csv(S / f"{a.before}_results.csv", keep_default_na=False).set_index("sample_id")
        h = out["released_heldout"] = dict(block(read_sheet(bsheet), rb, "evid_", held=True), config=a.before, sheet=bsheet.name)
    if "rerate" in v.columns:
        rr = v.rerate.str.strip().str.lower()
        extra = v.gallery_page.astype(str).eq("page_extra") if "gallery_page" in v.columns else pd.Series(False, index=v.index)
        out["rerating"] = {"figures_kept": int((rr == "n").sum()), "figures_rerated": int(((rr == "y") & ~extra).sum()),
                           "figures_added": int(((rr == "y") & extra).sum())}
    # figures quantified by the baseline but not by the revised pipeline: were they correct?
    dropped = [s for s in base.index if base.loc[s, "n_panels_ok"] > 0 and evid.loc[s, "n_panels_ok"] == 0 and s in v.index]
    if dropped:
        bb = v.loc[dropped]
        out["dropped_by_evidence"] = {"figures": len(dropped),
                                     "were_fully_correct": int(((bb.base_pairing_ok.str.lower() == "y") & (bb.base_scale_ok.str.lower() == "y")).sum())}
    RES.mkdir(exist_ok=True)
    (RES / "verification_v2_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    b, e = out["baseline"], out["evidence"]
    cols = [("기준(제출본)", b)] + ([(f"개선 ({h['config']}, held-out)", h)] if h else []) + [(f"개선 ({a.new})", e)]
    row = lambda name, f: f"| {name} | " + " | ".join(f(d) for _, d in cols) + " |"
    md = ["# 새 표본(1,000장) 사람 검증 결과 — 개선 전/후", "",
          f"그림 {out['figures']}장 (읽을 수 없는 이미지 {out['unreadable']}장). 시트 {sheet.name}. 채점 누락: "
          + ", ".join(f"{n} {d['ratings_missing']}" for n, d in cols), "",
          "| | " + " | ".join(n for n, _ in cols) + " |", "|---|" + "---|" * len(cols),
          row("정량화된 그림", lambda d: str(d["figures_quantified"])),
          row("정량화된 패널", lambda d: f"{d['panels_quantified']}" + (f" (보류 {d['panels_held']}, 그중 평가 그림 안 {d['panels_held_in_rated']})" if "panels_held" in d else "")),
          row("짝 모두 정확", lambda d: f"{d['pairing_correct']}/{d['figures_rated']} ({100 * d['pairing_correct_frac']:.0f}%, CI {d['pairing_ci']})"),
          row("범위 모두 정확", lambda d: f"{d['scale_correct']}/{d['figures_rated']} ({100 * d['scale_correct_frac']:.0f}%)"),
          row("**둘 다 정확**", lambda d: f"**{d['fully_correct']}/{d['figures_rated']} ({100 * d['fully_correct_frac']:.0f}%, CI {d['fully_ci']})**"),
          row("잘못 보류된 패널 (사람 판정)", lambda d: f"{d['panels_held_wrongly']} / {d['panels_held_in_rated']}" if "panels_held" in d else "—")]
    if "dropped_by_evidence" in out:
        dd = out["dropped_by_evidence"]
        md.append(f"| 기준만 정량화한 그림 | {dd['figures']} (그중 완전 정확 {dd['were_fully_correct']}) |" + " — |" * (len(cols) - 1))
    if "rerating" in out:
        rr = out["rerating"]
        md += ["", f"재평가: 유지 {rr['figures_kept']}장, 다시 평가 {rr['figures_rerated']}장, 새로 추가 {rr['figures_added']}장"]
    (NOTES / "검증_v2_결과.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
