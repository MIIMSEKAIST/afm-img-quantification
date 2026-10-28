"""Re-rating sheet for a single-pipeline sample (yield_sample_v3 or v4) after a newer run of the released pipeline:
keeps every rating of the rater's workbook, marks the figures whose result changed in a rating-relevant way (status,
paired bar, unit, or an end value by more than 2% of the span) with `rerate = y` (their pairing_ok / scale_ok /
held_wrongly cells cleared and highlighted; cells that no longer apply set to na / 0 in grey), appends figures the new
run quantifies that were absent from the sheet, and writes a Korean guide. Build the gallery of the new run first:
    python make_verification_v3.py final7 --gallery-only            (-> gallery_v3_final7/)

Usage: python make_rerating_sheet_v3.py --old verification_v3.xlsx --before final6 --new final7 [--sample-dir ...]
Writes verification_<tag>_<new>_rerate_<date>.xlsx and 재평가_안내_<tag>_<date>.md; prints RERATE: <ids>."""
import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.comments import Comment
from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)

TOL = 0.02
MARK = PatternFill("solid", fgColor="FFF2CC")
GREY = PatternFill("solid", fgColor="EDEDED")
NONE = PatternFill(fill_type=None)
PER_PAGE = 40


def panels(S, cfg, sid):
    f = S / f"out_{cfg}" / f"{sid}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    out = []
    for p in d.get("panels", []):
        b = d["bars"][p["bar_idx"]] if p.get("bar_idx") is not None and d.get("bars") else None
        rng = (float(b["vmin"]), float(b["vmax"]), b.get("unit")) if b and b.get("ok") else None
        out.append((p["status"], p.get("bar_idx"), rng))
    return out


def same(pa, pb):
    if pa is None or pb is None or len(pa) != len(pb):
        return pa == pb
    for (sa, ba, ra), (sb, bb, rb) in zip(pa, pb):
        if sa != sb or ba != bb or (ra is None) != (rb is None):
            return False
        if ra is not None:
            span = max(abs(ra[1] - ra[0]), abs(rb[1] - rb[0]), 1e-12)
            if ra[2] != rb[2] or abs(ra[0] - rb[0]) > TOL * span or abs(ra[1] - rb[1]) > TOL * span:
                return False
    return True


def counts(res, sid):
    if sid not in res.index:
        return 0, 0
    r = res.loc[sid]
    return int(r["n_panels_ok"]), int(r["n_held"]) + int(r.get("n_own_unread", 0))


def mark(ws, r, col, n_new, n_held):
    for name in ("pairing_ok", "scale_ok"):
        c = ws.cell(row=r, column=col[name])
        if n_new == 0:
            c.value, c.fill = "na", GREY
        else:
            c.value, c.fill = None, MARK
    c = ws.cell(row=r, column=col["held_wrongly"])
    if n_held == 0:
        c.value, c.fill = 0, GREY
    else:
        c.value, c.fill = None, MARK
    c = ws.cell(row=r, column=col["rerate"], value="y"); c.fill = MARK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--before", default="final6")
    ap.add_argument("--new", default="final7")
    ap.add_argument("--sample-dir", type=Path, default=ROOT / "revision" / "detection" / "yield_sample_v3")
    a = ap.parse_args()
    S = a.sample_dir
    tag = S.name.rsplit("_", 1)[-1]
    new = pd.read_csv(S / f"{a.new}_results.csv", keep_default_na=False).set_index("sample_id")
    man = pd.read_csv(S / "sample_manifest.csv").set_index("sample_id")
    wb = load_workbook(S / a.old)
    ws = wb["verify"]
    head = [c.value for c in ws[1]]
    col = {name: i + 1 for i, name in enumerate(head)}
    if "rerate" not in col:
        c = ws.cell(row=1, column=len(head) + 1, value="rerate"); c.font = Font(bold=True); c.fill = MARK
        col["rerate"] = len(head) + 1
    sids = [ws.cell(row=r, column=col["sample_id"]).value for r in range(2, ws.max_row + 1)]
    changed, kept, pages = [], [], defaultdict(list)
    for r, sid in zip(range(2, ws.max_row + 1), sids):
        if not sid:
            continue
        pa, pb = panels(S, a.before, sid), panels(S, a.new, sid)
        n_before = int(ws.cell(row=r, column=col["n_ok"]).value or 0)
        n_new, n_held = counts(new, sid)
        ws.cell(row=r, column=col["n_ok"], value=n_new)
        ws.cell(row=r, column=col["n_held"], value=n_held)
        if same(pa, pb):
            c = ws.cell(row=r, column=col["rerate"], value="n"); c.fill = NONE
            kept.append(sid)
            continue
        mark(ws, r, col, n_new, n_held)
        changed.append(sid)
        pages[str(ws.cell(row=r, column=col["gallery_page"]).value)].append(f"{sid} (초록 {n_before}→{n_new}, 보류 {n_held})")
    added = []
    r = ws.max_row + 1
    known = set(s for s in sids if s)
    extra = sorted(set(new[new.n_panels_ok > 0].index) - known)
    for k, sid in enumerate(extra):
        n_new, n_held = counts(new, sid)
        vals = {"sample_id": sid, "gallery_page": f"page_extra", "n_ok": n_new, "n_held": n_held,
                "caption": str(man.loc[sid, "caption"])[:400] if sid in man.index else ""}
        for name, v in vals.items():
            if name in col:
                ws.cell(row=r, column=col[name], value=v)
        mark(ws, r, col, n_new, n_held)
        added.append(sid)
        pages["page_extra"].append(f"{sid} (새로 정량화, 초록 {n_new}, 보류 {n_held})")
        r += 1
    ws.cell(row=1, column=col["rerate"]).comment = Comment("y = 결과가 바뀐 그림: 노란 칸을 다시 평가. n = 바뀌지 않음(기존 평가 유지).", "Claude")
    today = f"{dt.date.today():%Y%m%d}"
    out = S / f"verification_{tag}_{a.new}_rerate_{today}.xlsx"
    wb.save(out)
    # gallery page for the appended figures (the new run's overlays)
    if extra:
        gal = S / f"gallery_{tag}_{a.new}"; gal.mkdir(exist_ok=True)
        STYLE = ("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;font-size:12px} .c{border-bottom:2px solid #999;padding:8px} "
                 "img{max-width:1000px;max-height:620px;border:1px solid #ccc} .cap{max-width:1000px;color:#444}</style></head><body>")
        html = [STYLE, f"<h3>sample {tag} — figures newly quantified by {a.new}</h3>"]
        for sid in extra:
            n_new, n_held = counts(new, sid)
            html.append(f"<div class='c'><b>{sid}</b> &nbsp; ok={n_new} 보류={n_held}<div><img src='../out_{a.new}/{sid}_overlay.png'></div>"
                        f"<div class='cap'><i>{str(man.loc[sid, 'caption'])[:400] if sid in man.index else ''}</i></div></div>")
        html.append("</body></html>")
        (gal / "page_extra.html").write_text("\n".join(html), encoding="utf-8")
    n_re = len(changed) + len(added)
    guide = [f"# 재평가 안내 — 표본 {tag}, 마지막 수정({a.new}) 후, {dt.date.today():%Y-%m-%d}", "",
             f"- 파일: `{out.name}` (기존 평가는 모두 그대로 있습니다.)",
             f"- 대상: `rerate` 열이 **y**인 {n_re}행만 (바뀐 그림 {len(changed)}장 + 새로 정량화된 그림 {len(added)}장). `n`인 {len(kept)}행은 결과가 바뀌지 않아 기존 평가가 그대로 유효합니다.",
             f"- 그림: `gallery_{tag}_{a.new}\\page_XX.html` (마지막 수정 결과). 새로 정량화된 그림은 같은 폴더의 `page_extra.html`.",
             "- 채울 칸(노란색): `pairing_ok`, `scale_ok`, `held_wrongly` — 기준은 1차 평가와 같습니다. 회색 na/0은 해당 없음.",
             "- `note`: n이면 이유를 짧게 (단위 / 부호 / 범위 / 짝).",
             f"- 예상 시간: 그림당 1분 안팎 ({n_re}장 ≈ {max(5, n_re)}분).",
             f"- 끝나면 같은 파일에 저장하고 Claude에게 \"재평가 {tag} 끝\"이라고 알려 주세요.", "",
             "## 다시 볼 그림 (갤러리 페이지별)", ""]
    for pg in sorted(pages):
        guide.append(f"- `{pg}`: " + ", ".join(pages[pg]))
    (S / f"재평가_안내_{tag}_{today}.md").write_text("\n".join(guide) + "\n", encoding="utf-8")
    print(f"written {out.name}: kept {len(kept)}, to re-rate {len(changed)}, appended {len(added)}; guide 재평가_안내_{tag}_{today}.md")
    print("RERATE:", ",".join(changed + added) if (changed or added) else "none")


if __name__ == "__main__":
    main()
