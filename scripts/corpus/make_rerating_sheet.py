"""Re-rating sheet for the held-out sample after a newer run of the revised pipeline (cycle 2 = final5).

Starts from the rater's completed workbook (final4 ratings), keeps every rating, and asks for a new rating only
where the revised pipeline's result changed in a rating-relevant way (status, paired bar, unit, or an end value by
more than 2% of the span): those rows get their evid_* cells cleared and highlighted and `rerate = y` (cells that
do not apply, e.g. pairing/scale of a figure the new run no longer quantifies, are set to na / 0 in grey). Figures
that the new run quantifies but that were absent from the sheet (neither pipeline quantified them before) are
appended with `rerate = y` (base_* = na, as the submitted pipeline did not quantify them). The galleries must be
refreshed first (`refresh_gallery_v2.py final5`): page numbers are taken from the sheet (unchanged) and page_extra
for the appended figures. Also writes the Korean guide 재평가_안내_<date>.md next to the workbook.

Usage: python make_rerating_sheet.py --old verification_v2_final4_reviewed_20261026.xlsx --before final4 --new final5
Writes yield_sample_v2/verification_v2_<new>_rerate_<date>.xlsx and prints the figures to re-rate."""
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

S = ROOT / "revision" / "detection" / "yield_sample_v2"
TOL = 0.02
MARK = PatternFill("solid", fgColor="FFF2CC")
GREY = PatternFill("solid", fgColor="EDEDED")
NONE = PatternFill(fill_type=None)


def panels(cfg, sid):
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


def counts(new, sid):
    if sid not in new.index:
        return 0, 0
    r = new.loc[sid]
    return int(r["n_panels_ok"]), int(r["n_held"]) + int(r.get("n_own_unread", 0))


def mark_rating_cells(ws, r, col, n_new, n_held):
    """Clear and highlight the cells to re-rate; grey out the ones that do not apply."""
    for name in ("evid_pairing_ok", "evid_scale_ok"):
        c = ws.cell(row=r, column=col[name])
        if n_new == 0:
            c.value, c.fill = "na", GREY
        else:
            c.value, c.fill = None, MARK
    c = ws.cell(row=r, column=col["evid_held_wrongly"])
    if n_held == 0:
        c.value, c.fill = 0, GREY
    else:
        c.value, c.fill = None, MARK
    c = ws.cell(row=r, column=col["rerate"], value="y")
    c.fill = MARK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="the rater's completed workbook (file name in yield_sample_v2)")
    ap.add_argument("--before", default="final4")
    ap.add_argument("--new", default="final5")
    a = ap.parse_args()
    old = S / a.old
    new = pd.read_csv(S / f"{a.new}_results.csv", keep_default_na=False).set_index("sample_id")
    man = pd.read_csv(S / "sample_manifest.csv").set_index("sample_id")
    wb = load_workbook(old)
    ws = wb["verify"]
    head = [c.value for c in ws[1]]
    col = {name: i + 1 for i, name in enumerate(head)}
    if "rerate" not in col:
        c = ws.cell(row=1, column=len(head) + 1, value="rerate")
        c.font = Font(bold=True); c.fill = MARK
        col["rerate"] = len(head) + 1
    sids = [ws.cell(row=r, column=col["sample_id"]).value for r in range(2, ws.max_row + 1)]
    changed, kept, pages = [], [], defaultdict(list)
    for r, sid in zip(range(2, ws.max_row + 1), sids):
        if not sid:
            continue
        pa, pb = panels(a.before, sid), panels(a.new, sid)
        n_before = int(ws.cell(row=r, column=col["evid_n_ok"]).value or 0)
        n_new, n_held = counts(new, sid)
        ws.cell(row=r, column=col["evid_n_ok"], value=n_new)
        ws.cell(row=r, column=col["evid_n_held"], value=n_held)
        if same(pa, pb):
            c = ws.cell(row=r, column=col["rerate"], value="n"); c.fill = NONE
            kept.append(sid)
            continue
        mark_rating_cells(ws, r, col, n_new, n_held)
        changed.append((sid, n_before, n_new, n_held))
        pages[str(ws.cell(row=r, column=col["gallery_page"]).value)].append(f"{sid} (초록 {n_before}→{n_new}, 보류 {n_held})")
    # figures newly quantified by the new run, absent from the sheet
    added = []
    r = ws.max_row + 1
    for sid in sorted(set(new[new.n_panels_ok > 0].index) - set(s for s in sids if s)):
        n_new, n_held = counts(new, sid)
        vals = {"sample_id": sid, "gallery_page": "page_extra", "base_pairing_ok": "na", "base_scale_ok": "na",
                "base_n_ok": 0, "evid_n_ok": n_new, "evid_n_held": n_held,
                "caption": str(man.loc[sid, "caption"])[:300] if sid in man.index and "caption" in man.columns else ""}
        for name, v in vals.items():
            if name in col:
                ws.cell(row=r, column=col[name], value=v)
        for name in ("base_pairing_ok", "base_scale_ok"):
            ws.cell(row=r, column=col[name]).fill = GREY
        mark_rating_cells(ws, r, col, n_new, n_held)
        added.append((sid, 0, n_new, n_held))
        pages["page_extra"].append(f"{sid} (새로 정량화, 초록 {n_new}, 보류 {n_held})")
        r += 1
    ws.cell(row=1, column=col["rerate"]).comment = Comment(
        "y = 오른쪽(개선) 결과가 바뀐 그림: 노란 칸(evid_*)을 다시 평가. n = 바뀌지 않음(기존 평가 유지).", "Claude")
    today = f"{dt.date.today():%Y%m%d}"
    out = S / f"verification_v2_{a.new}_rerate_{today}.xlsx"
    wb.save(out)
    n_re = len(changed) + len(added)
    guide = [f"# 재평가 안내 — 리더 사이클 2 ({a.new}), {dt.date.today():%Y-%m-%d}", "",
             f"- 파일: `{out.name}` (기존 평가는 모두 그대로 있습니다. 왼쪽 기준 파이프라인 결과는 바뀌지 않았습니다.)",
             f"- 대상: `rerate` 열이 **y**인 {n_re}행만 (바뀐 그림 {len(changed)}장 + 새로 정량화된 그림 {len(added)}장). "
             f"`n`인 {len(kept)}행은 마지막 수정으로 결과가 바뀌지 않아 기존 평가가 그대로 유효합니다.",
             f"- 그림: `gallery_v2\\page_XX.html`의 **오른쪽(개선 {a.new})** 그림만 보면 됩니다. 새로 정량화된 그림은 `gallery_v2\\page_extra.html`.",
             "- 채울 칸(노란색): `evid_pairing_ok`, `evid_scale_ok`, `evid_held_wrongly` — 기준은 1차 평가와 같습니다 "
             "(짝: 초록 상자의 선이 모두 자기 컬러바로 가면 y; 범위: 자홍 범위·단위가 눈금과 맞으면 y, 자릿수·부호·단위가 틀리면 n, "
             "눈금보다 조금 넓은 것은 y, 단위 None이면 숫자만; 보류: 주황 상자 중 바로 옆 자기 컬러바로 정량화할 수 있었던 맵의 수).",
             "- 회색 na/0: 해당 없음(오른쪽에 초록 상자가 없거나 주황 상자가 없음) — 건너뜁니다.",
             "- `note`: n이면 이유를 짧게 적어 주세요 (단위 / 부호 / 범위 / 짝). 기존 메모는 지우지 않아도 됩니다.",
             f"- 예상 시간: 그림당 1분 안팎 ({n_re}장 ≈ {max(5, round(n_re * 1.0))}분).",
             f"- 끝나면 같은 파일에 저장(또는 `verification_v2_{a.new}_reviewed_{today}.xlsx`로 저장)하고 Claude에게 \"재평가 끝\"이라고 알려 주세요.",
             "", "## 다시 볼 그림 (갤러리 페이지별)", ""]
    for pg in sorted(pages):
        guide += [f"- `{pg}`: " + ", ".join(pages[pg])]
    (S / f"재평가_안내_{today}.md").write_text("\n".join(guide) + "\n", encoding="utf-8")
    print(f"written {out.name}: rows kept {len(kept)}, to re-rate {len(changed)}, appended {len(added)}; guide 재평가_안내_{today}.md")
    print("RERATE:", ",".join(c[0] for c in changed + added) if (changed or added) else "none")


if __name__ == "__main__":
    main()
