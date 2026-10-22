"""Optional detection sheet for the fresh sample: 200 random figures of yield_sample_v2 with the FINAL detector's
overlays, for map counts (A columns) and box judgements (B columns) -> in-the-wild precision/recall of the final
detector on a held-out sample, with a figure-level bootstrap CI.

    python make_detection_sheet_v2.py [--n 200] [--config final]

Outputs: yield_sample_v2/detection_sheet_v2.xlsx, yield_sample_v2/gallery_det/page_XX.html"""
import argparse, random
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter
from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)

S = ROOT / "revision" / "detection" / "yield_sample_v2"
HEAD = {"info": PatternFill("solid", fgColor="DDDDDD"), "A": PatternFill("solid", fgColor="BDD7EE"), "B": PatternFill("solid", fgColor="F8CBAD")}
GREY = PatternFill("solid", fgColor="EDEDED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--config", default="final")
    a = ap.parse_args()
    man = pd.read_csv(S / "sample_manifest.csv").set_index("sample_id")
    res = pd.read_csv(S / f"{a.config}_results.csv", keep_default_na=False).set_index("sample_id")
    ids = [s for s in man.index if not bool(res.loc[s].get("unreadable", False))]
    random.Random(7).shuffle(ids)
    sel = sorted(ids[:a.n])
    cols = [("sample_id", "info", 9, None), ("gallery_page", "info", 11, None),
            ("A_has_map", "A", 10, "맵(색·밝기가 숫자인 데이터 그림)이 있으면 y, 없으면 n"),
            ("A_n_maps", "A", 9, "맵 수 (없으면 0)"),
            ("A_has_afm", "A", 10, "그중 AFM 맵이 있으면 y (캡션: AFM, topography, height, phase, PFM, ESM, KPFM, C-AFM, MFM)"),
            ("A_n_afm_maps", "A", 11, "AFM 맵 수"),
            ("A_note", "A", 22, "특이사항"),
            ("B_tp", "B", 7, "초록·빨강·주황 상자 중 맵을 둘러싼 것의 수"),
            ("B_fp", "B", 7, "초록·빨강·주황 상자 중 맵이 아닌 것을 둘러싼 것의 수"),
            ("B_note", "B", 22, "특이사항"),
            ("n_boxes", "info", 8, None), ("caption", "info", 70, None)]
    wb = Workbook(); ws = wb.active; ws.title = "verify"
    for j, (name, grp, w, cm) in enumerate(cols, start=1):
        c = ws.cell(row=1, column=j, value=name); c.fill = HEAD[grp]; c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = w
        if cm:
            c.comment = Comment(cm, "Claude")
    dv = DataValidation(type="list", formula1='"y,n"', allow_blank=True); ws.add_data_validation(dv)
    dvi = DataValidation(type="whole", operator="between", formula1="0", formula2="99", allow_blank=True); ws.add_data_validation(dvi)
    for i, sid in enumerate(sel, start=2):
        page = f"page_{(i - 2) // 40 + 1:02d}"
        nb = int(res.loc[sid, "n_maps"])
        vals = {"sample_id": sid, "gallery_page": page, "n_boxes": nb, "caption": str(man.loc[sid, "caption"])[:400]}
        for j, (name, grp, w, cm) in enumerate(cols, start=1):
            c = ws.cell(row=i, column=j)
            if name in vals:
                c.value = vals[name]
            elif name in ("B_tp", "B_fp") and nb == 0:
                c.value = 0; c.fill = GREY
            if name in ("A_has_map", "A_has_afm"):
                dv.add(c)
            if name in ("A_n_maps", "A_n_afm_maps", "B_tp", "B_fp"):
                dvi.add(c)
    ws.freeze_panes = "C2"; ws.auto_filter.ref = ws.dimensions
    wb.save(S / "detection_sheet_v2.xlsx")
    gal = S / "gallery_det"; gal.mkdir(exist_ok=True)
    for pg in range(0, len(sel), 40):
        chunk = sel[pg:pg + 40]
        html = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;font-size:12px} .c{display:flex;gap:8px;border-bottom:1px solid #ccc;padding:6px} img{max-width:640px;max-height:480px}</style></head><body>",
                f"<h3>detection sheet v2 — page {pg // 40 + 1} (최종 검출기의 상자: 초록 = 정량화, 주황 = 보류, 빨강 = 실패)</h3>"]
        for sid in chunk:
            html.append(f"<div class='c'><div><b>{sid}</b> boxes={int(res.loc[sid, 'n_maps'])}<br><img src='../out_{a.config}/{sid}_overlay.png'></div>"
                        f"<div style='max-width:420px'><i>{str(man.loc[sid, 'caption'])[:400]}</i></div></div>")
        html.append("</body></html>")
        (gal / f"page_{pg // 40 + 1:02d}.html").write_text("\n".join(html), encoding="utf-8")
    print(f"detection sheet: {len(sel)} figures, {(len(sel) + 39) // 40} gallery pages")


if __name__ == "__main__":
    main()
