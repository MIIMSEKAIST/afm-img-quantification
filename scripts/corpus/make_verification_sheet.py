"""Rating sheet of the first pass (see README.md) for the second corpus sample (yield_sample_v2): only figures in which at least one panel
was quantified by the baseline OR the evidence configuration. For every such figure the rater sees both overlays
side by side (baseline left, evidence right) and judges, per configuration, whether all quantified panels are
paired with their own colour bar and whether the read scales are right. One sheet, one pass, both configurations.

Usage: python make_verification_v2.py
Outputs: yield_sample_v2/verification_v2.xlsx, yield_sample_v2/gallery_v2/page_XX.html, 검증_안내_v2.md"""
import json
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter

S = ROOT / "revision" / "detection" / "yield_sample_v2"
HEAD = {"info": PatternFill("solid", fgColor="DDDDDD"), "A": PatternFill("solid", fgColor="BDD7EE"), "B": PatternFill("solid", fgColor="F8CBAD")}
GREY = PatternFill("solid", fgColor="EDEDED")

man = pd.read_csv(S / "sample_manifest.csv").set_index("sample_id")
base = pd.read_csv(S / "baseline_results.csv", keep_default_na=False).set_index("sample_id")
import sys
from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
NEW = sys.argv[1] if len(sys.argv) > 1 else "evidence"           # configuration compared with the baseline
evid = pd.read_csv(S / f"{NEW}_results.csv", keep_default_na=False).set_index("sample_id")
sel = sorted(set(base[base.n_panels_ok > 0].index) | set(evid[evid.n_panels_ok > 0].index))
print("figures to rate:", len(sel), "| baseline quantified", int((base.n_panels_ok > 0).sum()), "| evidence quantified", int((evid.n_panels_ok > 0).sum()))

cols = [("sample_id", "info", 9, None), ("gallery_page", "info", 11, None),
        ("base_pairing_ok", "B", 15, "왼쪽 그림(기준): 초록 상자의 선이 모두 자기 컬러바로 가면 y, 하나라도 틀리면 n, 초록 상자가 없으면 na"),
        ("base_scale_ok", "B", 14, "왼쪽 그림(기준): 초록 상자와 연결된 컬러바의 자홍 범위·단위가 눈금과 맞으면 y; 자릿수·부호·단위가 틀리면 n; 초록 상자가 없으면 na"),
        ("evid_pairing_ok", "B", 15, "오른쪽 그림(개선): 같은 기준"),
        ("evid_scale_ok", "B", 14, "오른쪽 그림(개선): 같은 기준"),
        ("evid_held_wrongly", "B", 17, "오른쪽 그림에서 주황(보류) 상자 중 실제로는 자기 컬러바가 바로 옆에 있어 정량화됐어야 하는 맵의 수 (없으면 0)"),
        ("note", "B", 30, "특이사항"),
        ("base_n_ok", "info", 9, None), ("evid_n_ok", "info", 9, None), ("evid_n_held", "info", 10, None), ("caption", "info", 70, None)]
wb = Workbook(); ws = wb.active; ws.title = "verify"
for j, (name, grp, w, cm) in enumerate(cols, start=1):
    c = ws.cell(row=1, column=j, value=name); c.fill = HEAD[grp]; c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.column_dimensions[get_column_letter(j)].width = w
    if cm:
        c.comment = Comment(cm, "Claude")
dv = DataValidation(type="list", formula1='"y,n,na"', allow_blank=True); ws.add_data_validation(dv)
dvi = DataValidation(type="whole", operator="between", formula1="0", formula2="99", allow_blank=True); ws.add_data_validation(dvi)
for i, sid in enumerate(sel, start=2):
    page = f"page_{(i - 2) // 40 + 1:02d}"
    vals = {"sample_id": sid, "gallery_page": page, "base_n_ok": int(base.loc[sid, "n_panels_ok"]), "evid_n_ok": int(evid.loc[sid, "n_panels_ok"]),
            "evid_n_held": int(evid.loc[sid, "n_held"]) + int(evid.loc[sid].get("n_own_unread", 0)), "caption": str(man.loc[sid, "caption"])[:400]}
    for j, (name, grp, w, cm) in enumerate(cols, start=1):
        c = ws.cell(row=i, column=j)
        if name in vals:
            c.value = vals[name]
        elif name.startswith("base_") and vals["base_n_ok"] == 0:
            c.value = "na"; c.fill = GREY
        elif name.startswith("evid_") and name != "evid_held_wrongly" and vals["evid_n_ok"] == 0:
            c.value = "na"; c.fill = GREY
        elif name == "evid_held_wrongly" and vals["evid_n_held"] == 0:
            c.value = 0; c.fill = GREY
        if name in ("base_pairing_ok", "base_scale_ok", "evid_pairing_ok", "evid_scale_ok"):
            dv.add(c)
        if name == "evid_held_wrongly":
            dvi.add(c)
ws.freeze_panes = "C2"; ws.auto_filter.ref = ws.dimensions
wb.save(S / "verification_v2.xlsx")

gal = S / "gallery_v2"; gal.mkdir(exist_ok=True)
for pg in range(0, len(sel), 40):
    chunk = sel[pg:pg + 40]
    html = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;font-size:12px} .c{border-bottom:2px solid #999;padding:8px} "
            ".row{display:flex;gap:10px} img{max-width:620px;max-height:460px;border:1px solid #ccc} .cap{max-width:1260px;color:#444}</style></head><body>",
            f"<h3>held-out sample v2 — page {pg // 40 + 1} (왼쪽 = 기준 파이프라인, 오른쪽 = 개선 파이프라인)</h3>"]
    for sid in chunk:
        html.append(f"<div class='c'><b>{sid}</b> &nbsp; 기준 ok={int(base.loc[sid,'n_panels_ok'])} &nbsp; 개선 ok={int(evid.loc[sid,'n_panels_ok'])} 보류={int(evid.loc[sid,'n_held'])}"
                    f"<div class='row'><div><div>기준</div><img src='../out_baseline/{sid}_overlay.png'></div>"
                    f"<div><div>개선</div><img src='../out_{NEW}/{sid}_overlay.png'></div></div>"
                    f"<div class='cap'><i>{str(man.loc[sid,'caption'])[:400]}</i></div></div>")
    html.append("</body></html>")
    (gal / f"page_{pg // 40 + 1:02d}.html").write_text("\n".join(html), encoding="utf-8")
print("saved verification_v2.xlsx and", (len(sel) + 39) // 40, "gallery pages")
