"""Rating sheet of the first pass (see README.md) for the third corpus sample (yield_sample_v3): the released pipeline only, rated on the
figures in which it quantified at least one panel. Per figure the rater sees the overlay and the caption and judges
whether every quantified panel is paired with its own colour bar, whether every read range and unit matches the
tick labels, and how many withheld (orange) panels an adjacent colour bar would have allowed to quantify.

Usage: python make_verification_v3.py [final6]
Outputs: yield_sample_v3/verification_v3.xlsx, yield_sample_v3/gallery_v3/page_XX.html, 검증_안내_v3.md"""
import sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter
from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)

S = ROOT / "revision" / "detection" / "yield_sample_v3"
CFG = sys.argv[1] if len(sys.argv) > 1 else "final6"
GALLERY_ONLY = "--gallery-only" in sys.argv          # a later run: rebuild gallery_v3_<cfg>/ only, keep the rated workbook
if "--sample-dir" in sys.argv:                         # another single-pipeline sample (e.g. yield_sample_v4)
    S = Path(sys.argv[sys.argv.index("--sample-dir") + 1])
TAG = S.name.rsplit("_", 1)[-1] if S.name.rsplit("_", 1)[-1].startswith("v") else "v3"
HEAD = {"info": PatternFill("solid", fgColor="DDDDDD"), "B": PatternFill("solid", fgColor="F8CBAD")}
GREY = PatternFill("solid", fgColor="EDEDED")
PER_PAGE = 40

man = pd.read_csv(S / "sample_manifest.csv").set_index("sample_id")
res = pd.read_csv(S / f"{CFG}_results.csv", keep_default_na=False).set_index("sample_id")
sel = sorted(res[res.n_panels_ok > 0].index)
print(f"figures to rate: {len(sel)} of {len(res)} (quantified panels {int(res.n_panels_ok.sum())}, "
      f"held {int(res.n_held.sum()) + int(res.n_own_unread.sum())})")

cols = [("sample_id", "info", 9, None), ("gallery_page", "info", 11, None),
        ("pairing_ok", "B", 12, "초록 상자의 선이 모두 자기 컬러바로 가면 y, 하나라도 틀리면 n"),
        ("scale_ok", "B", 11, "초록 상자와 연결된 컬러바의 자홍 범위·단위가 눈금과 맞으면 y; 자릿수·부호·단위가 틀리면 n (눈금보다 조금 넓은 것은 y; 단위 None이면 숫자만)"),
        ("held_wrongly", "B", 13, "주황(보류) 상자 중 실제로는 자기 컬러바가 바로 옆에 있어 정량화됐어야 하는 맵의 수 (없으면 0)"),
        ("note", "B", 34, "n이면 이유를 짧게 (단위 / 부호 / 범위 / 짝); 특이사항"),
        ("n_ok", "info", 7, None), ("n_held", "info", 8, None), ("caption", "info", 70, None)]
wb = Workbook(); ws = wb.active; ws.title = "verify"
for j, (name, grp, w, cm) in enumerate(cols, start=1):
    c = ws.cell(row=1, column=j, value=name); c.fill = HEAD[grp]; c.font = Font(bold=True)
    c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.column_dimensions[get_column_letter(j)].width = w
    if cm:
        c.comment = Comment(cm, "Claude")
dv = DataValidation(type="list", formula1='"y,n"', allow_blank=True); ws.add_data_validation(dv)
dvi = DataValidation(type="whole", operator="between", formula1="0", formula2="99", allow_blank=True); ws.add_data_validation(dvi)
for i, sid in enumerate(sel, start=2):
    n_held = int(res.loc[sid, "n_held"]) + int(res.loc[sid, "n_own_unread"])
    vals = {"sample_id": sid, "gallery_page": f"page_{(i - 2) // PER_PAGE + 1:02d}", "n_ok": int(res.loc[sid, "n_panels_ok"]),
            "n_held": n_held, "caption": str(man.loc[sid, "caption"])[:400]}
    for j, (name, grp, w, cm) in enumerate(cols, start=1):
        c = ws.cell(row=i, column=j)
        if name in vals:
            c.value = vals[name]
        elif name == "held_wrongly" and n_held == 0:
            c.value = 0; c.fill = GREY
        if name in ("pairing_ok", "scale_ok"):
            dv.add(c)
        if name == "held_wrongly":
            dvi.add(c)
ws.freeze_panes = "C2"; ws.auto_filter.ref = ws.dimensions
if not GALLERY_ONLY:
    wb.save(S / f"verification_{TAG}.xlsx")

gal = S / (f"gallery_{TAG}_{CFG}" if GALLERY_ONLY else f"gallery_{TAG}"); gal.mkdir(exist_ok=True)
STYLE = ("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;font-size:12px} .c{border-bottom:2px solid #999;padding:8px} "
         "img{max-width:1000px;max-height:620px;border:1px solid #ccc} .cap{max-width:1000px;color:#444}</style></head><body>")
for pg in range(0, len(sel), PER_PAGE):
    chunk = sel[pg:pg + PER_PAGE]
    html = [STYLE, f"<h3>sample {TAG} — page {pg // PER_PAGE + 1} (공개 파이프라인 {CFG}; 초록 = 정량화, 주황 = 보류, 빨강 = 실패, 자홍 = 컬러바와 읽은 범위, 선 = 짝)</h3>"]
    for sid in chunk:
        n_held = int(res.loc[sid, "n_held"]) + int(res.loc[sid, "n_own_unread"])
        html.append(f"<div class='c'><b>{sid}</b> &nbsp; ok={int(res.loc[sid, 'n_panels_ok'])} 보류={n_held}"
                    f"<div><img src='../out_{CFG}/{sid}_overlay.png'></div>"
                    f"<div class='cap'><i>{str(man.loc[sid, 'caption'])[:400]}</i></div></div>")
    html.append("</body></html>")
    (gal / f"page_{pg // PER_PAGE + 1:02d}.html").write_text("\n".join(html), encoding="utf-8")
n_pages = (len(sel) + PER_PAGE - 1) // PER_PAGE
guide = f"""# 사람 검증 안내 — 표본 {TAG} (최종 파이프라인의 held-out 시험)

- 입력 파일: `verification_{TAG}.xlsx` (엑셀로 열어 입력하고 그대로 저장)
- 그림: `gallery_{TAG}\\page_01.html` … `page_{n_pages:02d}.html` ({PER_PAGE}장씩, 총 {len(sel)}장). 그림마다 최종 파이프라인({CFG})의 결과 하나만 있습니다.
- 대상: 최종 파이프라인이 숫자로 바꾼(초록 상자가 있는) 그림만. 짝과 범위만 채점합니다. 이 표본은 개선에 한 번도 쓰지 않았으므로, 여기서 나온 수치가 그대로 논문의 held-out 수치가 됩니다. 고치지 않고 그대로 보고합니다.
- 예상 시간: 그림당 1분 안팎 ({len(sel)}장 ≈ {max(10, round(len(sel) * 1.0))}분)

## 상자 색
| 색 | 뜻 |
|---|---|
| 초록 | 맵을 찾고 숫자로 바꾸기까지 성공 |
| **주황** | **보류**한 맵: 자기 컬러바가 확실하지 않거나(`no_exclusive_bar`), 자기 컬러바가 있는데 읽지 못함(`own_bar_unreadable`) |
| 빨강 | 맵은 찾았지만 숫자로 바꾸지 못함 |
| 자홍 | 찾은 컬러바. 위 글씨는 읽은 범위와 단위 |
| 선 | 맵과 컬러바의 짝 |

## 칸
| 칸 | 질문 | 답 |
|---|---|---|
| `pairing_ok` | 초록 상자의 선이 모두 자기 컬러바로 가나요? | y / n |
| `scale_ok` | 초록 상자와 연결된 컬러바의 자홍 범위·단위가 눈금과 맞나요? | y / n |
| `held_wrongly` | 주황 상자 중, 실제로는 자기 컬러바가 바로 옆에 있어서 숫자로 바꿀 수 있었던 맵의 수 | 숫자 (없으면 0; 회색 0은 주황 상자 없음) |
| `note` | n이면 이유를 짧게 (단위 / 부호 / 범위 / 짝), 특이사항 | 자유 |

- 자홍 범위가 눈금보다 조금 넓은 것은 정상(y)입니다. 자릿수·부호·단위가 틀리면 n입니다. 단위가 "None"이면 단위만 비교하지 않고 숫자만 봅니다.
- 끝나면 Claude에게 "검증 {TAG} 끝"이라고 알려 주세요.
"""
if not GALLERY_ONLY:
    (S / f"검증_안내_{TAG}.md").write_text(guide, encoding="utf-8")
print(f"written {'gallery only' if GALLERY_ONLY else f'verification_{TAG}.xlsx ({len(sel)} rows)'}: {gal.name}/page_01..{n_pages:02d}.html" + ("" if GALLERY_ONLY else f", 검증_안내_{TAG}.md"))
