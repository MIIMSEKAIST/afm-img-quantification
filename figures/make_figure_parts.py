"""Generate individual figure "parts" (one PNG per panel) for assembly.

Renders the per-panel images and graphs that make up Figures 3, 4, 5 and S4 from
the ground-truth IBW (via the validated `afmquant` engine), saving each panel
separately under figures/parts/{fig3,fig4,fig5,figS4}/. Panel labels (a/b/c) and
final layout are added in a vector editor.

Conventions:
  - AFM image panels include the scale bar + color bar; no titles/labels.
  - Graph panels include axes/legends; no panel titles.
  - All panels are saved with minimal margins (tight, pad~0) for easy alignment.

Usage:  python figures/make_figure_parts.py
"""

import re
from pathlib import Path
import numpy as np, pandas as pd
import igor2.binarywave as bw
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as ssim_f

# ===================== CONFIG =====================
import os, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
IBW_DIR   = Path(os.environ.get("IBW_DIR", "data/esm_ibw"))  # ground-truth IBW dir
META_CSV  = "results/meta_scanlevel.csv"   # for Fig5 (auto-generated from IBW if absent)
OUT       = Path("figures/parts")          # output root for figure parts
COLORMAPS = ["copper", "hot", "jet", "viridis"]
CMAP_HEX  = {"copper":"#b87333","hot":"#d62728","jet":"#1f77b4","viridis":"#2ca02c"}
JPEG_Q    = 85
SCALEBAR_UM = 1.0
DPI       = 300
# Fig4 대표 스캔 (stem, 재료, 렌더 colormap) — 자유 교체
FIG4_REPS = [
    ("LICGC_01_4um",           "LICGC",   "copper"),
    ("LMNO_pristine_01_4um",   "LMNO",    "jet"),
    ("NCM811_pristine_03_4um", "NCM811",  "viridis"),
]
QUALITIES = [50, 65, 75, 85, 95]
SIZES     = [256, 192, 128, 96, 64]
MATS      = ["LMNO", "LICGC", "NCM811"]
plt.rcParams.update({"font.size":9})
# =================================================

# ---- engine (thin wrappers over the validated afmquant package) ----
from afmquant import quantify_map as _qm, render_to_published_style as _render
from benchmark.reproduce_benchmark import list_benchmark_ibw
def quantify_map(map_rgb, strip_rgb, vmin, vmax, distance_threshold=20.0):
    vm, valid, _ = _qm(map_rgb, strip_rgb, vmin, vmax, distance_threshold)
    return vm, valid
def render_published(gt, cmap_name, q=85):
    return _render(gt, cmap_name, q)

# ---- 공용 ----
def material_of(stem):
    s = stem.upper()
    return "LICGC" if s.startswith("LICGC") else "LMNO" if s.startswith("LMNO") else "NCM811" if s.startswith("NCM") else stem.split("_")[0]
def scan_size_um(stem):
    m = re.search(r"_(\d+)um", stem); return int(m.group(1)) if m else 4
def load_scans():
    out=[]
    for p in list_benchmark_ibw(IBW_DIR):
        gt = bw.load(str(p))["wave"]["wData"][:,:,2].astype(float)
        out.append((p.stem, material_of(p.stem), np.nan_to_num(gt, nan=np.nanmedian(gt))))
    if not out: raise FileNotFoundError(f"No benchmark .ibw under {IBW_DIR.resolve()}")
    return out
def metrics(gt, rec, valid):
    rng = gt.max()-gt.min()+1e-12
    nmae = float(np.abs(gt[valid]-rec[valid]).mean()/rng*100)
    gn, rn = (gt-gt.min())/rng, (rec-gt.min())/rng
    return nmae, float(ssim_f(gn, rn, data_range=1.0))
def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("  +", path)

# ---- AFM 이미지 부품 (scale bar + colorbar 포함, 제목 없음) ----
def save_afm_image(matrix_pm, size_um, cmap, out_path, vlim=None, is_rgb=False, rgb=None):
    """단일 AFM 이미지 부품. is_rgb=True면 rgb(렌더 결과)를 그대로 표시."""
    lo, hi = (vlim if vlim else (matrix_pm.min(), matrix_pm.max()))
    npx = (rgb.shape[0] if is_rgb else matrix_pm.shape[0])
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    if is_rgb:
        ax.imshow(rgb)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    else:
        im = ax.imshow(matrix_pm, cmap=cmap, vmin=lo, vmax=hi)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Amplitude (pm)", fontsize=8); cb.ax.tick_params(labelsize=7)
    # scale bar (좌하단, 흰색)
    bar_px = npx*SCALEBAR_UM/size_um; x0, y0 = npx*0.05, npx*0.93
    ax.add_patch(Rectangle((x0,y0), bar_px, npx*0.018, facecolor="white", edgecolor="none", zorder=10))
    ax.text(x0+bar_px/2, y0-npx*0.025, f"{SCALEBAR_UM:g} \u00b5m", color="white",
            ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=11)
    ax.set_xticks([]); ax.set_yticks([])
    savefig(fig, out_path)

# ================= Fig 4 부품 =================
def parts_fig4(scans):
    d = OUT/"fig4"; lut = {s:gt for s,_,gt in scans}
    for stem, mat, cm in FIG4_REPS:
        gt = lut[stem]; size_um = scan_size_um(stem); gt_pm = gt*1e12
        m, s, vmin, vmax = render_published(gt, cm, q=JPEG_Q)
        vm, valid = quantify_map(m, s, vmin, vmax); vm_pm = vm*1e12
        lo, hi = gt_pm.min(), gt_pm.max()
        # (a) 원본 — gray
        save_afm_image(gt_pm, size_um, "gray", d/f"{mat}_a_original.png", vlim=(lo,hi))
        # (b) published-style 렌더 — 해당 colormap + colorbar
        save_afm_image(None, size_um, cm, d/f"{mat}_b_rendered_{cm}.png", vlim=(lo,hi), is_rgb=True, rgb=m)
        # (c) 복원 — gray
        save_afm_image(vm_pm, size_um, "gray", d/f"{mat}_c_reconstructed.png", vlim=(lo,hi))
        # (d) 분포 그래프 — 부품 (축 포함, 제목 없음)
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        bins = np.linspace(lo, hi, 60)
        ax.hist(gt_pm.ravel(), bins=bins, color="black", histtype="step", lw=1.4, label="Original", density=True)
        ax.hist(vm_pm[valid].ravel(), bins=bins, color=mpl.colormaps[cm](0.6), alpha=0.55, label="Reconstructed", density=True)
        ax.set_xlabel("Amplitude (pm)"); ax.set_ylabel("Density"); ax.set_yticks([]); ax.legend(frameon=False, fontsize=8)
        savefig(fig, d/f"{mat}_d_distribution.png")

# ================= Fig 3 부품 =================
def parts_fig3(scans):
    d = OUT/"fig3"
    rows=[]
    for stem, mat, gt in scans:
        for cm in COLORMAPS:
            m,s,vmin,vmax = render_published(gt, cm, q=JPEG_Q)
            vm,valid = quantify_map(m,s,vmin,vmax); nmae,ss = metrics(gt,vm,valid)
            rows.append(dict(colormap=cm, nMAE=nmae, SSIM=ss))
    df = pd.DataFrame(rows)
    # (a) colormap별 nMAE 막대 — 부품
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    g = df.groupby("colormap").nMAE; means, stds = g.mean().reindex(COLORMAPS), g.std().reindex(COLORMAPS)
    ax.bar(COLORMAPS, means.values, yerr=stds.values, capsize=4,
           color=[CMAP_HEX[c] for c in COLORMAPS], alpha=0.85, edgecolor="k", lw=0.5)
    ax.set_ylabel("nMAE (%)"); savefig(fig, d/"a_nMAE_by_colormap.png")
    # (b) SSIM 히스토 — 부품
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.hist(df.SSIM, bins=30, color="#4472C4", edgecolor="k", lw=0.4)
    ax.axvline(df.SSIM.mean(), color="red", ls="--", label=f"Mean: {df.SSIM.mean():.3f}")
    ax.set_xlabel("SSIM"); ax.set_ylabel("Count"); ax.legend(frameon=False); savefig(fig, d/"b_SSIM_hist.png")
    # (c) nMAE 히스토 — 부품
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.hist(df.nMAE, bins=30, color="#70AD47", edgecolor="k", lw=0.4)
    ax.axvline(df.nMAE.mean(), color="red", ls="--", label=f"Mean: {df.nMAE.mean():.2f}%")
    ax.set_xlabel("nMAE (%)"); ax.set_ylabel("Count"); ax.legend(frameon=False); savefig(fig, d/"c_nMAE_hist.png")

# ================= Fig S4 부품 =================
def parts_figS4(scans):
    d = OUT/"figS4"
    rows=[]
    for stem, mat, gt in scans:
        for cm in COLORMAPS:
            for q in QUALITIES:
                m,s,vmin,vmax = render_published(gt,cm,q=q); vm,valid = quantify_map(m,s,vmin,vmax)
                nmae,ss = metrics(gt,vm,valid); rows.append(dict(sweep="quality",colormap=cm,param=q,nMAE=nmae,SSIM=ss))
            m_full,s,vmin,vmax = render_published(gt,cm,q=JPEG_Q)
            for size in SIZES:
                if size==256: m,gtr = m_full,gt
                else:
                    m=np.array(Image.fromarray(m_full).resize((size,size),Image.BILINEAR))
                    gtr=np.array(Image.fromarray(gt.astype(np.float32),mode="F").resize((size,size),Image.BILINEAR)).astype(float)
                vm,valid=quantify_map(m,s,vmin,vmax); nmae,ss=metrics(gtr,vm,valid)
                rows.append(dict(sweep="resolution",colormap=cm,param=size,nMAE=nmae,SSIM=ss))
    df = pd.DataFrame(rows)
    panels = [("quality","nMAE","a_nMAE_vs_quality","JPEG quality","nMAE (%)",True),
              ("quality","SSIM","b_SSIM_vs_quality","JPEG quality","SSIM",True),
              ("resolution","nMAE","c_nMAE_vs_resolution","Figure resolution (pixels)","nMAE (%)",False),
              ("resolution","SSIM","d_SSIM_vs_resolution","Figure resolution (pixels)","SSIM",False)]
    for sweep, metric, fname, xl, yl, invert in panels:
        fig, ax = plt.subplots(figsize=(4.0, 3.2)); sub = df[df.sweep==sweep]
        for cm in COLORMAPS:
            g = sub[sub.colormap==cm].groupby("param")[metric]
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, marker="o", ms=4,
                        capsize=3, color=CMAP_HEX[cm], label=cm, lw=1.3)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=8)
        if invert: ax.invert_xaxis()
        savefig(fig, d/f"{fname}.png")

# ================= Fig 5 부품 =================
def parts_fig5(scans):
    d = OUT/"fig5"
    if Path(META_CSV).exists():
        meta = pd.read_csv(META_CSV)
    else:
        rows=[]
        for stem, mat, gt in scans:
            for cm in COLORMAPS:
                m,s,vmin,vmax=render_published(gt,cm,q=JPEG_Q); vm,valid=quantify_map(m,s,vmin,vmax)
                rows.append(dict(scan=stem,material=mat,colormap=cm,gt_med=np.median(gt),rec_med=np.median(vm[valid])))
        meta=pd.DataFrame(rows); meta.to_csv(META_CSV,index=False)
    meta["material"]=meta.material.replace({"NCM":"NCM811"})
    meta["gt_pm"]=meta.gt_med*1e12; meta["rec_pm"]=meta.rec_med*1e12
    gt26 = meta[meta.colormap==COLORMAPS[0]][["material","gt_pm"]]
    xpos = {m:i for i,m in enumerate(MATS)}; rng=np.random.default_rng(1)
    # (a) 원본 결론 부품
    fig, ax = plt.subplots(figsize=(4.0, 3.6)); med={}
    for m in MATS:
        v=gt26[gt26.material==m].gt_pm.values
        ax.scatter(np.full(len(v),xpos[m])+rng.uniform(-0.13,0.13,len(v)),v,s=26,color="black",alpha=0.75,zorder=3); med[m]=np.median(v)
    for m in MATS: ax.hlines(med[m],xpos[m]-0.32,xpos[m]+0.32,color="red",lw=2,zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels(MATS); ax.set_ylabel("Scan-median ESM amplitude (pm)"); ax.set_ylim(0,950)
    savefig(fig, d/"a_conclusions_original.png")

    # (b) 복원 결론 부품 (4 colormap)
    fig, ax = plt.subplots(figsize=(4.0, 3.6)); offs={"copper":-0.20,"hot":-0.07,"jet":0.07,"viridis":0.20}; medr={}
    for cm in COLORMAPS:
        sub=meta[meta.colormap==cm]
        for m in MATS:
            v=sub[sub.material==m].rec_pm.values
            ax.scatter(np.full(len(v),xpos[m])+offs[cm],v,s=15,color=CMAP_HEX[cm],alpha=0.85,zorder=3,label=cm if m==MATS[0] else None)
    for m in MATS: medr[m]=np.median(meta[meta.material==m].rec_pm.values); ax.hlines(medr[m],xpos[m]-0.32,xpos[m]+0.32,color="red",lw=2,zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels(MATS); ax.set_ylabel("Scan-median ESM amplitude (pm)"); ax.set_ylim(0,950)
    ax.legend(frameon=False, fontsize=7, loc="upper right"); savefig(fig, d/"b_conclusions_reconstructed.png")

    # (c) effect size 부품
    def cliffs(a,b):
        a,b=np.asarray(a),np.asarray(b); return (sum(x>y for x in a for y in b)-sum(x<y for x in a for y in b))/(len(a)*len(b))
    DP=[("LMNO","LICGC"),("LMNO","NCM811"),("LICGC","NCM811")]; width=0.15
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    entries=[("Ground truth", lambda m: gt26[gt26.material==m].gt_pm.values, "black")]
    for cm in COLORMAPS: entries.append((cm,(lambda m,cm=cm: meta[(meta.colormap==cm)&(meta.material==m)].rec_pm.values),CMAP_HEX[cm]))
    for j,(lab,getv,col) in enumerate(entries):
        ds=[cliffs(getv(a),getv(b)) for a,b in DP]
        ax.bar(np.arange(3)+(j-2)*width,ds,width=width*0.92,color=col,label=lab,edgecolor="k",lw=0.4)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{a} vs\n{b}" for a,b in DP],fontsize=8)
    ax.set_ylabel("Cliff's delta"); ax.set_ylim(0,1.18); ax.legend(frameon=False,fontsize=7)
    savefig(fig, d/"c_effect_sizes.png")

if __name__ == "__main__":
    scans = load_scans()
    print(f"Loaded {len(scans)} scans from {IBW_DIR.resolve()}\n")
    print("Fig 3 parts:"); parts_fig3(scans)
    print("Fig 4 parts:"); parts_fig4(scans)
    print("Fig 5 parts:"); parts_fig5(scans)
    print("Fig S4 parts:"); parts_figS4(scans)
    print(f"\nDONE. All parts under ./{OUT}/  (fig3, fig4, fig5, figS4)")