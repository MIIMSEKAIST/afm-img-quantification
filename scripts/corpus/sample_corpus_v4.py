"""Fourth corpus sample: the held-out test of the released pipeline after its last round of reader revisions.

Same paper-wise draw as sample_corpus.py / sample_corpus_v3.py, excluding every paper of the three earlier samples
(yield_sample, yield_sample_v2, yield_sample_v3), every paper with a figure in the detector training sets
(datasets/afm_detection, afm_detection_v2) and every image identical to a training image. The remaining pool is
small (436 papers), so up to --per-paper figures are taken from each paper (default 2; the earlier samples took one
per paper and a second only when the pool was exhausted). Figure ids V####.

Usage: python sample_corpus_v4.py --n 1000 --per-paper 2 --seed 3 --out D:/AFM_Database_SH/revision/detection/yield_sample_v4
"""
import argparse, hashlib, json, random, re, shutil
from pathlib import Path
import pandas as pd

from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
RAW = ROOT / "data" / "02_figures" / "_raw"
DATASETS = [ROOT / "datasets" / "afm_detection", ROOT / "datasets" / "afm_detection_v2"]
EARLIER = [ROOT / "revision" / "detection" / d / "sample_manifest.csv" for d in ("yield_sample", "yield_sample_v2", "yield_sample_v3")]
KW = re.compile(r"\b(AFM|atomic force|topograph|PFM|piezoresponse|ESM|electrochemical strain|KPFM|Kelvin|C-AFM|conductive AFM|MFM|"
                r"magnetic force|phase image|height image|adhesion|friction|force map|nanoindent|scanning probe|SPM)\b", re.I)


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per-paper", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "revision" / "detection" / "yield_sample_v4")
    a = ap.parse_args()
    (a.out / "imgs").mkdir(parents=True, exist_ok=True)

    train_hashes, train_fids = set(), set()
    for d in DATASETS:
        for p in d.glob("*/images/*"):
            train_hashes.add(md5(p))
            if p.stem.startswith("F"):
                train_fids.add(p.stem)
    figs = pd.read_parquet(ROOT / "data" / "02_figures" / "figures.parquet")
    excl = set(figs[figs.figure_id.isin(train_fids)].paper_id)
    n_train = len(excl)
    for m in EARLIER:
        excl |= set(pd.read_csv(m).paper_id)
    papers = pd.read_parquet(ROOT / "data" / "00_papers" / "papers.parquet").set_index("paper_id")

    records = []
    for pdir in sorted(RAW.iterdir()):
        pid = pdir.name
        if pid in excl:
            continue
        jf = pdir / "json" / f"{pid}.json"
        if not jf.exists():
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        entries = data if isinstance(data, list) else data.get("figures", [])
        for e in entries:
            if not isinstance(e, dict) or e.get("figType") != "Figure":
                continue
            img = pdir / "images" / Path(str(e.get("renderURL", ""))).name
            if not img.exists():
                continue
            records.append(dict(paper_id=pid, fig_name=str(e.get("name")), page=e.get("page"), image=str(img),
                                caption=str(e.get("caption", ""))[:2000]))
    df = pd.DataFrame(records)
    print(f"corpus figures with captions: {len(df)} from {df.paper_id.nunique()} papers "
          f"(excluded {len(excl)} papers: {n_train} training + the three earlier samples)")

    rng = random.Random(a.seed)
    by_paper = {pid: g.sample(frac=1, random_state=a.seed) for pid, g in df.groupby("paper_id")}
    pids = list(by_paper); rng.shuffle(pids)
    chosen = []
    for rnd in range(a.per_paper):
        for pid in pids:
            g = by_paper[pid]
            if len(g) > rnd:
                chosen.append(g.iloc[rnd])
            if len(chosen) >= a.n:
                break
        if len(chosen) >= a.n:
            break
    sample = pd.DataFrame(chosen).reset_index(drop=True)

    rows = []
    for i, r in sample.iterrows():
        src = Path(r.image)
        if md5(src) in train_hashes:
            continue
        sid = f"V{i:04d}"
        shutil.copy2(src, a.out / "imgs" / f"{sid}.png")
        meta = papers.loc[r.paper_id] if r.paper_id in papers.index else None
        rows.append(dict(sample_id=sid, paper_id=r.paper_id, fig_name=r.fig_name, page=r.page, src=r.image, caption=r.caption,
                         doi=None if meta is None else meta.doi, journal=None if meta is None else meta.journal,
                         year=None if meta is None else meta.year))
    out = pd.DataFrame(rows)
    out["caption_afm_kw"] = out.caption.fillna("").str.contains(KW)
    out.to_csv(a.out / "sample_manifest.csv", index=False)
    print(f"sampled {len(out)} figures from {out.paper_id.nunique()} papers (up to {a.per_paper} per paper) -> {a.out}; "
          f"captions with AFM keywords: {out.caption_afm_kw.mean():.1%}")


if __name__ == "__main__":
    main()
