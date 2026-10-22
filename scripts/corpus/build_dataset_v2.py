"""Detector training set v2: the original labelled set plus human-verified figures of the corpus sample.

From the human verification of the 1,000-figure corpus sample (verification_reviewed_20261021.xlsx):
  * verified positives: figures with maps in which every map was detected and no box was false (B_fp == 0 and
    B_tp == A_n_maps) AND the number of detected colour bars equals the human count -> the detector's own boxes are
    confirmed by count and class and are used as labels (human-verified pseudo-labels);
  * hard negatives: map-free figures (A_has_map == n) in which the detector produced boxes -> images with empty
    label files; plus a few clean map-free figures.
The validation and test splits are unchanged (the original hand-labelled splits), so the held-out detector metrics
stay comparable. The fresh corpus sample (yield_sample_v2) is never used, so it remains a held-out test.

Writes datasets/afm_detection_v2/{train,valid,test}/{images,labels} and data.yaml, plus a manifest CSV.
"""
import json, shutil, random
from pathlib import Path
import pandas as pd
from PIL import Image

from _paths import ROOT, REPO  # noqa: E402,F401  (workspace from AFM_DB_ROOT; see README.md)
SRC = ROOT / "datasets" / "afm_detection"
DST = ROOT / "datasets" / "afm_detection_v2"
YS = ROOT / "revision" / "detection" / "yield_sample"
N_CLEAN_NEG = 40
CLASS = {"afm_map": 0, "colorbar": 1}


def yolo_line(cls, box, W, H):
    x0, y0, x1, y1 = box
    return f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"


def main():
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            (DST / split / sub).mkdir(parents=True)
            for f in (SRC / split / sub).iterdir():
                shutil.copy2(f, DST / split / sub / f.name)
    d = pd.read_excel(YS / "verification_reviewed_20261021.xlsx", sheet_name="verify", dtype=str).fillna("")
    for c in ["A_n_maps", "A_n_hbars", "A_n_vbars", "B_tp", "B_fp", "n_maps"]:
        d[c] = pd.to_numeric(d[c])
    rows = []
    pos = d[(d.A_has_map == "y") & (d.B_fp == 0) & (d.B_tp == d.A_n_maps) & (d.A_n_maps > 0)]
    n_pos = 0
    for r in pos.itertuples():
        j = json.load(open(YS / "out" / f"{r.sample_id}.json", encoding="utf-8"))
        if len(j["bars"]) != int(r.A_n_hbars + r.A_n_vbars):
            continue                                            # bar boxes not confirmed by count -> skip the figure
        img = YS / "imgs" / f"{r.sample_id}.png"
        W, H = Image.open(img).size
        lines = [yolo_line(CLASS["afm_map"], m["xyxy"], W, H) for m in j["maps"]]
        lines += [yolo_line(CLASS["colorbar"], b["box"], W, H) for b in j["bars"]]
        shutil.copy2(img, DST / "train" / "images" / f"{r.sample_id}.png")
        (DST / "train" / "labels" / f"{r.sample_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append(dict(sample_id=r.sample_id, role="verified_positive", n_maps=len(j["maps"]), n_bars=len(j["bars"])))
        n_pos += 1
    neg = d[d.A_has_map == "n"]
    hard = list(neg[neg.B_fp > 0].sample_id)
    clean = list(neg[neg.B_fp == 0].sample_id)
    random.Random(0).shuffle(clean)
    for sid, role in [(s, "hard_negative") for s in hard] + [(s, "clean_negative") for s in clean[:N_CLEAN_NEG]]:
        shutil.copy2(YS / "imgs" / f"{sid}.png", DST / "train" / "images" / f"{sid}.png")
        (DST / "train" / "labels" / f"{sid}.txt").write_text("", encoding="utf-8")
        rows.append(dict(sample_id=sid, role=role, n_maps=0, n_bars=0))
    (DST / "data.yaml").write_text(
        f"train: {DST / 'train' / 'images'}\nval: {DST / 'valid' / 'images'}\ntest: {DST / 'test' / 'images'}\n\nnc: 2\nnames: ['afm_map', 'colorbar']\n",
        encoding="utf-8")
    man = pd.DataFrame(rows)
    man.to_csv(DST / "added_from_corpus_sample.csv", index=False)
    print(f"train images: {len(list((DST / 'train' / 'images').iterdir()))} (original 131 + verified positives {n_pos} + hard negatives {len(hard)} + clean negatives {min(N_CLEAN_NEG, len(clean))})")
    print("valid/test unchanged:", len(list((DST / 'valid' / 'images').iterdir())), len(list((DST / 'test' / 'images').iterdir())))


if __name__ == "__main__":
    main()
