"""
Step 2: PDFigCapX(pdffigures2)로 figure-caption 추출
Usage: python -m src.step2_extract
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def extract_figure_list(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "figures" in data and isinstance(data["figures"], list):
            return data["figures"]
        if "pdffigures" in data and isinstance(data["pdffigures"], dict):
            figs = data["pdffigures"].get("figures", [])
            if isinstance(figs, list):
                return figs
    return []


def run_pdffigures2_single(pdf_path: Path, raw_dir: Path, jar_path: Path, timeout: int = 300, heap: str = "4g"):
    raw_img_dir = raw_dir / "images"
    raw_json_dir = raw_dir / "json"
    stat_path = raw_dir / "stat_file.json"

    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    raw_img_dir.mkdir(parents=True, exist_ok=True)
    raw_json_dir.mkdir(parents=True, exist_ok=True)

    # 한 파일만 들어있는 임시 폴더 생성
    temp_pdf_dir = raw_dir / "pdf_input"
    temp_pdf_dir.mkdir(parents=True, exist_ok=True)
    temp_pdf_path = temp_pdf_dir / pdf_path.name

    try:
        with open(pdf_path, "rb") as fsrc:
            data = fsrc.read()
        with open(temp_pdf_path, "wb") as fdst:
            fdst.write(data)
    except Exception as e:
        return -1, "", f"copy_failed: {pdf_path} | {e}", raw_img_dir, raw_json_dir

    
    cmd = [
        "java",
        f"-Xmx{heap}",
        "-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider",
        "-cp",
        str(jar_path),
        "org.allenai.pdffigures2.FigureExtractorBatchCli",
        str(temp_pdf_dir),
        "-s", str(stat_path),
        "-m", str(raw_img_dir) + "/",
        "-d", str(raw_json_dir) + "/",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return result.returncode, result.stdout, result.stderr, raw_img_dir, raw_json_dir


def main():
    cfg = load_config()
    root = Path(cfg["project"]["root"])
    jar_path = Path(cfg["step2"]["jar_path"])
    heap = cfg.get("step2", {}).get("java_heap", "4g")
    timeout = int(cfg.get("step2", {}).get("timeout", 300))

    pdf_dir = root / "data" / "01_pdfs"
    fig_root = root / "data" / "02_figures"
    papers_path = root / "data" / "00_papers" / "papers.parquet"

    if not jar_path.exists():
        raise FileNotFoundError(f"pdffigures2 jar not found: {jar_path}")

    papers_df = pd.read_parquet(papers_path)
    downloaded = papers_df[papers_df["pdf_status"] == "downloaded"].copy()

    print(f"Processing {len(downloaded)} downloaded PDFs")
    print(f"Using jar: {jar_path}")

    all_figures = []
    fig_counter = 0
    failed_pdfs = []

    for _, row in tqdm(downloaded.iterrows(), total=len(downloaded), desc="Extracting figures"):
        paper_id = row["paper_id"]
        pdf_path = pdf_dir / f"{paper_id}.pdf"

        if not pdf_path.exists():
            continue

        raw_dir = fig_root / "_raw" / paper_id
        returncode, stdout, stderr, raw_img_dir, raw_json_dir = run_pdffigures2_single(
            pdf_path=pdf_path,
            raw_dir=raw_dir,
            jar_path=jar_path,
            timeout=timeout,
            heap=heap,
        )

        if returncode != 0:
            failed_pdfs.append({"paper_id": paper_id, "pdf_path": str(pdf_path), "stderr": stderr[:1000]})
            continue

        image_files = sorted(
            [p for p in raw_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
            key=lambda p: natural_key(p.name),
        )
        json_files = sorted(raw_json_dir.rglob("*.json"), key=lambda p: natural_key(p.name))

        if not json_files:
            continue

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            figures = extract_figure_list(data)
            if not figures:
                continue

            paper_out_dir = fig_root / paper_id
            paper_out_dir.mkdir(parents=True, exist_ok=True)

            for i, fig in enumerate(figures):
                if i >= len(image_files):
                    continue

                src_img = image_files[i]
                fig_id = f"F{fig_counter:06d}"
                fig_counter += 1

                dst_img = paper_out_dir / f"{fig_id}{src_img.suffix.lower()}"
                shutil.copy2(src_img, dst_img)

                caption = fig.get("caption", "")
                if isinstance(caption, dict):
                    caption = json.dumps(caption, ensure_ascii=False)
                elif caption is None:
                    caption = ""

                all_figures.append({
                    "figure_id": fig_id,
                    "paper_id": paper_id,
                    "figure_num": i,
                    "image_path": str(dst_img),
                    "caption": str(caption),
                    "fig_type": str(fig.get("figType", "unknown")),
                })

    fig_df = pd.DataFrame(
        all_figures,
        columns=["figure_id", "paper_id", "figure_num", "image_path", "caption", "fig_type"],
    )
    fig_df.to_parquet(fig_root / "figures.parquet", index=False)

    failed_df = pd.DataFrame(failed_pdfs)
    failed_df.to_csv(fig_root / "failed_pdfs.csv", index=False, encoding="utf-8-sig")

    print("\n--- Summary ---")
    print(f"  Total figures extracted: {len(fig_df)}")
    print(f"  Papers with figures: {fig_df['paper_id'].nunique() if len(fig_df) else 0}")
    print(f"  Failed PDFs: {len(failed_df)}")
    print(f"  Saved: {fig_root / 'figures.parquet'}")
    print(f"  Failed log: {fig_root / 'failed_pdfs.csv'}")


if __name__ == "__main__":
    main()