"""
Step 1: PDF 다운로드 + 즉시 검증
Usage: python -m src.step1_download
"""
import sys
import requests
import pandas as pd
import time
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.pdf_validator import is_valid_pdf


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_pdf_url_unpaywall(doi: str, email: str = "victoryhwan@kaist.ac.kr"):
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        best = data.get("best_oa_location")
        if best and best.get("url_for_pdf"):
            return best["url_for_pdf"]
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
    except Exception:
        pass
    return None


def download_pdf(url: str, save_path: Path, timeout: int = 30) -> bool:
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code == 200 and len(resp.content) > 1000:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


def main():
    cfg = load_config()
    root = Path(cfg["project"]["root"])
    timeout = cfg.get("step1", {}).get("download_timeout", 30)

    papers_path = root / "data" / "00_papers" / "papers.parquet"
    df = pd.read_parquet(papers_path)
    pdf_dir = root / "data" / "01_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pending = df[df["pdf_status"] == "pending"]
    print(f"Papers to process: {len(pending)}")

    success = 0
    invalid = 0
    no_url = 0

    for idx, row in tqdm(pending.iterrows(), total=len(pending), desc="Downloading"):
        doi = row["doi"]
        save_path = pdf_dir / f"{row['paper_id']}.pdf"

        # 이미 존재하면 스킵
        if save_path.exists():
            valid, reason = is_valid_pdf(save_path)
            df.at[idx, "pdf_status"] = "downloaded" if valid else "invalid"
            continue

        pdf_url = get_pdf_url_unpaywall(doi)
        if not pdf_url and row.get("pdf_url"):
            pdf_url = row["pdf_url"]

        if not pdf_url:
            df.at[idx, "pdf_status"] = "not_available"
            no_url += 1
            continue

        if download_pdf(pdf_url, save_path, timeout=timeout):
            # 즉시 검증
            valid, reason = is_valid_pdf(save_path)
            if valid:
                df.at[idx, "pdf_status"] = "downloaded"
                success += 1
            else:
                df.at[idx, "pdf_status"] = "invalid"
                save_path.unlink(missing_ok=True)  # 깨진 파일 삭제
                invalid += 1
        else:
            df.at[idx, "pdf_status"] = "failed"

        time.sleep(0.3)

    df.to_parquet(papers_path, index=False)

    total_valid = (df["pdf_status"] == "downloaded").sum()
    print(f"\n--- Summary ---")
    print(f"  New downloads: {success}")
    print(f"  Invalid (deleted): {invalid}")
    print(f"  No URL: {no_url}")
    print(f"  Total valid PDFs: {total_valid}")


if __name__ == "__main__":
    main()