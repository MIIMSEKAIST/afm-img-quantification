"""
Step 0: 문헌 메타데이터 수집 (Crossref API)
Usage: python -m src.step0_collect
"""
import requests
import pandas as pd
import time
import yaml
import os
from tqdm import tqdm
from pathlib import Path


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def search_crossref(query: str, max_results: int = 50000, rows_per_page: int = 200,
                    year_min: int = 2010):
    """Crossref API — cursor 기반 페이징, journal article만"""
    base_url = "https://api.crossref.org/works"
    papers = []
    cursor = "*"

    pbar = tqdm(total=max_results, desc="Collecting papers")
    while len(papers) < max_results:
        params = {
            "query": query,
            "rows": rows_per_page,
            "cursor": cursor,
            "select": "DOI,title,container-title,published-print,published-online,author,link,type",
            "sort": "relevance",
            "order": "desc",
            "mailto": "victoryhwan@kaist.ac.kr",
            "filter": f"from-pub-date:{year_min}-01-01,type:journal-article",
        }

        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"\n  API error: {e}")
            time.sleep(10)
            continue

        message = data.get("message", {})
        items = message.get("items", [])
        next_cursor = message.get("next-cursor", None)

        if not items:
            print(f"\n  No more items at {len(papers)} papers")
            break

        for item in items:
            doi = item.get("DOI", "")
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""
            journal_list = item.get("container-title", [])
            journal = journal_list[0] if journal_list else ""

            pub = item.get("published-print") or item.get("published-online") or {}
            date_parts = pub.get("date-parts", [[None]])[0]
            year = date_parts[0] if date_parts else None

            pdf_url = ""
            for link in item.get("link", []):
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL", "")
                    break

            papers.append({
                "doi": doi,
                "title": title,
                "journal": journal,
                "year": year,
                "pdf_url": pdf_url,
                "pdf_status": "pending",
            })

        pbar.update(len(items))

        if not next_cursor:
            print(f"\n  No next cursor at {len(papers)} papers")
            break

        cursor = next_cursor
        time.sleep(0.5)

    pbar.close()
    return papers[:max_results]

def main():
    cfg = load_config()
    root = Path(cfg["project"]["root"])
    query = cfg["step0"]["search_query"]
    max_papers = cfg["step0"]["max_papers"]
    year_min = cfg["step0"].get("year_min", 2010)

    print(f"Search query: {query}")
    print(f"Max papers: {max_papers}")
    print(f"Year min: {year_min}")
    print()

    papers = search_crossref(query, max_results=max_papers, year_min=year_min)
    # ... 이하 동일
    df = pd.DataFrame(papers)

    # 중복 제거
    before = len(df)
    df = df.drop_duplicates(subset="doi", keep="first")
    after = len(df)
    print(f"\nCollected: {before} → After dedup: {after}")

    # paper_id 부여
    df.insert(0, "paper_id", [f"P{i:05d}" for i in range(len(df))])

    # 저장
    out_dir = root / "data" / "00_papers"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "papers.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    # 요약
    print(f"\n--- Summary ---")
    print(f"  Total papers: {len(df)}")
    print(f"  With PDF URL: {(df['pdf_url'] != '').sum()}")
    print(f"  Year range: {df['year'].min()} – {df['year'].max()}")
    print(f"  Top journals:")
    print(df["journal"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()