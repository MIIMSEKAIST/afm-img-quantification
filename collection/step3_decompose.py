"""
Step 3: Composite figure → sub-panel 분할
Usage: python -m src.step3_decompose
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_split_positions(profile: np.ndarray, min_gap: int = 20, threshold: float = 0.95) -> list:
    """
    1D intensity profile에서 분할 위치 탐지.
    흰색(밝은) 영역이 연속되는 곳 = 패널 사이 gap.
    """
    max_val = profile.max()
    is_gap = profile > max_val * threshold

    # 연속 gap 구간 찾기
    gaps = []
    in_gap = False
    start = 0
    for i in range(len(is_gap)):
        if is_gap[i] and not in_gap:
            start = i
            in_gap = True
        elif not is_gap[i] and in_gap:
            if i - start >= min_gap:
                gaps.append((start, i))
            in_gap = False

    # gap 중앙을 분할 위치로
    splits = [(g[0] + g[1]) // 2 for g in gaps]
    return splits


def decompose_figure(image_path: str, output_dir: Path, figure_id: str) -> list:
    """
    단일 figure를 sub-panel로 분할.
    가로/세로 방향으로 whitespace gap을 탐지하여 분할.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가로 방향 분할 (세로 gap 탐지)
    col_profile = np.mean(gray, axis=0)
    v_splits = find_split_positions(col_profile, min_gap=max(10, w // 50))

    # 세로 방향 분할 (가로 gap 탐지)
    row_profile = np.mean(gray, axis=1)
    h_splits = find_split_positions(row_profile, min_gap=max(10, h // 50))

    # 분할 경계 생성
    x_bounds = [0] + v_splits + [w]
    y_bounds = [0] + h_splits + [h]

    panels = []
    panel_idx = 0

    for yi in range(len(y_bounds) - 1):
        for xi in range(len(x_bounds) - 1):
            x1, x2 = x_bounds[xi], x_bounds[xi + 1]
            y1, y2 = y_bounds[yi], y_bounds[yi + 1]

            # 너무 작은 panel 제외
            pw, ph = x2 - x1, y2 - y1
            if pw < 50 or ph < 50:
                continue
            if pw * ph < 0.01 * w * h:
                continue

            panel_img = img[y1:y2, x1:x2]
            panel_name = f"{figure_id}_panel{panel_idx:02d}.png"
            panel_path = output_dir / panel_name
            cv2.imwrite(str(panel_path), panel_img)

            panels.append({
                "figure_id": figure_id,
                "panel_idx": panel_idx,
                "image_path": str(panel_path),
                "bbox": f"[{x1},{y1},{x2},{y2}]",
                "split_method": "auto",
            })
            panel_idx += 1

    # 분할이 안 된 경우 원본을 그대로 1개 panel로
    if not panels:
        panel_path = output_dir / f"{figure_id}_panel00.png"
        cv2.imwrite(str(panel_path), img)
        panels.append({
            "figure_id": figure_id,
            "panel_idx": 0,
            "image_path": str(panel_path),
            "bbox": f"[0,0,{w},{h}]",
            "split_method": "none",
        })

    return panels


def main():
    cfg = load_config()
    root = Path(cfg["project"]["root"])

    fig_df = pd.read_parquet(root / "data" / "02_figures" / "figures.parquet")
    panel_dir = root / "data" / "03_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    all_panels = []

    for _, row in tqdm(fig_df.iterrows(), total=len(fig_df), desc="Decomposing panels"):
        figure_id = row["figure_id"]
        image_path = row["image_path"]

        if not image_path or not Path(image_path).exists():
            continue

        out_dir = panel_dir / figure_id
        out_dir.mkdir(parents=True, exist_ok=True)

        panels = decompose_figure(image_path, out_dir, figure_id)
        for p in panels:
            p["panel_id"] = f"PN{len(all_panels):07d}"
            all_panels.append(p)

    panel_df = pd.DataFrame(all_panels)
    panel_df.to_parquet(root / "data" / "03_panels" / "panels.parquet", index=False)

    print(f"\n--- Summary ---")
    print(f"  Total panels: {len(panel_df)}")
    print(f"  From {fig_df.figure_id.nunique()} figures")
    print(f"  Split method distribution:")
    print(panel_df["split_method"].value_counts().to_string())


if __name__ == "__main__":
    main()