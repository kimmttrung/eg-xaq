"""Độ đồng thuận giữa hai người gán nhãn (Cohen's κ).

    python scripts/label_agreement.py
    python scripts/label_agreement.py --first data/eval/episodes.yaml --second data/eval/episodes.annotator2.yaml

Chỉ tính trên episode mà CẢ HAI người đều đã `status: labeled`.

Đọc kết quả (thang Landis & Koch, dùng phổ biến):
    < 0.20 kém   0.21–0.40 tạm   0.41–0.60 trung bình   0.61–0.80 tốt   > 0.80 rất tốt

κ thấp không phải thất bại cần giấu: nó cho biết định nghĩa cơ chế trong LABELING.md chưa
đủ rõ. Đọc danh sách bất đồng, thống nhất cách hiểu, ghi lại vào LABELING.md, rồi báo
cáo cả κ trước lẫn sau khi thống nhất.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from config import DATA_DIR
from evaluation import EpisodeLabelError, label_agreement, load_episodes
from kb import get_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Cohen's κ giữa hai người gán nhãn")
    parser.add_argument("--first", default=str(DATA_DIR / "eval" / "episodes.yaml"))
    parser.add_argument("--second", default=str(DATA_DIR / "eval" / "episodes.annotator2.yaml"))
    args = parser.parse_args()

    kb = get_knowledge_base()
    try:
        first = load_episodes(Path(args.first), kb)
        second = load_episodes(Path(args.second), kb)
    except (FileNotFoundError, EpisodeLabelError) as exc:
        print(f"✗ {exc}")
        return 2

    result = label_agreement(first, second, kb)
    print(f"Episode cả hai đã gán nhãn : {result.common}")
    print(f"Chỉ người thứ nhất / thứ hai : {result.only_first} / {result.only_second}")
    if not result.common:
        print("\nChưa có episode chung nào ở status: labeled.")
        return 1

    def show(value):
        return (
            "không xác định (hai người luôn chọn cùng một giá trị)"
            if value is None
            else f"{value:.2f}"
        )

    print(f"\nκ vai trò cơ chế    : {show(result.mechanism_kappa)}")
    print(f"κ loại câu trả lời  : {show(result.outcome_kappa)}")

    if result.disagreements:
        print(f"\nBất đồng ({len(result.disagreements)}):")
        for line in result.disagreements:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
