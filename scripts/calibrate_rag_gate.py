"""Hiệu chỉnh cổng chặn `RetrievalConfig.min_score` trên corpus thật.

    pip install qdrant-client          # nhẹ, KHÔNG cần torch
    python scripts/calibrate_rag_gate.py
    python scripts/calibrate_rag_gate.py --top 5 --gates 0.4,0.5,0.6

Script này làm đúng việc mà cell cuối của notebook Kaggle làm, nhưng chạy tại máy:
`query_vectors.json` đã có sẵn vector của 12 truy vấn nên không cần GPU, không cần
tải model, không cần mở lại Kaggle.

VÌ SAO PHẢI HIỆU CHỈNH, KHÔNG ĐƯỢC ĐOÁN
=======================================

`min_score` mặc định 0.30 là giá trị của backend `hash` — một hàm băm n-gram
không hiểu ngữ nghĩa. Phân phối cosine của BGE-M3 hoàn toàn khác: tài liệu liên
quan thường rơi vào 0.6–0.85, nên giữ cổng 0.30 sẽ cho gần như mọi thứ lọt qua và
tầng gating mất tác dụng — đúng thứ INV-3 dựa vào.

Hội đồng sẽ hỏi "ngưỡng này ở đâu ra?". Câu trả lời phải là bảng quét dưới đây,
kèm việc bạn đã ĐỌC tiêu đề để xác nhận, chứ không phải một con số chọn cho đẹp.

CÁCH ĐỌC KẾT QUẢ
================
Điểm cao chưa chắc đúng. Một cơ chế có điểm 0.72 mà tiêu đề bài lạc đề nghĩa là
cổng đang cho rác lọt. Luôn đọc tiêu đề trước khi chốt.
"""

from __future__ import annotations

import argparse
import statistics

import _bootstrap  # noqa: F401
from config import get_settings
from kb import get_knowledge_base
from rag.embedding import get_embedder
from rag.store import get_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Hiệu chỉnh cổng chặn RAG trên corpus thật")
    parser.add_argument("--top", type=int, default=5, help="Số kết quả xem cho mỗi cơ chế")
    parser.add_argument(
        "--gates",
        default="0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7",
        help="Các mức cổng để quét, phân tách bằng dấu phẩy",
    )
    parser.add_argument("--store", default="qdrant", choices=["qdrant", "memory"])
    args = parser.parse_args()

    settings = get_settings()
    kb = get_knowledge_base()
    embedder = get_embedder()

    if embedder.name == "hash":
        print(
            "⚠ Đang dùng embedding backend 'hash'. Hiệu chỉnh cổng trên backend này\n"
            "  là vô nghĩa — kết quả không mang sang BGE-M3 được.\n"
            "  Đặt EGXAQ_EMBEDDING_BACKEND=precomputed trong .env.\n"
        )

    store = get_vector_store(args.store)
    total_points = store.count()
    print(f"Collection '{settings.qdrant_collection}': {total_points} điểm")
    print(f"Embedder: {embedder.name} (dim {embedder.dim})\n")

    mechanisms = [m for m in kb.mechanisms.values() if m.rag_query.strip()]
    results: dict[str, list[tuple[float, str]]] = {}

    for mech in mechanisms:
        query = " ".join(mech.rag_query.split())
        vector = embedder.encode([query], is_query=True)[0]
        hits = store.search(vector, limit=args.top)
        results[mech.id] = [(h.score, h.chunk.title or "(không tiêu đề)") for h in hits]

    _print_per_mechanism(results, args.top)
    gates = [float(g) for g in args.gates.split(",")]
    _print_gate_sweep(results, gates, len(mechanisms))
    _print_advice(results, gates)
    return 0


def _print_per_mechanism(results: dict[str, list[tuple[float, str]]], top: int) -> None:
    print("=" * 78)
    print(f"TOP {top} CHO TỪNG CƠ CHẾ — ĐỌC TIÊU ĐỀ, ĐỪNG CHỈ NHÌN ĐIỂM")
    print("=" * 78)
    for mech_id, hits in results.items():
        if not hits:
            print(f"\n  {mech_id}\n      (không có kết quả nào)")
            continue
        print(f"\n  {mech_id}")
        for score, title in hits:
            print(f"      {score:.3f}  {title[:66]}")


def _print_gate_sweep(
    results: dict[str, list[tuple[float, str]]], gates: list[float], n_mech: int
) -> None:
    """Mỗi mức cổng giữ lại bao nhiêu cơ chế, và trung bình mấy trích dẫn."""
    print("\n" + "=" * 78)
    print("QUÉT CỔNG")
    print("=" * 78)
    print(f"  {'cổng':<8}{'cơ chế có trích dẫn':<24}{'TB trích dẫn/cơ chế':<24}")
    print("  " + "-" * 60)

    for gate in gates:
        covered = 0
        kept_counts = []
        for hits in results.values():
            kept = [h for h in hits if h[0] >= gate]
            kept_counts.append(len(kept))
            if kept:
                covered += 1
        avg = statistics.mean(kept_counts) if kept_counts else 0.0
        bar = "█" * round(20 * covered / n_mech) if n_mech else ""
        print(f"  {gate:<8.2f}{covered:>3}/{n_mech:<20}{avg:>5.1f}{'':<18}{bar}")


def _print_advice(results: dict[str, list[tuple[float, str]]], gates: list[float]) -> None:
    best_scores = [hits[0][0] for hits in results.values() if hits]
    if not best_scores:
        print("\nKhông truy xuất được gì — kiểm tra lại collection và embedder.")
        return

    print("\n" + "=" * 78)
    print("THỐNG KÊ ĐIỂM KHỚP TỐT NHẤT (mỗi cơ chế một giá trị)")
    print("=" * 78)
    print(
        f"  min={min(best_scores):.3f}   p25={_pct(best_scores, 25):.3f}   "
        f"trung vị={statistics.median(best_scores):.3f}   "
        f"p75={_pct(best_scores, 75):.3f}   max={max(best_scores):.3f}"
    )

    print(
        "\nCÁCH CHỐT:\n"
        "  1. Nhìn bảng quét, tìm mức cổng mà số cơ chế có trích dẫn bắt đầu TỤT NHANH.\n"
        "     Cổng nên đặt ngay TRƯỚC chỗ tụt đó.\n"
        "  2. Quay lên phần TOP, kiểm tra các bài quanh mức cổng đó có thực sự nói về\n"
        "     cơ chế tương ứng không. Đây là bước không thể bỏ.\n"
        "  3. Cơ chế không có tài liệu phù hợp thì PHẢI bị chặn — rag_support = 0 và\n"
        "     narrator nói thiếu căn cứ. Đó là INV-3 hoạt động đúng, không phải lỗi.\n"
        "\n  Chốt xong, sửa `min_score` trong src/rag/retrieve.py (RetrievalConfig) và\n"
        "  ghi lại lý do chọn — hội đồng sẽ hỏi con số này ở đâu ra."
    )


def _pct(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(percentile / 100 * (len(ordered) - 1))))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
