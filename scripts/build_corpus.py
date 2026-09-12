"""Xây corpus khoa học từ OpenAlex.

Truy vấn = `rag_query` của TỪNG cơ chế trong knowledge/mechanisms.yaml, cộng thêm
các truy vấn nền trong rag/corpus.py. Nhờ vậy corpus được xây đúng theo nhu cầu của
reasoning engine chứ không phải một đống tài liệu chung chung.

    python scripts/build_corpus.py --per-query 20
    python scripts/build_corpus.py --per-query 30 --year-min 2018 --oa-only
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from config import get_settings
from kb import get_knowledge_base
from rag.corpus import (
    SEED_QUERIES,
    build_corpus,
    corpus_stats,
    licensing_report,
    save_corpus,
    snowball,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tải metadata + abstract từ OpenAlex")
    parser.add_argument("--per-query", type=int, default=20, help="Số bài mỗi truy vấn")
    parser.add_argument("--year-min", type=int, default=2015)
    parser.add_argument("--oa-only", action="store_true", help="Chỉ lấy bài open-access")
    parser.add_argument("--out", default=None, help="Mặc định data/corpus/papers.jsonl")
    parser.add_argument(
        "--snowball",
        action="store_true",
        help="Mở rộng theo danh mục tham khảo của các bài hạt giống (tăng độ phủ)",
    )
    parser.add_argument(
        "--snowball-seeds",
        type=int,
        default=30,
        help="Số bài hạt giống dùng để snowball (mặc định 30)",
    )
    parser.add_argument(
        "--snowball-min-citations",
        type=int,
        default=5,
        help="Ứng viên snowball phải có ít nhất ngần này trích dẫn",
    )
    args = parser.parse_args()

    settings = get_settings()
    kb = get_knowledge_base()

    mechanism_queries = [
        " ".join(m.rag_query.split()) for m in kb.mechanisms.values() if m.rag_query.strip()
    ]
    queries = list(dict.fromkeys(mechanism_queries + SEED_QUERIES))

    print(
        f"Truy vấn: {len(queries)} ({len(mechanism_queries)} từ cơ chế + {len(SEED_QUERIES)} nền)"
    )
    if not settings.openalex_mailto:
        print("⚠ Chưa đặt OPENALEX_MAILTO trong .env — sẽ bị giới hạn tốc độ nặng hơn.")
    print()

    papers = build_corpus(
        queries,
        per_query=args.per_query,
        year_min=args.year_min,
        with_references=args.snowball,
    )

    if args.snowball:
        print()
        seeds = sorted(papers, key=lambda p: p.cited_by_count, reverse=True)
        found = snowball(
            seeds,
            year_min=args.year_min,
            min_citations=args.snowball_min_citations,
            max_seeds=args.snowball_seeds,
        )
        known = {(p.doi or p.paper_id).lower() for p in papers}
        added = [p for p in found if (p.doi or p.paper_id).lower() not in known]
        papers.extend(added)
        print(f"  Snowball bổ sung {len(added)} bài mới (tổng {len(papers)}).")

    if args.oa_only:
        before = len(papers)
        papers = [p for p in papers if p.is_open_access]
        print(f"\nLọc open-access: {before} → {len(papers)}")

    out = _resolve_out(args.out, settings)
    save_corpus(papers, out)

    print(f"\nĐã lưu: {out}")
    for key, value in corpus_stats(papers).items():
        print(f"  {key:<20} {value}")

    print("\n" + "=" * 60)
    print("HỒ SƠ BẢN QUYỀN (đưa vào phụ lục khóa luận)")
    print("=" * 60)
    print(licensing_report(papers))

    print("\nBước tiếp theo: python scripts/index_corpus.py")
    return 0


def _resolve_out(out: str | None, settings):
    from pathlib import Path

    return Path(out) if out else settings.corpus_dir / "papers.jsonl"


if __name__ == "__main__":
    raise SystemExit(main())
