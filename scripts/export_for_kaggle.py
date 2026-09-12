"""Đóng gói corpus để mang lên Kaggle/Colab embed bằng GPU.

    python scripts/export_for_kaggle.py
    python scripts/export_for_kaggle.py --out data/corpus/kaggle

PHÂN CÔNG CÔNG VIỆC
===================

    LOCAL (script này)          KAGGLE (T4)                 LOCAL (lúc chạy)
    ─────────────────────       ──────────────────────      ──────────────────
    tải corpus                  đọc chunks.jsonl            đọc query_vectors
    cắt chunk                   embed BGE-M3 trên GPU       gọi Qdrant Cloud
    dựng payload         ──►    nạp thẳng vào Qdrant  ──►   KHÔNG cần torch
    trích rag_query             embed 12 truy vấn
                                xuất query_vectors.json

Vì sao chunking ở lại local: nó là Python thuần, không cần GPU, và quan trọng hơn
— nó quyết định `chunk_id`, `point_id` và `payload`. Nếu để notebook tự cắt lại
thì logic bị nhân đôi và hai bên sẽ trôi khỏi nhau lúc nào không biết. Xuất sẵn
payload nghĩa là notebook chỉ còn đúng một việc: biến `text` thành vector.

Hai file xuất ra:

    chunks.jsonl      mỗi dòng {point_id, text, payload} — nạp thẳng vào Qdrant
    queries.json      danh sách rag_query của các cơ chế, đã chuẩn hóa khoảng trắng
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from config import get_settings
from kb import get_knowledge_base
from rag.chunking import chunk_paper
from rag.corpus import load_corpus
from rag.embedding import normalize_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Xuất chunk + truy vấn cho Kaggle")
    parser.add_argument("--corpus", default=None, help="Mặc định data/corpus/papers.jsonl")
    parser.add_argument("--out", default=None, help="Thư mục xuất, mặc định data/corpus/kaggle")
    parser.add_argument("--chunk-words", type=int, default=320)
    args = parser.parse_args()

    settings = get_settings()
    corpus_path = Path(args.corpus) if args.corpus else settings.corpus_dir / "papers.jsonl"
    out_dir = Path(args.out) if args.out else settings.corpus_dir / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = list(load_corpus(corpus_path))
    chunks = [c for p in papers for c in chunk_paper(p, chunk_words=args.chunk_words)]
    if not chunks:
        print("Không có chunk nào — kiểm tra lại corpus.")
        return 1

    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            record = {
                "point_id": chunk.point_id(),
                "text": chunk.text,
                "payload": chunk.payload(),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    kb = get_knowledge_base()
    queries = sorted(
        {normalize_query(m.rag_query) for m in kb.mechanisms.values() if m.rag_query.strip()}
    )
    queries_path = out_dir / "queries.json"
    queries_path.write_text(
        json.dumps(
            {
                "model": settings.embedding_model,
                "dim": settings.embedding_dim,
                "mechanisms_version": kb.feature_map.version,
                "queries": queries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Corpus : {len(papers)} bài từ {corpus_path}")
    print(f"Chunk  : {len(chunks)} đoạn → {chunks_path}")
    print(f"Truy vấn: {len(queries)} → {queries_path}")
    print(f"\nCollection Qdrant : {settings.qdrant_collection}")
    print(f"Model embedding   : {settings.embedding_model} (dim {settings.embedding_dim})")
    print(
        "\nBước tiếp theo:\n"
        f"  1. Upload thư mục {out_dir} lên Kaggle làm Dataset\n"
        "  2. Chạy notebooks/kaggle_index_corpus.py trên Kaggle (GPU T4 + Internet ON)\n"
        "  3. Tải query_vectors.json về đặt vào data/corpus/\n"
        "  4. Đặt trong .env:\n"
        "       EGXAQ_EMBEDDING_BACKEND=precomputed\n"
        "       EGXAQ_QDRANT_URL=<url cluster Qdrant Cloud>\n"
        "       EGXAQ_QDRANT_API_KEY=<api key>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
