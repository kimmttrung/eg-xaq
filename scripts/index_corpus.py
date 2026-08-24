"""Chunk → embed → nạp vào Qdrant.

    docker compose up -d qdrant
    python scripts/index_corpus.py
    python scripts/index_corpus.py --recreate --embedding hf

⚠ Mặc định dùng embedding backend từ .env. Nếu là `hash`, script sẽ cảnh báo:
kết quả khóa luận BẮT BUỘC dùng backend `hf`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from config import get_settings
from rag.chunking import chunk_paper
from rag.corpus import load_corpus
from rag.embedding import get_embedder, hash_backend_warning
from rag.store import get_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Index corpus vào Qdrant")
    parser.add_argument("--corpus", default=None, help="Mặc định data/corpus/papers.jsonl")
    parser.add_argument("--embedding", default=None, choices=["hash", "hf"])
    parser.add_argument("--store", default="qdrant", choices=["qdrant", "memory"])
    parser.add_argument("--recreate", action="store_true", help="Xóa collection cũ trước khi nạp")
    parser.add_argument("--chunk-words", type=int, default=320)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    corpus_path = Path(args.corpus) if args.corpus else settings.corpus_dir / "papers.jsonl"

    papers = list(load_corpus(corpus_path))
    print(f"Corpus: {len(papers)} bài từ {corpus_path}")

    chunks = [c for p in papers for c in chunk_paper(p, chunk_words=args.chunk_words)]
    print(f"Chunk : {len(chunks)} đoạn")
    if not chunks:
        print("Không có chunk nào — kiểm tra lại corpus.")
        return 1

    embedder = get_embedder(args.embedding)
    print(f"Embed : {embedder.name} (dim={embedder.dim})")
    if embedder.name == "hash":
        print("  " + hash_backend_warning())

    store = get_vector_store(args.store)
    store.ensure_collection(dim=embedder.dim, recreate=args.recreate)

    total = 0
    for start in range(0, len(chunks), args.batch):
        batch = chunks[start : start + args.batch]
        vectors = embedder.encode([c.text for c in batch], is_query=False)
        total += store.upsert(batch, vectors)
        print(f"  đã nạp {total}/{len(chunks)}", end="\r")

    print(f"\nXong. Collection '{settings.qdrant_collection}' có {store.count()} điểm.")
    print("Kiểm tra tại: http://localhost:6333/dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
