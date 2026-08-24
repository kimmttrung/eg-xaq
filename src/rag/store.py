"""Vector store — Qdrant là backend chính thức của dự án.

`InMemoryStore` chỉ tồn tại để unit test chạy được mà không cần Docker. Nó KHÔNG
phải phương án dự phòng lúc chạy thật: nếu Qdrant không kết nối được, `QdrantStore`
ném lỗi rõ ràng chứ không âm thầm chuyển sang bộ nhớ (CLAUDE.md §8 bẫy #5) — im
lặng thất bại ở tầng này sẽ tạo ra câu trả lời không có trích dẫn mà không ai biết.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from config import get_settings

from .models import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def ensure_collection(self, dim: int, recreate: bool = False) -> None: ...
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int: ...
    def search(self, vector: np.ndarray, limit: int = 10) -> list[SearchHit]: ...
    def count(self) -> int: ...


# =============================================================================
# Qdrant
# =============================================================================


class QdrantUnavailable(RuntimeError):
    """Qdrant chưa chạy hoặc chưa cài client."""


class QdrantStore:
    name = "qdrant"

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ) -> None:
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise QdrantUnavailable(
                "Chưa cài qdrant-client. Cài: pip install -r requirements-rag.txt"
            ) from exc

        self._client = QdrantClient(url=self.url, api_key=api_key or settings.qdrant_api_key)

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client.http import models as qmodels

        try:
            exists = self._client.collection_exists(self.collection)
        except Exception as exc:
            raise QdrantUnavailable(
                f"Không kết nối được Qdrant tại {self.url}. "
                f"Chạy: docker compose up -d qdrant\n  Chi tiết: {exc}"
            ) from exc

        if exists and recreate:
            self._client.delete_collection(self.collection)
            exists = False

        if not exists:
            self._client.create_collection(
                collection_name=self.collection,
                # COSINE vì mọi embedder ở đây đều trả vector đã chuẩn hóa L2.
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        from qdrant_client.http import models as qmodels

        if len(chunks) != len(vectors):
            raise ValueError(f"Lệch số lượng: {len(chunks)} chunk vs {len(vectors)} vector")
        if not chunks:
            return 0

        points = [
            qmodels.PointStruct(
                id=chunk.point_id(), vector=vector.tolist(), payload=chunk.payload()
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(self, vector: np.ndarray, limit: int = 10) -> list[SearchHit]:
        results = self._client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
        ).points
        return [SearchHit(chunk=_chunk_from_payload(r.payload), score=float(r.score)) for r in results]

    def count(self) -> int:
        return self._client.count(self.collection, exact=True).count


# =============================================================================
# In-memory — CHỈ DÙNG CHO TEST
# =============================================================================


class InMemoryStore:
    """Tìm kiếm cosine bằng numpy. Không bền vững, không mở rộng được.

    Tồn tại để `tests/` chạy được mà không cần Docker. Không dùng để lấy số liệu.
    """

    name = "memory"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        if recreate:
            self._chunks, self._vectors = [], None

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"Lệch số lượng: {len(chunks)} chunk vs {len(vectors)} vector")
        if not chunks:
            return 0
        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        return len(chunks)

    def search(self, vector: np.ndarray, limit: int = 10) -> list[SearchHit]:
        if self._vectors is None or not len(self._chunks):
            return []
        scores = self._vectors @ vector
        order = np.argsort(-scores)[:limit]
        return [SearchHit(chunk=self._chunks[i], score=float(scores[i])) for i in order]

    def count(self) -> int:
        return len(self._chunks)


def _chunk_from_payload(payload: dict[str, Any] | None) -> Chunk:
    data = dict(payload or {})
    data.setdefault("chunk_id", "unknown")
    data.setdefault("paper_id", "unknown")
    data.setdefault("text", "")
    return Chunk.model_validate(data)


def get_vector_store(backend: str = "qdrant") -> VectorStore:
    if backend == "qdrant":
        return QdrantStore()
    if backend == "memory":
        return InMemoryStore()
    raise ValueError(f"Vector store không hợp lệ: {backend!r}")
