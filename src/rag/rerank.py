"""Rerank bằng cross-encoder.

Vì sao cần thêm một bước nữa sau vector search: bi-encoder (embedding) nén cả đoạn
văn vào một vector duy nhất trước khi biết truy vấn là gì, nên nó giỏi gọi ra ~50
ứng viên nhưng xếp hạng chưa sắc. Cross-encoder đọc ĐỒNG THỜI truy vấn và đoạn
văn nên phân biệt được "bài này nhắc tới boundary layer" với "bài này CHỨNG MINH
boundary layer thấp làm tăng PM2.5".

Trong hệ thống này, precision quan trọng hơn recall: một trích dẫn sai làm hỏng
tính "evidence-grounded" nặng hơn là thiếu một trích dẫn đúng.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from config import get_settings

from .store import SearchHit


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


class NoopReranker:
    """Giữ nguyên thứ hạng của vector search. Dùng làm baseline trong ablation RAG."""

    name = "noop"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        return hits[:top_k]


class CrossEncoderReranker:
    """bge-reranker qua sentence-transformers CrossEncoder. Import trễ."""

    name = "cross-encoder"

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reranker cần sentence-transformers. Cài: pip install -r requirements-embed.txt"
            ) from exc
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = self.model.predict([(query, hit.chunk.text) for hit in hits])
        rescored = [
            SearchHit(chunk=hit.chunk, score=_to_unit(float(score)))
            for hit, score in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


def _to_unit(score: float) -> float:
    """Cross-encoder trả logit; ép về [0,1] để cùng thang với cosine.

    Cần cùng thang vì `rag_support` được so với một ngưỡng duy nhất bất kể có
    rerank hay không — nếu hai thang khác nhau thì bật/tắt reranker sẽ vô tình
    đổi luôn độ chặt của cổng chặn, làm hỏng so sánh ablation.
    """
    import math

    return 1.0 / (1.0 + math.exp(-score))


def get_reranker(use_reranker: bool | None = None) -> Reranker:
    settings = get_settings()
    enabled = settings.use_reranker if use_reranker is None else use_reranker
    if not enabled:
        return NoopReranker()
    try:
        return CrossEncoderReranker()
    except ImportError:
        print("⚠ Không cài được cross-encoder, dùng NoopReranker.")
        return NoopReranker()
