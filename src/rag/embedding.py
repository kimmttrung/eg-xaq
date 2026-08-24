"""Embedding cho corpus khoa học.

Hai backend:

- ``hf``   — sentence-transformers với model chuyên khoa học (mặc định BGE-M3, hỗ
             trợ đa ngôn ngữ nên xử lý được cả tài liệu tiếng Việt). ĐÂY là backend
             phải dùng cho mọi kết quả trong khóa luận.
- ``hash`` — embedding băm xác định, không cần tải model, chạy offline. CHỈ để
             unit test và để pipeline chạy được trên máy chưa cài torch.

⚠ Vì sao vẫn giữ backend ``hash``: nếu bắt buộc phải có torch mới chạy được test,
thì test sẽ không chạy trên CI và trên máy yếu, và trong thực tế là không ai chạy.
Backend hash giữ cho toàn bộ đường ống RAG luôn được kiểm thử. Nó KHÔNG có chất
lượng ngữ nghĩa — `hash_backend_warning()` nhắc điều đó ở mọi nơi cần thiết.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

from config import get_settings

_TOKEN = re.compile(r"[a-zA-ZÀ-ỹ0-9]+")


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


def hash_backend_warning() -> str:
    return (
        "⚠ Đang dùng embedding backend 'hash' — chỉ để test, KHÔNG có chất lượng "
        "ngữ nghĩa. Đặt EGXAQ_EMBEDDING_BACKEND=hf trước khi lấy số cho khóa luận."
    )


class HashEmbedder:
    """Băm n-gram từ vào vector thưa rồi chuẩn hóa L2.

    Đây thực chất là hashing trick / random projection: các văn bản dùng chung
    nhiều từ và cụm từ sẽ có cosine cao. Đủ để kiểm tra đường ống hoạt động, không
    đủ để hiểu ngữ nghĩa ("boundary layer" và "mixing height" sẽ không gần nhau).
    """

    name = "hash"

    def __init__(self, dim: int = 1024, ngram: int = 2) -> None:
        self.dim = dim
        self.ngram = ngram

    def _vector(self, text: str) -> np.ndarray:
        tokens = _TOKEN.findall(text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            return vec

        grams = list(tokens)
        for n in range(2, self.ngram + 1):
            grams += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))


class HFEmbedder:
    """sentence-transformers. Import trễ để repo chạy được khi chưa cài torch."""

    name = "hf"

    def __init__(self, model_name: str, dim: int | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - cần requirements-embed.txt
            raise ImportError(
                "Backend 'hf' cần sentence-transformers. Cài: "
                "pip install -r requirements-embed.txt"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = dim or self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Một số model (E5, BGE) cần tiền tố khác nhau cho truy vấn và tài liệu.
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)


def get_embedder(backend: str | None = None) -> Embedder:
    settings = get_settings()
    backend = backend or settings.embedding_backend

    if backend == "hash":
        return HashEmbedder(dim=settings.embedding_dim)
    if backend == "hf":
        return HFEmbedder(settings.embedding_model, dim=None)
    raise ValueError(f"Embedding backend không hợp lệ: {backend!r}")
