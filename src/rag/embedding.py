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
import json
import re
from pathlib import Path
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
                "Backend 'hf' cần sentence-transformers. Cài: pip install -r requirements-embed.txt"
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


def normalize_query(text: str) -> str:
    """Chuẩn hóa khoảng trắng của truy vấn.

    Bắt buộc dùng CHUNG một hàm ở cả hai phía (lúc precompute trên Kaggle và lúc
    tra cứu ở local), nếu không thì `rag_query` nhiều dòng trong YAML sẽ cho ra
    hai khóa khác nhau và không bao giờ khớp.
    """
    return " ".join(text.split())


class PrecomputedQueryEmbedder:
    """Tra vector truy vấn từ bảng đã tính sẵn — KHÔNG cần torch ở máy local.

    VÌ SAO LÀM ĐƯỢC
    ---------------
    Gated retrieval khiến tập truy vấn thành TẬP ĐÓNG: truy vấn không phải câu hỏi
    của người dùng mà là `rag_query` của cơ chế, lấy từ `knowledge/mechanisms.yaml`.
    Hiện có 12 cơ chế → đúng 12 truy vấn, biết trước hoàn toàn. Vậy nên chúng được
    embed sẵn một lần (cùng model, cùng lượt với tài liệu) rồi đóng gói thành một
    file ~150 KB.

    Hệ quả kiến trúc: việc nặng (BGE-M3 trên GPU) chạy ở nơi có GPU; máy local chỉ
    còn tra bảng và gọi Qdrant. Đây KHÔNG phải cách rút gọn chất lượng — vector tra
    ra là vector BGE-M3 thật, giống hệt cái mà HFEmbedder sẽ tính.

    RÀNG BUỘC
    ---------
    Sửa `rag_query` trong YAML → phải embed lại. Lớp này ném lỗi ngay khi gặp truy
    vấn lạ thay vì trả vector rỗng: một truy vấn không khớp mà im lặng sẽ cho ra
    "cơ chế này không có tài liệu hỗ trợ" — sai lệch đúng vào thứ khóa luận đang đo.
    """

    name = "precomputed"

    def __init__(self, vectors_path: Path | str) -> None:
        path = Path(vectors_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Chưa có file vector truy vấn: {path}\n"
                "Sinh bằng notebook Kaggle (notebooks/kaggle_index_corpus.py), "
                "tải về rồi đặt đường dẫn vào EGXAQ_QUERY_VECTORS."
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.model_name = payload.get("model", "unknown")
        self.dim = int(payload["dim"])
        self._table: dict[str, np.ndarray] = {
            normalize_query(text): np.asarray(vec, dtype=np.float32)
            for text, vec in payload["vectors"].items()
        }
        if not self._table:
            raise ValueError(f"{path} không chứa vector nào.")

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if not is_query:
            raise RuntimeError(
                "PrecomputedQueryEmbedder chỉ phục vụ TRUY VẤN. Việc embed tài liệu "
                "phải chạy trên Kaggle/GPU bằng backend 'hf' rồi nạp thẳng vào Qdrant."
            )

        vectors = []
        for text in texts:
            key = normalize_query(text)
            if key not in self._table:
                raise KeyError(
                    f"Truy vấn chưa có vector tính sẵn:\n  {key[:100]}…\n"
                    f"Bảng hiện có {len(self._table)} truy vấn (model {self.model_name}).\n"
                    "`rag_query` trong mechanisms.yaml đã đổi? Chạy lại notebook Kaggle "
                    "để sinh lại file vector."
                )
            vectors.append(self._table[key])
        return np.vstack(vectors)

    def known_queries(self) -> list[str]:
        return sorted(self._table)


def get_embedder(backend: str | None = None) -> Embedder:
    settings = get_settings()
    backend = backend or settings.embedding_backend

    if backend == "hash":
        return HashEmbedder(dim=settings.embedding_dim)
    if backend == "hf":
        return HFEmbedder(settings.embedding_model, dim=None)
    if backend == "precomputed":
        return PrecomputedQueryEmbedder(settings.query_vectors_path)
    raise ValueError(f"Embedding backend không hợp lệ: {backend!r}")
