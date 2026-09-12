"""Scientific RAG — neo mỗi cơ chế vào tài liệu khoa học trích dẫn được.

Khác biệt cốt lõi so với RAG thông thường: câu truy vấn KHÔNG phải câu hỏi của
người dùng, mà là `rag_query` của cơ chế đã được rule kích hoạt. Xem `retrieve.py`.
"""

from .chunking import chunk_paper
from .corpus import licensing_report, snowball
from .embedding import get_embedder
from .models import Chunk, Paper, is_redistributable
from .retrieve import GatedRetriever, RetrievalConfig
from .store import QdrantStore

__all__ = [
    "Chunk",
    "GatedRetriever",
    "Paper",
    "QdrantStore",
    "RetrievalConfig",
    "chunk_paper",
    "get_embedder",
    "is_redistributable",
    "licensing_report",
    "snowball",
]
