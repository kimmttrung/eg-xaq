"""Cấu trúc dữ liệu của corpus khoa học.

⚖️ Bản quyền (docs/05-rag.md §1): với tài liệu KHÔNG open-access, corpus chỉ lưu
metadata + abstract. Trường `full_text` chỉ được điền khi `is_open_access=True`.
`Paper.validate_licensing()` thực thi ràng buộc này — không phải để cho đẹp, mà
để bảo vệ tính hợp lệ của khóa luận.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, model_validator


class Paper(BaseModel):
    """Một tài liệu trong corpus."""

    paper_id: str
    doi: str | None = None
    title: str
    abstract: str | None = None
    full_text: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    is_open_access: bool = False
    source: str = "openalex"
    topics: list[str] = Field(default_factory=list)
    cited_by_count: int = 0
    query_tags: list[str] = Field(default_factory=list, description="rag_query nào tìm ra bài này")

    @model_validator(mode="after")
    def validate_licensing(self) -> Paper:
        if self.full_text and not self.is_open_access:
            raise ValueError(
                f"Bài '{self.title[:60]}' không open-access nhưng lại có full_text. "
                "Corpus chỉ được lưu full-text của bài OA (docs/05-rag.md §1)."
            )
        return self

    def author_string(self) -> str:
        if not self.authors:
            return "n/a"
        if len(self.authors) <= 2:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al."

    def content(self) -> str:
        """Phần văn bản dùng để index."""
        return self.full_text or self.abstract or self.title


class Chunk(BaseModel):
    """Một đoạn văn bản đã cắt, kèm đủ metadata để sinh trích dẫn cấp câu."""

    chunk_id: str
    paper_id: str
    text: str
    section: str = "abstract"
    position: int = 0

    # metadata nhân bản từ Paper để không phải join lúc truy xuất
    doi: str | None = None
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    is_open_access: bool = False

    def point_id(self) -> str:
        """ID ổn định cho Qdrant — cùng chunk index lại sẽ ghi đè, không nhân bản."""
        digest = hashlib.sha1(self.chunk_id.encode("utf-8")).hexdigest()
        return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

    def payload(self) -> dict:
        return self.model_dump(exclude={"chunk_id"}) | {"chunk_id": self.chunk_id}
