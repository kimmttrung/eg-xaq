"""Cấu trúc dữ liệu của corpus khoa học.

⚖️ BẢN QUYỀN — đọc kỹ trước khi sửa (docs/05-rag.md §1)
======================================================

Corpus chỉ lưu toàn văn của tài liệu CÓ GIẤY PHÉP CHO PHÉP LƯU LẠI. Với mọi tài
liệu khác, corpus chỉ lưu metadata + abstract.

Bẫy quan trọng nhất: **`is_open_access = True` KHÔNG có nghĩa là được phép lưu.**
OpenAlex đánh dấu cả "bronze OA" là is_oa — tức bài đọc miễn phí trên web nhà xuất
bản nhưng KHÔNG kèm giấy phép mở nào. Nhà xuất bản có thể rút quyền đọc miễn phí
bất cứ lúc nào, và việc sao chép lại toàn văn vẫn là vi phạm bản quyền.

Vì vậy điều kiện để lưu `full_text` là có GIẤY PHÉP tường minh (Creative Commons
hoặc public domain), không phải chỉ là cờ is_oa. `Paper.validate_licensing()` thực
thi ràng buộc này ngay ở tầng dữ liệu — không phải cho đẹp, mà để một lỗi lúc thu
thập không âm thầm biến thành vấn đề pháp lý của cả khóa luận.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, model_validator

#: Giấy phép cho phép lưu và trích lại nguyên văn.
#: Mọi biến thể Creative Commons đều cho phép sao chép nguyên văn kèm ghi công —
#: kể cả NC (phi thương mại, khóa luận thỏa mãn) và ND (không tạo tác phẩm phái
#: sinh; trích nguyên văn không phải tác phẩm phái sinh).
REDISTRIBUTABLE_LICENSE_PREFIXES = ("cc-", "cc0", "public-domain", "pd")

#: Giấy phép đọc-được-nhưng-không-được-lưu. Đây chính là bronze OA.
#: `publisher-specific-oa` nghĩa là "nhà xuất bản cho đọc theo điều khoản riêng".
READ_ONLY_LICENSES = ("publisher-specific-oa", "other-oa", "no-license")


def is_redistributable(license_code: str | None) -> bool:
    """Giấy phép này có cho phép lưu lại toàn văn trong corpus không?

    Không có giấy phép → KHÔNG. Mặc định luôn là phía an toàn: thiếu thông tin
    giấy phép được coi là không được phép, chứ không phải được phép.
    """
    if not license_code:
        return False
    code = license_code.strip().lower()
    if code in READ_ONLY_LICENSES:
        return False
    return code.startswith(REDISTRIBUTABLE_LICENSE_PREFIXES)


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

    # --- hồ sơ bản quyền: bằng chứng tuân thủ, phải đưa vào phụ lục khóa luận ---
    license: str | None = Field(
        None, description="Mã giấy phép từ OpenAlex, ví dụ 'cc-by'. None = không rõ → coi như đóng"
    )
    oa_status: str | None = Field(
        None,
        description="gold | hybrid | green | bronze | closed (bronze = đọc được, KHÔNG được lưu)",
    )
    pdf_url: str | None = Field(None, description="Link PDF hợp pháp, nếu có")
    referenced_works: list[str] = Field(
        default_factory=list,
        description="ID OpenAlex của tài liệu tham khảo — dùng cho snowball, không đưa vào index",
    )

    def may_store_full_text(self) -> bool:
        """Có được phép lưu toàn văn bài này vào corpus không?

        Điều kiện là GIẤY PHÉP, không phải cờ is_oa — xem docstring đầu module.
        """
        return is_redistributable(self.license)

    @model_validator(mode="after")
    def validate_licensing(self) -> Paper:
        if self.full_text and not self.may_store_full_text():
            raise ValueError(
                f"Bài '{self.title[:60]}' có full_text nhưng giấy phép không cho phép lưu "
                f"(license={self.license!r}, oa_status={self.oa_status!r}, "
                f"is_open_access={self.is_open_access}).\n"
                "Chỉ lưu toàn văn khi có giấy phép Creative Commons / public domain. "
                "Đọc được miễn phí (bronze OA) KHÔNG đồng nghĩa được phép lưu lại "
                "(docs/05-rag.md §1). Bài này chỉ được lưu metadata + abstract."
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
