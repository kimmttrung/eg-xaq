"""Cắt tài liệu thành chunk để index.

Chiến lược: cắt THEO CẤU TRÚC trước, theo độ dài sau.

Vì sao không cắt cứng theo số token: một câu khẳng định cơ chế ("PBLH below 500 m
was associated with a 2.3-fold increase in PM2.5") bị cắt đôi thì đoạn nào cũng
mất nghĩa, và trích dẫn sẽ không chứng minh được điều nó nói. Giữ ranh giới câu và
ranh giới mục làm chất lượng trích dẫn tốt hơn hẳn.

Với bài chỉ có abstract (đa số, do ràng buộc bản quyền), một abstract thường vừa
gọn trong 1–2 chunk — và abstract vốn là phần đậm đặc kết luận nhất, nên đây
không phải hạn chế nghiêm trọng như thoạt nghe.
"""

from __future__ import annotations

import re

from .models import Chunk, Paper

#: Cắt theo số TỪ (xấp xỉ token). 320 từ ≈ 400–450 token với văn bản khoa học.
DEFAULT_CHUNK_WORDS = 320
DEFAULT_OVERLAP_RATIO = 0.15

#: Tiêu đề mục thường gặp trong bài báo khoa học.
_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s*)?"
    r"(abstract|introduction|background|(?:data\s+and\s+)?methods?|methodology|"
    r"materials?\s+and\s+methods?|results?|discussion|results?\s+and\s+discussion|"
    r"conclusions?|summary)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Tách văn bản thành [(tên mục, nội dung)]. Không nhận ra mục nào → một mục 'body'."""
    matches = list(_SECTION_PATTERN.finditer(text))
    if not matches:
        return [("body", text.strip())]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        head = text[: matches[0].start()].strip()
        if head:
            sections.append(("frontmatter", head))

    for i, match in enumerate(matches):
        name = match.group(1).lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


def split_by_words(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[str]:
    """Cắt theo độ dài nhưng KHÔNG cắt giữa câu.

    Gom câu cho tới khi chạm giới hạn, rồi bắt đầu chunk mới với phần chồng lấn
    là vài câu cuối của chunk trước — để một ý trải qua ranh giới vẫn còn ngữ cảnh.
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    overlap_words = int(chunk_words * overlap_ratio)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        n_words = len(sentence.split())
        if current and current_len + n_words > chunk_words:
            chunks.append(" ".join(current))
            # giữ lại các câu cuối làm phần chồng lấn
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                prev_len = len(prev.split())
                if tail_len + prev_len > overlap_words:
                    break
                tail.insert(0, prev)
                tail_len += prev_len
            current, current_len = tail, tail_len
        current.append(sentence)
        current_len += n_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_paper(
    paper: Paper,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    min_words: int = 25,
) -> list[Chunk]:
    """Một Paper → danh sách Chunk kèm đủ metadata để trích dẫn.

    Tiêu đề bài được ghép vào đầu mỗi chunk. Lý do: embedding của một đoạn Methods
    rời rạc rất dễ lạc chủ đề; có tiêu đề thì vector neo đúng vào chủ đề bài báo.
    """
    content = paper.content()
    if not content or len(content.split()) < min_words:
        return []

    chunks: list[Chunk] = []
    position = 0

    for section_name, section_text in split_sections(content):
        for piece in split_by_words(section_text, chunk_words, overlap_ratio):
            if len(piece.split()) < min_words:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}::{position}",
                    paper_id=paper.paper_id,
                    text=f"{paper.title}. {piece}",
                    section=section_name,
                    position=position,
                    doi=paper.doi,
                    title=paper.title,
                    authors=paper.author_string(),
                    year=paper.year,
                    venue=paper.venue,
                    url=paper.url,
                    is_open_access=paper.is_open_access,
                )
            )
            position += 1

    return chunks
