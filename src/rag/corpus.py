"""Xây corpus khoa học từ OpenAlex.

Vì sao OpenAlex chứ không phải Google Scholar: OpenAlex có API mở, không hạn ngạch
ngặt, chỉ cần email để vào "polite pool", và trả về đủ metadata + abstract + trạng
thái open-access. Google Scholar KHÔNG có API chính thức; cào tự động vi phạm điều
khoản sử dụng — chỉ dùng để tra thủ công.

Chiến lược: truy vấn theo chính `rag_query` của từng cơ chế trong
`knowledge/mechanisms.yaml`, cộng thêm vài truy vấn địa phương cho Hà Nội / ĐNÁ.
Nhờ vậy corpus được xây ĐÚNG theo nhu cầu của reasoning engine, không phải một
đống tài liệu chung chung.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import requests

from config import get_settings

from .models import Paper

OPENALEX_URL = "https://api.openalex.org/works"

#: Truy vấn bổ sung ngoài rag_query của các cơ chế — phủ bối cảnh địa phương và
#: nền phương pháp luận mà khóa luận cần trích dẫn.
SEED_QUERIES: list[str] = [
    "Hanoi PM2.5 air pollution episode meteorology Vietnam",
    "Red River Delta crop residue burning air quality",
    "Southeast Asia urban particulate matter source apportionment",
    "PM2.5 source apportionment positive matrix factorization urban",
    "explainable machine learning PM2.5 forecasting SHAP feature attribution",
    "northeast monsoon cold surge air quality northern Vietnam",
    "winter haze episode boundary layer meteorology East Asia",
]


class OpenAlexError(RuntimeError):
    pass


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex trả abstract dạng inverted index để lách bản quyền — dựng lại.

    `{"Low": [0], "boundary": [1], ...}` → "Low boundary ...".
    """
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, indices in inverted_index.items():
        for i in indices:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _parse_work(work: dict, query: str) -> Paper | None:
    title = (work.get("title") or work.get("display_name") or "").strip()
    if not title:
        return None

    doi = work.get("doi")
    if doi:
        doi = doi.replace("https://doi.org/", "")

    oa = work.get("open_access") or {}
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (work.get("authorships") or [])
    ]
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")

    return Paper(
        paper_id=work.get("id", "").rsplit("/", 1)[-1] or (doi or title[:64]),
        doi=doi,
        title=title,
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=[a for a in authors if a],
        year=work.get("publication_year"),
        venue=venue,
        url=oa.get("oa_url") or work.get("doi"),
        is_open_access=bool(oa.get("is_oa")),
        source="openalex",
        topics=[(t.get("display_name") or "") for t in (work.get("topics") or [])][:5],
        cited_by_count=work.get("cited_by_count", 0) or 0,
        query_tags=[query],
    )


def search_openalex(
    query: str,
    limit: int = 25,
    year_min: int = 2015,
    open_access_only: bool = False,
    timeout: int = 60,
) -> list[Paper]:
    """Tìm tài liệu trên OpenAlex cho một truy vấn."""
    settings = get_settings()
    filters = [f"from_publication_date:{year_min}-01-01"]
    if open_access_only:
        filters.append("is_oa:true")

    params = {
        "search": query,
        "filter": ",".join(filters),
        "per-page": min(limit, 200),
        "sort": "relevance_score:desc",
    }
    if settings.openalex_mailto:
        params["mailto"] = settings.openalex_mailto

    response = requests.get(
        OPENALEX_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": f"EG-XAQ/0.1 (mailto:{settings.openalex_mailto or 'n/a'})"},
    )
    if response.status_code != 200:
        raise OpenAlexError(f"OpenAlex trả {response.status_code}: {response.text[:200]}")

    papers = [_parse_work(w, query) for w in response.json().get("results", [])]
    return [p for p in papers if p is not None]


def build_corpus(
    queries: Iterable[str],
    per_query: int = 20,
    year_min: int = 2015,
    pause_s: float = 0.35,
) -> list[Paper]:
    """Chạy nhiều truy vấn, khử trùng lặp theo DOI/paper_id.

    Bài xuất hiện ở nhiều truy vấn sẽ gộp `query_tags` — chỉ số hữu ích: bài nào
    phục vụ nhiều cơ chế thì thường là tổng quan tốt, đáng đọc kỹ.
    """
    collected: dict[str, Paper] = {}

    for query in queries:
        try:
            results = search_openalex(query, limit=per_query, year_min=year_min)
        except (OpenAlexError, requests.RequestException) as exc:
            print(f"  [bỏ qua] '{query[:50]}...': {exc}")
            continue

        for paper in results:
            key = (paper.doi or paper.paper_id).lower()
            if key in collected:
                existing = collected[key]
                existing.query_tags = sorted(set(existing.query_tags) | set(paper.query_tags))
            else:
                collected[key] = paper

        print(f"  '{query[:60]}...' → {len(results)} bài (tổng {len(collected)})")
        time.sleep(pause_s)

    return list(collected.values())


def save_corpus(papers: list[Paper], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for paper in papers:
            fh.write(paper.model_dump_json() + "\n")


def load_corpus(path: Path) -> Iterator[Paper]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có corpus tại {path}. Chạy: python scripts/build_corpus.py"
        )
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Paper.model_validate_json(line)


def corpus_stats(papers: list[Paper]) -> dict[str, int | float]:
    """Thống kê để đưa vào chương phương pháp của khóa luận."""
    with_abstract = sum(1 for p in papers if p.abstract)
    open_access = sum(1 for p in papers if p.is_open_access)
    years = [p.year for p in papers if p.year]
    return {
        "papers": len(papers),
        "with_abstract": with_abstract,
        "open_access": open_access,
        "year_min": min(years) if years else 0,
        "year_max": max(years) if years else 0,
        "median_citations": sorted(p.cited_by_count for p in papers)[len(papers) // 2]
        if papers
        else 0,
    }
