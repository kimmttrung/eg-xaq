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

from .models import Paper, is_redistributable

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


#: Cổng lọc miền cho ứng viên snowball. Tiêu đề hoặc abstract phải chứa ít nhất
#: một cụm ở đây thì mới giữ.
#:
#: Cố ý để RỘNG: mục đích là loại bài hoàn toàn lạc đề (y sinh, mùa đông hạt nhân,
#: vi sinh vật trong không khí) chứ không phải thay thế cổng ngữ nghĩa của RAG.
#: Lọc quá chặt ở đây sẽ giết mất chính những bài cơ chế mà snowball sinh ra để tìm.
DOMAIN_KEYWORDS: list[str] = [
    "pm2.5",
    "pm 2.5",
    "particulate",
    "aerosol",
    "air quality",
    "air pollution",
    "haze",
    "boundary layer",
    "mixing height",
    "mixing layer",
    "inversion",
    "stagnation",
    "ventilation",
    "dispersion",
    "deposition",
    "scavenging",
    "biomass burning",
    "crop residue",
    "meteorolog",
    "monsoon",
    "synoptic",
    "wind speed",
    "planetary boundary",
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


def _parse_work(work: dict, query: str, with_references: bool = False) -> Paper | None:
    title = (work.get("title") or work.get("display_name") or "").strip()
    if not title:
        return None

    doi = work.get("doi")
    if doi:
        doi = doi.replace("https://doi.org/", "")

    oa = work.get("open_access") or {}
    best_oa = work.get("best_oa_location") or {}
    authors = [
        (a.get("author") or {}).get("display_name", "") for a in (work.get("authorships") or [])
    ]
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")

    references: list[str] = []
    if with_references:
        references = [ref.rsplit("/", 1)[-1] for ref in (work.get("referenced_works") or [])]

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
        # --- hồ sơ bản quyền ---
        license=best_oa.get("license"),
        oa_status=oa.get("oa_status"),
        pdf_url=best_oa.get("pdf_url"),
        referenced_works=references,
    )


def _request(params: dict, timeout: int = 60) -> dict:
    """Một lần gọi OpenAlex, kèm định danh lịch sự (polite pool).

    OpenAlex không yêu cầu API key. Gửi kèm email đưa ta vào "polite pool" — hàng
    đợi riêng, hạn mức rộng hơn và ổn định hơn. Đây là điều khoản sử dụng của họ,
    không phải mẹo lách: cứ đặt OPENALEX_MAILTO trong .env.
    """
    settings = get_settings()
    if settings.openalex_mailto:
        params = {**params, "mailto": settings.openalex_mailto}

    response = requests.get(
        OPENALEX_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": f"EG-XAQ/0.1 (mailto:{settings.openalex_mailto or 'n/a'})"},
    )
    if response.status_code != 200:
        raise OpenAlexError(f"OpenAlex trả {response.status_code}: {response.text[:200]}")
    return response.json()


def search_openalex(
    query: str,
    limit: int = 25,
    year_min: int = 2015,
    open_access_only: bool = False,
    timeout: int = 60,
    with_references: bool = False,
) -> list[Paper]:
    """Tìm tài liệu trên OpenAlex cho một truy vấn."""
    filters = [f"from_publication_date:{year_min}-01-01"]
    if open_access_only:
        filters.append("is_oa:true")

    data = _request(
        {
            "search": query,
            "filter": ",".join(filters),
            "per-page": min(limit, 200),
            "sort": "relevance_score:desc",
        },
        timeout=timeout,
    )
    papers = [_parse_work(w, query, with_references) for w in data.get("results", [])]
    return [p for p in papers if p is not None]


def fetch_works_by_ids(
    work_ids: Iterable[str],
    query_tag: str = "snowball",
    batch_size: int = 50,
    pause_s: float = 0.35,
    timeout: int = 60,
) -> list[Paper]:
    """Lấy nhiều tài liệu theo ID OpenAlex, gọi theo lô.

    OpenAlex cho phép OR tối đa 50 giá trị trong một filter, nên 500 tài liệu tham
    khảo chỉ tốn 10 request thay vì 500. Đây là điểm khiến snowball rẻ.
    """
    ids = [wid for wid in dict.fromkeys(work_ids) if wid]
    collected: list[Paper] = []

    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        try:
            data = _request(
                {"filter": "openalex_id:" + "|".join(batch), "per-page": batch_size},
                timeout=timeout,
            )
        except (OpenAlexError, requests.RequestException) as exc:
            print(f"  [bỏ qua lô {start // batch_size + 1}]: {exc}")
            continue

        for work in data.get("results", []):
            paper = _parse_work(work, query_tag)
            if paper is not None:
                collected.append(paper)
        time.sleep(pause_s)

    return collected


def snowball(
    seeds: list[Paper],
    year_min: int = 2015,
    min_citations: int = 5,
    max_seeds: int = 30,
    keywords: Iterable[str] | None = None,
) -> list[Paper]:
    """Mở rộng corpus theo danh mục tham khảo của các bài hạt giống.

    VÌ SAO SNOWBALL, KHÔNG PHẢI TÌM THÊM TỪ KHÓA
    --------------------------------------------
    Tìm kiếm theo từ khóa có trần: đổi cách diễn đạt truy vấn vẫn ra gần đúng tập
    bài đó. Danh mục tham khảo thì đi theo *cấu trúc thật của tài liệu* — bài tổng
    quan tốt đã làm sẵn việc sàng lọc cho ta. Đây là cách đạt độ phủ cao với chi
    phí thấp, và là phương pháp chuẩn trong systematic review nên bảo vệ được
    trước hội đồng.

    LƯU Ý ĐỘ CHÍNH XÁC
    ------------------
    Danh mục tham khảo kéo về rất nhiều thứ lạc đề (một bài haze Bắc Kinh có thể
    trích cả bài về mùa đông hạt nhân). Vì vậy PHẢI lọc: theo năm, theo lượng trích
    dẫn, và theo từ khóa miền. Không lọc thì corpus loãng và cổng RAG sẽ phải gánh.
    """
    seed_ids = [p.paper_id for p in seeds[:max_seeds] if p.paper_id.startswith("W")]
    if not seed_ids:
        return []

    print(f"  Snowball: đọc danh mục tham khảo của {len(seed_ids)} bài hạt giống…")
    seeds_full = fetch_works_by_ids(seed_ids, query_tag="snowball-seed")

    reference_ids: list[str] = []
    for paper in seeds_full:
        reference_ids.extend(paper.referenced_works)
    # `fetch_works_by_ids` không xin references nên phải lấy lại từ chính seeds
    for paper in seeds[:max_seeds]:
        reference_ids.extend(paper.referenced_works)

    known = {p.paper_id for p in seeds}
    candidate_ids = [wid for wid in dict.fromkeys(reference_ids) if wid not in known]
    if not candidate_ids:
        print("  Snowball: hạt giống chưa có danh mục tham khảo (cần with_references=True).")
        return []

    print(f"  Snowball: {len(candidate_ids)} ứng viên, đang tải…")
    candidates = fetch_works_by_ids(candidate_ids)

    kept = [
        paper
        for paper in candidates
        if _passes_snowball_gate(paper, year_min, min_citations, keywords)
    ]
    print(f"  Snowball: giữ {len(kept)}/{len(candidates)} sau khi lọc.")
    return kept


def _passes_snowball_gate(
    paper: Paper,
    year_min: int,
    min_citations: int,
    keywords: Iterable[str] | None,
) -> bool:
    """Cổng lọc ứng viên snowball — giữ độ chính xác của corpus."""
    if paper.year is not None and paper.year < year_min:
        return False
    if paper.cited_by_count < min_citations:
        return False

    terms = list(keywords) if keywords is not None else DOMAIN_KEYWORDS
    if not terms:
        return True

    haystack = f"{paper.title} {paper.abstract or ''}".lower()
    return any(term in haystack for term in terms)


def build_corpus(
    queries: Iterable[str],
    per_query: int = 20,
    year_min: int = 2015,
    pause_s: float = 0.35,
    with_references: bool = False,
) -> list[Paper]:
    """Chạy nhiều truy vấn, khử trùng lặp theo DOI/paper_id.

    Bài xuất hiện ở nhiều truy vấn sẽ gộp `query_tags` — chỉ số hữu ích: bài nào
    phục vụ nhiều cơ chế thì thường là tổng quan tốt, đáng đọc kỹ.

    `with_references=True` xin thêm danh mục tham khảo để `snowball()` dùng sau.
    """
    collected: dict[str, Paper] = {}

    for query in queries:
        try:
            results = search_openalex(
                query, limit=per_query, year_min=year_min, with_references=with_references
            )
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
        raise FileNotFoundError(f"Chưa có corpus tại {path}. Chạy: python scripts/build_corpus.py")
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Paper.model_validate_json(line)


def corpus_stats(papers: list[Paper]) -> dict[str, int | float]:
    """Thống kê để đưa vào chương phương pháp của khóa luận."""
    with_abstract = sum(1 for p in papers if p.abstract)
    open_access = sum(1 for p in papers if p.is_open_access)
    storable = sum(1 for p in papers if p.may_store_full_text())
    years = [p.year for p in papers if p.year]
    return {
        "papers": len(papers),
        "with_abstract": with_abstract,
        "open_access": open_access,
        "full_text_allowed": storable,
        "year_min": min(years) if years else 0,
        "year_max": max(years) if years else 0,
        "median_citations": sorted(p.cited_by_count for p in papers)[len(papers) // 2]
        if papers
        else 0,
    }


def licensing_report(papers: list[Paper]) -> str:
    """Bảng phân bố giấy phép — bằng chứng tuân thủ cho phụ lục khóa luận.

    Hội đồng có quyền hỏi "tài liệu này lấy ở đâu, có được phép không?". Bảng này
    là câu trả lời: mỗi bài đều ghi rõ giấy phép, và số bài được lưu toàn văn luôn
    khớp với số bài có giấy phép cho phép.
    """
    from collections import Counter

    by_license = Counter((p.license or "(không có giấy phép)") for p in papers)
    by_status = Counter((p.oa_status or "(không rõ)") for p in papers)
    storable = sum(1 for p in papers if p.may_store_full_text())

    lines = [f"Tổng: {len(papers)} bài", ""]
    lines.append("Theo giấy phép:")
    for code, count in by_license.most_common():
        mark = (
            "toàn văn OK"
            if is_redistributable(None if code.startswith("(") else code)
            else "chỉ abstract"
        )
        lines.append(f"  {code:<28} {count:>4}   {mark}")

    lines.append("")
    lines.append("Theo trạng thái open access:")
    for status, count in by_status.most_common():
        note = "  ← đọc được nhưng KHÔNG được lưu toàn văn" if status == "bronze" else ""
        lines.append(f"  {status:<28} {count:>4}{note}")

    lines.append("")
    lines.append(f"Được phép lưu toàn văn : {storable}/{len(papers)}")
    lines.append(f"Chỉ metadata + abstract: {len(papers) - storable}/{len(papers)}")
    return "\n".join(lines)
