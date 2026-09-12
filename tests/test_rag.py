"""Scientific RAG: chunking, embedding, store, và gated retrieval.

Chạy trên `InMemoryStore` + `HashEmbedder` để không cần Docker và không cần torch.
Điều được kiểm ở đây là LOGIC của tầng RAG (cổng chặn, cấp nhãn bằng chứng, tính
rag_support), không phải chất lượng ngữ nghĩa — chất lượng đó thuộc về chương đánh
giá với backend `hf` và bộ query có nhãn.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag.chunking import chunk_paper, split_by_words, split_sections
from rag.corpus import _passes_snowball_gate, _reconstruct_abstract, licensing_report
from rag.embedding import HashEmbedder, PrecomputedQueryEmbedder, normalize_query
from rag.models import Chunk, Paper, is_redistributable
from rag.retrieve import GatedRetriever, NullRetriever, RetrievalConfig
from rag.store import InMemoryStore
from reasoning.hypotheses import build_hypotheses
from schemas import Hypothesis, Verdict

# =============================================================================
# Bản quyền
# =============================================================================


def test_closed_access_paper_cannot_hold_full_text():
    """Ràng buộc pháp lý được thực thi bằng code, không chỉ ghi trong tài liệu."""
    with pytest.raises(ValueError, match="giấy phép không cho phép lưu"):
        Paper(
            paper_id="x",
            title="Closed paper",
            full_text="toàn văn không được phép lưu",
            is_open_access=False,
        )


def test_cc_licensed_paper_may_hold_full_text():
    paper = Paper(
        paper_id="x",
        title="OA paper",
        full_text="nội dung",
        is_open_access=True,
        license="cc-by",
        oa_status="gold",
    )
    assert paper.content() == "nội dung"
    assert paper.may_store_full_text()


def test_bronze_oa_is_readable_but_not_storable():
    """BẪY PHÁP LÝ QUAN TRỌNG NHẤT CỦA TẦNG CORPUS.

    OpenAlex đánh dấu bronze OA là `is_oa = True`, nhưng bronze nghĩa là "nhà xuất
    bản cho đọc miễn phí theo điều khoản riêng" — KHÔNG có giấy phép mở. Sao chép
    toàn văn vào corpus vẫn là vi phạm bản quyền, và nhà xuất bản có thể rút quyền
    đọc bất cứ lúc nào.

    Nếu điều kiện lưu chỉ dựa vào `is_open_access` thì đúng nhóm bài này sẽ lọt.
    """
    metadata_only = Paper(
        paper_id="x",
        title="Bronze OA paper",
        abstract="đọc được miễn phí",
        is_open_access=True,
        license=None,
        oa_status="bronze",
    )
    assert metadata_only.is_open_access is True
    assert metadata_only.may_store_full_text() is False

    with pytest.raises(ValueError, match="bronze OA"):
        Paper(
            paper_id="x",
            title="Bronze OA paper",
            full_text="toàn văn KHÔNG được phép lưu",
            is_open_access=True,
            license=None,
            oa_status="bronze",
        )


@pytest.mark.parametrize(
    ("license_code", "allowed"),
    [
        ("cc-by", True),
        ("CC-BY-SA", True),
        ("cc-by-nc-nd", True),
        ("cc0", True),
        ("public-domain", True),
        ("publisher-specific-oa", False),
        ("other-oa", False),
        (None, False),
        ("", False),
    ],
)
def test_license_gate(license_code, allowed):
    assert is_redistributable(license_code) is allowed


# =============================================================================
# OpenAlex
# =============================================================================


def test_inverted_index_abstract_is_reconstructed():
    inverted = {"Low": [0], "boundary": [1], "layer": [2], "height": [3]}
    assert _reconstruct_abstract(inverted) == "Low boundary layer height"
    assert _reconstruct_abstract(None) is None


# =============================================================================
# Snowball — mở rộng corpus theo danh mục tham khảo
# =============================================================================


def _candidate(title: str, *, year: int = 2020, citations: int = 50, abstract: str = "") -> Paper:
    return Paper(paper_id="W1", title=title, year=year, cited_by_count=citations, abstract=abstract)


def test_snowball_gate_keeps_on_topic_paper():
    paper = _candidate("Boundary layer height controls on winter PM2.5 accumulation")
    assert _passes_snowball_gate(paper, year_min=2015, min_citations=5, keywords=None)


def test_snowball_gate_drops_off_topic_paper():
    """Danh mục tham khảo kéo về rất nhiều thứ lạc đề — cổng này giữ độ chính xác.

    Ví dụ có thật khi thử: một bài haze Bắc Kinh trích cả bài về mùa đông hạt nhân
    và vi sinh vật trong không khí.
    """
    paper = _candidate("The Atmosphere After a Nuclear War: Twilight at Noon")
    assert not _passes_snowball_gate(paper, year_min=2015, min_citations=5, keywords=None)


def test_snowball_gate_uses_abstract_not_only_title():
    """Tiêu đề có thể không chứa từ khóa miền, abstract thì có."""
    paper = _candidate(
        "Synoptic controls on a severe episode in the Red River Delta",
        abstract="We analyse mixing height and ventilation during the event.",
    )
    assert _passes_snowball_gate(paper, year_min=2015, min_citations=5, keywords=None)


def test_snowball_gate_filters_by_year_and_citations():
    on_topic = "PM2.5 haze boundary layer"
    assert not _passes_snowball_gate(
        _candidate(on_topic, year=2001), year_min=2015, min_citations=5, keywords=None
    )
    assert not _passes_snowball_gate(
        _candidate(on_topic, citations=1), year_min=2015, min_citations=5, keywords=None
    )


def test_licensing_report_separates_storable_from_readable():
    papers = [
        Paper(paper_id="a", title="A", license="cc-by", oa_status="gold", is_open_access=True),
        Paper(paper_id="b", title="B", license=None, oa_status="bronze", is_open_access=True),
        Paper(paper_id="c", title="C", license=None, oa_status="closed", is_open_access=False),
    ]
    text = licensing_report(papers)

    assert "Được phép lưu toàn văn : 1/3" in text
    assert "Chỉ metadata + abstract: 2/3" in text
    assert "bronze" in text


# =============================================================================
# Embedding tính sẵn — tách việc nặng sang GPU ngoài (Kaggle/Colab)
# =============================================================================


def _write_query_vectors(tmp_path, queries, dim=8, model="BAAI/bge-m3"):
    path = tmp_path / "query_vectors.json"
    path.write_text(
        json.dumps(
            {
                "model": model,
                "dim": dim,
                "vectors": {q: [float(i) / dim] * dim for i, q in enumerate(queries, start=1)},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_precomputed_embedder_returns_stored_vector(tmp_path):
    path = _write_query_vectors(tmp_path, ["boundary layer PM2.5 accumulation"])
    embedder = PrecomputedQueryEmbedder(path)

    vector = embedder.encode(["boundary layer PM2.5 accumulation"], is_query=True)

    assert vector.shape == (1, 8)


def test_precomputed_embedder_normalizes_whitespace(tmp_path):
    """`rag_query` trong YAML viết nhiều dòng — hai phía phải chuẩn hóa giống nhau.

    Không có bước này thì khóa lúc lưu và khóa lúc tra khác nhau, và MỌI truy vấn
    đều trượt.
    """
    path = _write_query_vectors(tmp_path, ["planetary boundary layer height PM2.5"])
    embedder = PrecomputedQueryEmbedder(path)

    messy = "planetary boundary   layer\n  height     PM2.5\n"
    assert embedder.encode([messy], is_query=True).shape == (1, 8)


def test_precomputed_embedder_fails_loudly_on_unknown_query(tmp_path):
    """Truy vấn lạ phải NÉM LỖI, không được trả vector rỗng.

    Trả vector rỗng sẽ cho ra "cơ chế này không có tài liệu hỗ trợ" — tức là bịa
    ra một kết luận fail-safe từ một lỗi cấu hình, sai lệch đúng vào thứ chương
    đánh giá đang đo.
    """
    path = _write_query_vectors(tmp_path, ["truy vấn đã biết"])
    embedder = PrecomputedQueryEmbedder(path)

    with pytest.raises(KeyError, match="chưa có vector tính sẵn"):
        embedder.encode(["một truy vấn hoàn toàn khác"], is_query=True)


def test_precomputed_embedder_refuses_to_embed_documents(tmp_path):
    """Nó chỉ tra bảng truy vấn — embed tài liệu phải chạy trên GPU."""
    path = _write_query_vectors(tmp_path, ["q"])
    embedder = PrecomputedQueryEmbedder(path)

    with pytest.raises(RuntimeError, match="chỉ phục vụ TRUY VẤN"):
        embedder.encode(["nội dung một bài báo"], is_query=False)


def test_precomputed_embedder_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        PrecomputedQueryEmbedder(tmp_path / "khong-ton-tai.json")


def test_every_mechanism_query_is_covered_by_the_exported_table(kb, tmp_path):
    """HỢP ĐỒNG GIỮA HAI PHÍA.

    `export_for_kaggle.py` xuất truy vấn bằng `normalize_query(m.rag_query)`;
    `GatedRetriever` tra bằng `hypothesis.rag_query`. Test này khóa lại việc hai
    đường đó cho ra CÙNG một khóa — nếu lệch, mọi cơ chế đều mất trích dẫn mà
    không có lỗi nào hiện ra.
    """
    exported = sorted(
        {normalize_query(m.rag_query) for m in kb.mechanisms.values() if m.rag_query.strip()}
    )
    path = _write_query_vectors(tmp_path, exported)
    embedder = PrecomputedQueryEmbedder(path)

    hypotheses = build_hypotheses(kb, dict.fromkeys(kb.rules, 0.0), {}, {})
    lookups = [h.rag_query for h in hypotheses if h.rag_query]
    assert lookups, "không có cơ chế nào có rag_query"

    for query in lookups:
        embedder.encode([query], is_query=True)  # không được ném KeyError


# =============================================================================
# Chunking
# =============================================================================


def test_sections_are_detected():
    text = (
        "Abstract\nWe study haze.\nIntroduction\nBackground text here.\nResults\nPM2.5 increased.\n"
    )
    names = [name for name, _ in split_sections(text)]
    assert "abstract" in names
    assert "results" in names


def test_text_without_headings_becomes_single_body_section():
    assert split_sections("Just a paragraph.")[0][0] == "body"


def test_chunks_never_split_a_sentence():
    sentences = [f"Sentence number {i} about boundary layer height." for i in range(60)]
    chunks = split_by_words(" ".join(sentences), chunk_words=50, overlap_ratio=0.2)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip().endswith("."), "chunk bị cắt giữa câu"


def test_chunks_overlap_to_preserve_context():
    sentences = [f"Fact {i} about aerosols." for i in range(40)]
    chunks = split_by_words(" ".join(sentences), chunk_words=30, overlap_ratio=0.25)
    first_tail = chunks[0].split(".")[-2].strip()
    assert first_tail in chunks[1]


def test_chunk_carries_title_and_citation_metadata():
    paper = Paper(
        paper_id="W1",
        doi="10.1000/xyz",
        title="Boundary layer and PM2.5",
        abstract=" ".join(["Low mixing height increases surface concentrations."] * 8),
        authors=["Nguyen A", "Tran B", "Le C"],
        year=2022,
    )
    chunks = chunk_paper(paper)
    assert chunks
    assert chunks[0].text.startswith("Boundary layer and PM2.5.")
    assert chunks[0].doi == "10.1000/xyz"
    assert chunks[0].authors == "Nguyen A et al."


def test_too_short_paper_produces_no_chunks():
    assert chunk_paper(Paper(paper_id="W2", title="t", abstract="Short.")) == []


# =============================================================================
# Embedding & store
# =============================================================================


def test_hash_embedder_is_deterministic_and_normalised():
    embedder = HashEmbedder(dim=256)
    a = embedder.encode(["boundary layer height"])
    b = embedder.encode(["boundary layer height"])
    assert np.allclose(a, b)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, rel=1e-5)


def test_hash_embedder_ranks_shared_vocabulary_higher():
    embedder = HashEmbedder(dim=512)
    vecs = embedder.encode(
        [
            "planetary boundary layer height controls PM2.5 accumulation",
            "shallow planetary boundary layer traps PM2.5 near the surface",
            "ocean acidification affects coral reef calcification rates",
        ]
    )
    assert float(vecs[0] @ vecs[1]) > float(vecs[0] @ vecs[2])


def test_store_roundtrip_preserves_metadata():
    store = InMemoryStore()
    embedder = HashEmbedder(dim=128)
    chunk = Chunk(chunk_id="c1", paper_id="W1", text="low boundary layer", doi="10.1/a", year=2021)

    store.ensure_collection(dim=128)
    store.upsert([chunk], embedder.encode([chunk.text]))

    hits = store.search(embedder.encode(["low boundary layer"], is_query=True)[0], limit=3)
    assert hits[0].chunk.doi == "10.1/a"
    assert store.count() == 1


def test_store_rejects_length_mismatch():
    store = InMemoryStore()
    with pytest.raises(ValueError, match="Lệch số lượng"):
        store.upsert([Chunk(chunk_id="a", paper_id="p", text="t")], np.zeros((2, 4)))


# =============================================================================
# Gated retrieval
# =============================================================================


@pytest.fixture
def retriever():
    embedder = HashEmbedder(dim=512)
    store = InMemoryStore()
    store.ensure_collection(dim=512)

    chunks = [
        Chunk(
            chunk_id="c1",
            paper_id="W1",
            text=(
                "planetary boundary layer height shallow mixing layer PM2.5 accumulation "
                "urban haze episode nocturnal inversion"
            ),
            doi="10.1/pblh",
            title="PBLH and haze",
            authors="Nguyen A",
            year=2022,
        ),
        Chunk(
            chunk_id="c2",
            paper_id="W2",
            text="coral reef calcification ocean acidification marine biology survey",
            doi="10.1/coral",
            title="Coral reefs",
            authors="Smith B",
            year=2019,
        ),
    ]
    store.upsert(chunks, embedder.encode([c.text for c in chunks]))
    return GatedRetriever(
        store, embedder, config=RetrievalConfig(fetch_k=10, top_k=2, min_score=0.25)
    )


def _hypothesis(mech_id: str, query: str, verdict=Verdict.CONFIRMED) -> Hypothesis:
    return Hypothesis(
        mechanism_id=mech_id,
        name=mech_id,
        name_en=mech_id,
        category="accumulation",
        effect="increase",
        rule_strength=0.8,
        verdict=verdict,
        rag_query=query,
    )


def test_gate_blocks_irrelevant_corpus(retriever):
    """Kho không có tài liệu phù hợp → nói 'không có', không trích dẫn bừa (INV-3)."""
    h = _hypothesis("MECH_X", "volcanic sulfur dioxide stratospheric injection")
    retriever.attach([h])
    assert h.citations == []
    assert h.rag_support == 0.0


def test_relevant_mechanism_gets_cited(retriever):
    h = _hypothesis(
        "MECH_LOW_PBLH", "planetary boundary layer height shallow mixing PM2.5 accumulation"
    )
    retriever.attach([h])
    assert h.citations, "cơ chế có tài liệu phù hợp mà không được trích dẫn"
    assert h.citations[0].doi == "10.1/pblh"
    assert h.rag_support > 0.0


def test_evidence_ids_are_shared_across_mechanisms(retriever):
    """Cùng một tài liệu chống lưng hai cơ chế phải mang CÙNG nhãn E#."""
    query = "planetary boundary layer height shallow mixing PM2.5 accumulation"
    h1 = _hypothesis("MECH_A", query)
    h2 = _hypothesis("MECH_B", query)
    retriever.attach([h1, h2])

    assert h1.citations[0].evidence_id == h2.citations[0].evidence_id


# =============================================================================
# Một bài báo = một bằng chứng
# =============================================================================


@pytest.fixture
def multi_chunk_retriever():
    """Kho có MỘT bài bị cắt thành ba đoạn, cộng một bài khác hẳn.

    Đây là tình huống thật: bài dài cắt ra nhiều chunk, và nhiều chunk cùng khớp
    một truy vấn.
    """
    embedder = HashEmbedder(dim=512)
    store = InMemoryStore()
    store.ensure_collection(dim=512)

    shared = dict(doi="10.1/same", title="Một bài dài", authors="Nguyen A", year=2021)
    chunks = [
        Chunk(
            chunk_id="s::0",
            paper_id="W9",
            position=0,
            section="abstract",
            text="boundary layer height PM2.5 accumulation haze episode stagnation",
            **shared,
        ),
        Chunk(
            chunk_id="s::1",
            paper_id="W9",
            position=1,
            section="results",
            text="boundary layer height PM2.5 accumulation haze episode measurements",
            **shared,
        ),
        Chunk(
            chunk_id="s::2",
            paper_id="W9",
            position=2,
            section="discussion",
            text="boundary layer height PM2.5 accumulation haze episode implications",
            **shared,
        ),
        Chunk(
            chunk_id="other::0",
            paper_id="W8",
            position=0,
            text="boundary layer height PM2.5 accumulation haze episode independent study",
            doi="10.1/other",
            title="Bài độc lập",
            authors="Tran B",
            year=2022,
        ),
    ]
    store.upsert(chunks, embedder.encode([c.text for c in chunks]))
    return GatedRetriever(
        store, embedder, config=RetrievalConfig(fetch_k=10, top_k=3, min_score=0.05)
    )


def test_one_paper_yields_one_citation(multi_chunk_retriever):
    """Ba đoạn của cùng một bài chỉ được cấp MỘT nhãn.

    Trước khi gộp, câu trả lời có [E1][E2][E3] cùng trỏ về một bài — người đọc
    tưởng ba bằng chứng độc lập.
    """
    h = _hypothesis("MECH_A", "boundary layer height PM2.5 accumulation haze episode")
    multi_chunk_retriever.attach([h])

    dois = [c.doi for c in h.citations]
    assert len(dois) == len(set(dois)), f"một bài bị cấp nhiều nhãn: {dois}"
    assert dois.count("10.1/same") == 1


def test_evidence_labels_are_unique_per_paper(multi_chunk_retriever):
    h = _hypothesis("MECH_A", "boundary layer height PM2.5 accumulation haze episode")
    multi_chunk_retriever.attach([h])

    labels = [c.evidence_id for c in h.citations]
    assert len(labels) == len(set(labels))


def test_coverage_counts_papers_not_passages(multi_chunk_retriever):
    """`rag_support` đo ĐỒNG THUẬN — nhiều bài độc lập, không phải nhiều đoạn.

    Ba đoạn của một bài phải cho điểm THẤP HƠN ba bài khác nhau, vì một bài đơn
    lẻ không thay thế được đồng thuận của nhiều nhóm nghiên cứu độc lập.
    """
    query = "boundary layer height PM2.5 accumulation haze episode"
    h = _hypothesis("MECH_A", query)
    multi_chunk_retriever.attach([h])
    support_two_papers = h.rag_support

    # Cùng nội dung nhưng mỗi đoạn là một bài riêng.
    embedder = HashEmbedder(dim=512)
    store = InMemoryStore()
    store.ensure_collection(dim=512)
    distinct = [
        Chunk(
            chunk_id=f"d{i}::0",
            paper_id=f"W{i}",
            text=f"boundary layer height PM2.5 accumulation haze episode variant {i}",
            doi=f"10.1/paper{i}",
            title=f"Bài {i}",
            year=2020 + i,
        )
        for i in range(3)
    ]
    store.upsert(distinct, embedder.encode([c.text for c in distinct]))
    other = GatedRetriever(
        store, embedder, config=RetrievalConfig(fetch_k=10, top_k=3, min_score=0.05)
    )
    h2 = _hypothesis("MECH_A", query)
    other.attach([h2])

    assert h2.rag_support > support_two_papers


def test_citation_keeps_the_best_matching_passage(multi_chunk_retriever):
    """Đoạn hiển thị cho một nhãn là đoạn khớp nhất của bài đó."""
    h = _hypothesis("MECH_A", "boundary layer height PM2.5 accumulation haze episode")
    multi_chunk_retriever.attach([h])

    same = next(c for c in h.citations if c.doi == "10.1/same")
    assert same.score == pytest.approx(max(c.score for c in h.citations if c.doi == "10.1/same"))
    assert same.text in {
        "boundary layer height PM2.5 accumulation haze episode stagnation",
        "boundary layer height PM2.5 accumulation haze episode measurements",
        "boundary layer height PM2.5 accumulation haze episode implications",
    }


def test_inactive_hypotheses_are_not_queried(retriever):
    h = _hypothesis("MECH_Z", "planetary boundary layer height", verdict=Verdict.INACTIVE)
    retriever.attach([h])
    assert h.citations == []


def test_null_retriever_marks_layer_as_off():
    h = _hypothesis("MECH_A", "anything")
    h.rag_support = 0.9
    NullRetriever().attach([h])
    assert h.rag_support == 0.0
    assert h.citations == []
