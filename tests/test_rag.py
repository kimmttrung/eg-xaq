"""Scientific RAG: chunking, embedding, store, và gated retrieval.

Chạy trên `InMemoryStore` + `HashEmbedder` để không cần Docker và không cần torch.
Điều được kiểm ở đây là LOGIC của tầng RAG (cổng chặn, cấp nhãn bằng chứng, tính
rag_support), không phải chất lượng ngữ nghĩa — chất lượng đó thuộc về chương đánh
giá với backend `hf` và bộ query có nhãn.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.chunking import chunk_paper, split_by_words, split_sections
from rag.corpus import _reconstruct_abstract
from rag.embedding import HashEmbedder
from rag.models import Chunk, Paper
from rag.retrieve import GatedRetriever, NullRetriever, RetrievalConfig
from rag.store import InMemoryStore
from schemas import Hypothesis, Verdict

# =============================================================================
# Bản quyền
# =============================================================================


def test_closed_access_paper_cannot_hold_full_text():
    """Ràng buộc pháp lý được thực thi bằng code, không chỉ ghi trong tài liệu."""
    with pytest.raises(ValueError, match="open-access"):
        Paper(
            paper_id="x",
            title="Closed paper",
            full_text="toàn văn không được phép lưu",
            is_open_access=False,
        )


def test_open_access_paper_may_hold_full_text():
    paper = Paper(paper_id="x", title="OA paper", full_text="nội dung", is_open_access=True)
    assert paper.content() == "nội dung"


# =============================================================================
# OpenAlex
# =============================================================================


def test_inverted_index_abstract_is_reconstructed():
    inverted = {"Low": [0], "boundary": [1], "layer": [2], "height": [3]}
    assert _reconstruct_abstract(inverted) == "Low boundary layer height"
    assert _reconstruct_abstract(None) is None


# =============================================================================
# Chunking
# =============================================================================


def test_sections_are_detected():
    text = (
        "Abstract\nWe study haze.\n"
        "Introduction\nBackground text here.\n"
        "Results\nPM2.5 increased.\n"
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
    """Cùng một đoạn văn chống lưng hai cơ chế phải mang CÙNG nhãn E#."""
    query = "planetary boundary layer height shallow mixing PM2.5 accumulation"
    h1 = _hypothesis("MECH_A", query)
    h2 = _hypothesis("MECH_B", query)
    retriever.attach([h1, h2])

    assert h1.citations[0].evidence_id == h2.citations[0].evidence_id


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
