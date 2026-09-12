"""Gated retrieval — điểm khác biệt cốt lõi so với RAG thông thường.

RAG thông thường:
    câu hỏi người dùng  →  vector search  →  đưa cho LLM

Ở đây:
    câu hỏi người dùng  →  dữ liệu quan sát  →  rule kích hoạt cơ chế
                        →  `rag_query` CỦA CƠ CHẾ  →  vector search  →  cổng chặn
                        →  trích dẫn gắn vào đúng cơ chế đó

Ba hệ quả:

1. **Truy vấn bằng ngôn ngữ khoa học, không phải ngôn ngữ người dùng.** Người dùng
   hỏi "sao hôm nay bụi thế?" — truy vấn gửi đi là "planetary boundary layer height
   PM2.5 accumulation urban haze episode". Corpus tiếng Anh nên khớp tốt hơn hẳn.
2. **Trích dẫn gắn vào cơ chế, không gắn vào câu trả lời nói chung.** Nhờ vậy
   narrator biết chính xác câu nào được chứng minh bởi tài liệu nào.
3. **Cổng chặn (gate).** Không tài liệu nào vượt ngưỡng → `rag_support = 0` và cơ
   chế đó bị đánh dấu thiếu căn cứ. Đây là INV-3: thà nói "chưa có tài liệu trong
   kho hỗ trợ cơ chế này" còn hơn trích dẫn một bài không liên quan.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import Citation, Hypothesis, Verdict

from .embedding import Embedder
from .rerank import NoopReranker, Reranker
from .store import SearchHit, VectorStore


@dataclass(frozen=True)
class RetrievalConfig:
    """Tham số của tầng RAG. Tất cả đều là biến ablation tiềm năng."""

    #: Số ứng viên lấy từ vector search trước khi rerank.
    fetch_k: int = 20

    #: Số trích dẫn giữ lại cho mỗi cơ chế. Giữ ít để câu trả lời không loãng.
    top_k: int = 3

    #: CỔNG CHẶN. Dưới ngưỡng này coi như không có bằng chứng.
    #: Giá trị mặc định thận trọng; cần hiệu chỉnh trên bộ query có nhãn
    #: (docs/06-evaluation.md §3) chứ không đoán.
    min_score: float = 0.30

    #: Chỉ truy xuất cho cơ chế thực sự đang được xét — tiết kiệm truy vấn và
    #: tránh nhét vào bundle những trích dẫn cho cơ chế không liên quan.
    only_relevant_verdicts: tuple[Verdict, ...] = (
        Verdict.CONFIRMED,
        Verdict.PARTIAL,
        Verdict.CONFLICT,
        Verdict.MODEL_ONLY,
        Verdict.NO_SHAP,
    )


class GatedRetriever:
    """Truy xuất tài liệu theo cơ chế và gắn trích dẫn vào giả thuyết."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or NoopReranker()
        self.config = config or RetrievalConfig()

    # ------------------------------------------------------------------ truy xuất
    def retrieve(self, query: str) -> list[SearchHit]:
        """Vector search → rerank → lọc theo cổng chặn."""
        if not query.strip():
            return []
        vector = self.embedder.encode([query], is_query=True)[0]
        candidates = self.store.search(vector, limit=self.config.fetch_k)
        if not candidates:
            return []
        ranked = self.reranker.rerank(query, candidates, self.config.top_k)
        return [hit for hit in ranked if hit.score >= self.config.min_score]

    # ------------------------------------------------------------------ gắn vào
    def attach(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Gán `citations` và `rag_support` cho từng giả thuyết liên quan.

        MỘT BÀI BÁO = MỘT BẰNG CHỨNG
        ----------------------------
        Nhãn (E1, E2, ...) định danh một TÀI LIỆU, không phải một đoạn văn. Hai
        đoạn khác nhau của cùng một bài chỉ sinh ra một nhãn duy nhất.

        Vì sao không cấp nhãn theo đoạn: câu trả lời sẽ có [E3] và [E4] cùng trỏ
        về một bài, người đọc tưởng là hai bằng chứng độc lập. Tệ hơn, thành phần
        `độ_phủ` trong `_support` được thiết kế để đo ĐỒNG THUẬN KHOA HỌC — nhiều
        tài liệu độc lập cùng nói một điều. Đếm hai đoạn của một bài thành hai
        nguồn làm thổi phồng đúng con số đó, và `rag_support` đi thẳng vào công
        thức chấm điểm.

        Nhãn cũng được cấp phát TOÀN CỤC: một bài chống lưng cho hai cơ chế thì
        mang cùng một nhãn ở cả hai chỗ, như cách một mục trong danh mục tham
        khảo được nhiều chỗ trong bài trích tới.

        Đoạn trích hiển thị cho mỗi nhãn là đoạn KHỚP NHẤT tìm được cho bài đó,
        xét trên mọi cơ chế đã trích nó.
        """
        labels: dict[str, Citation] = {}  # khóa tài liệu -> Citation đã cấp nhãn

        for hypothesis in hypotheses:
            if hypothesis.verdict not in self.config.only_relevant_verdicts:
                continue

            hits = self.retrieve(hypothesis.rag_query)
            if not hits:
                hypothesis.rag_support = 0.0
                hypothesis.citations = []
                continue

            best_per_paper = self._best_hit_per_paper(hits)
            hypothesis.citations = [self._citation_for(hit, labels) for hit in best_per_paper]
            hypothesis.rag_support = self._support(best_per_paper)

        return hypotheses

    @staticmethod
    def _paper_key(hit: SearchHit) -> str:
        """Định danh TÀI LIỆU của một đoạn. DOI là chuẩn nhất; thiếu thì lùi về id."""
        chunk = hit.chunk
        return (chunk.doi or chunk.paper_id or chunk.title or "").strip().lower()

    def _best_hit_per_paper(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Mỗi tài liệu giữ lại đúng một đoạn — đoạn khớp nhất.

        `hits` đã xếp hạng giảm dần nên đoạn xuất hiện trước là đoạn tốt nhất của
        bài đó; giữ thứ tự để trích dẫn vẫn theo độ liên quan.
        """
        best: dict[str, SearchHit] = {}
        for hit in hits:
            best.setdefault(self._paper_key(hit), hit)
        return list(best.values())

    def _citation_for(self, hit: SearchHit, labels: dict[str, Citation]) -> Citation:
        """Cấp nhãn mới cho tài liệu, hoặc dùng lại nhãn đã cấp.

        Nếu cơ chế sau tìm được đoạn khớp hơn của cùng bài, cập nhật đoạn trích tại
        chỗ: nhãn giữ nguyên, nhưng đoạn văn hiển thị luôn là đoạn mạnh nhất.
        """
        key = self._paper_key(hit)
        existing = labels.get(key)
        if existing is not None:
            if hit.score > existing.score:
                existing.text = hit.chunk.text
                existing.section = hit.chunk.section
                existing.score = hit.score
            return existing

        citation = Citation(
            evidence_id=f"E{len(labels) + 1}",
            doi=hit.chunk.doi,
            title=hit.chunk.title,
            authors=hit.chunk.authors,
            year=hit.chunk.year,
            venue=hit.chunk.venue,
            section=hit.chunk.section,
            text=hit.chunk.text,
            score=hit.score,
            url=hit.chunk.url,
            is_open_access=hit.chunk.is_open_access,
        )
        labels[key] = citation
        return citation

    def _support(self, hits: list[SearchHit]) -> float:
        """Quy đổi kết quả truy xuất thành `rag_support` ∈ [0, 1].

            support = 0.65 · độ_khớp_tốt_nhất  +  0.35 · độ_phủ

        Hai thành phần vì hai thứ khác nhau đều đáng kể: một tài liệu khớp rất sát
        là bằng chứng mạnh, nhưng nhiều tài liệu độc lập cùng nói một điều thì đó
        là đồng thuận khoa học — điều mà một bài đơn lẻ không thay thế được.

        ⚠ `hits` truyền vào ĐÃ phải gộp theo bài (`_best_hit_per_paper`), nếu không
        `độ_phủ` sẽ đếm nhiều đoạn của một bài thành nhiều nguồn độc lập và thổi
        phồng điểm.
        """
        if not hits:
            return 0.0
        gate = self.config.min_score
        span = max(1e-6, 1.0 - gate)
        best = (max(h.score for h in hits) - gate) / span
        coverage = len(hits) / self.config.top_k
        return round(max(0.0, min(1.0, 0.65 * best + 0.35 * coverage)), 4)


class NullRetriever:
    """Không truy xuất gì — dùng cho cấu hình ablation A–D (tắt tầng RAG).

    Đặt `rag_support = 0` một cách tường minh thay vì bỏ trống, để bundle ghi rõ
    là tầng này đã bị TẮT chứ không phải đã tìm mà không thấy.
    """

    def attach(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        for hypothesis in hypotheses:
            hypothesis.rag_support = 0.0
            hypothesis.citations = []
        return hypotheses
