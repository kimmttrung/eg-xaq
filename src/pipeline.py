"""Orchestrator — ghép toàn bộ pipeline từ câu hỏi tới câu trả lời.

    câu hỏi → dữ liệu → SHAP → rule/KG → RAG có cổng chặn → consistency
            → xếp hạng → EvidenceBundle → narrator

Mọi tầng bằng chứng bật/tắt được bằng cờ trong `PipelineConfig`, để chạy thẳng các
cấu hình ablation A–E của chương đánh giá mà không phải sửa code (docs/01 §6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date as Date

from data.base import ObservationProvider
from kb import KnowledgeBase, get_knowledge_base
from narrator.narrator import DryRunNarrator, Narrator
from reasoning.engine import ReasoningEngine
from reasoning.kg import attach_citations, build_knowledge_graph
from reasoning.scoring import overall_confidence
from schemas import Answer, Attribution, EvidenceBundle, Observation, QuestionContext
from xai.base import AttributionProvider

HANOI = (21.03, 105.85, "Hà Nội")


@dataclass
class PipelineConfig:
    """Cờ bật/tắt từng tầng bằng chứng.

    Cấu hình ablation (docs/06-evaluation.md §4):
        A. LLM-only     : tất cả False
        B. + Data       : use_observation
        C. + SHAP       : use_observation, use_shap
        D. + Rule/KG    : use_observation, use_shap, use_rules
        E. Full EG-XAQ  : tất cả True
    """

    use_observation: bool = True
    use_shap: bool = True
    use_rules: bool = True
    use_rag: bool = True

    #: Số đặc trưng SHAP đưa vào bundle. Giữ nhỏ để narrator không bị ngợp.
    shap_top_k: int = 6

    @classmethod
    def ablation(cls, level: str) -> PipelineConfig:
        presets = {
            "A": cls(False, False, False, False),
            "B": cls(True, False, False, False),
            "C": cls(True, True, False, False),
            "D": cls(True, True, True, False),
            "E": cls(True, True, True, True),
        }
        key = level.strip().upper()
        if key not in presets:
            raise ValueError(f"Cấu hình ablation không hợp lệ: {level!r}. Chọn A–E.")
        return presets[key]


class ExplanationPipeline:
    """Điều phối toàn bộ. Không tự chứa tri thức nào — chỉ nối các tầng lại."""

    def __init__(
        self,
        observation_provider: ObservationProvider,
        attribution_provider: AttributionProvider | None = None,
        retriever=None,
        narrator: Narrator | None = None,
        kb: KnowledgeBase | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.kb = kb or get_knowledge_base()
        self.observations = observation_provider
        self.attributions = attribution_provider
        self.retriever = retriever
        self.narrator = narrator or DryRunNarrator()
        self.config = config or PipelineConfig()
        self.engine = ReasoningEngine(self.kb)
        self.graph = build_knowledge_graph(self.kb)

    # ------------------------------------------------------------------ chính
    def explain(
        self,
        question: str,
        lat: float = HANOI[0],
        lon: float = HANOI[1],
        date: Date | None = None,
        step: int = 0,
        place_label: str = HANOI[2],
    ) -> Answer:
        bundle = self.build_bundle(question, lat, lon, date, step, place_label)
        return self.narrator.narrate(bundle)

    def build_bundle(
        self,
        question: str,
        lat: float = HANOI[0],
        lon: float = HANOI[1],
        date: Date | None = None,
        step: int = 0,
        place_label: str = HANOI[2],
    ) -> EvidenceBundle:
        """Dựng gói bằng chứng. Tách khỏi `explain()` để test được mà không cần LLM."""
        date = date or Date.today()
        context = QuestionContext(
            raw_question=question, lat=lat, lon=lon, date=date, step=step, place_name=place_label
        )

        # --- Tầng 1: dữ liệu quan sát ---
        if self.config.use_observation:
            observation = self.observations.get(lat, lon, date, step)
        else:
            # Ablation A: không có dữ liệu nào. Bundle rỗng là cố ý — narrator phải
            # trả lời "không đủ căn cứ", và đó chính là điều cần đo ở cấu hình này.
            observation = Observation(lat=lat, lon=lon, date=date, step=step, provenance="disabled")

        # --- Tầng 2: attribution mô hình ---
        attribution: Attribution | None = None
        if self.config.use_shap and self.attributions is not None:
            attribution = self.attributions.get(lat, lon, date, step, observation=observation)

        # --- Tầng 3: rule + KG ---
        result = self.engine.reason(
            observation,
            attribution,
            use_rules=self.config.use_rules,
            use_shap=self.config.use_shap,
        )

        # --- Tầng 4: Scientific RAG có cổng chặn ---
        if self.config.use_rag and self.retriever is not None:
            self.retriever.attach(result.hypotheses)

        # --- Tổng hợp và xếp hạng ---
        causes, suppressors = self.engine.rank(result.hypotheses)
        attach_citations(self.graph, [*causes, *suppressors])

        return EvidenceBundle(
            question=context,
            place_label=place_label,
            observation=observation,
            derived=result.derived,
            data_evidence=result.data_evidence,
            attribution=attribution,
            shap_summary=attribution.top(self.config.shap_top_k) if attribution else [],
            non_mechanistic_share=(attribution.non_mechanistic_share() if attribution else None),
            hypotheses=causes,
            suppressors=suppressors,
            conflicts=result.conflicts,
            missing=result.missing + self._layer_notes(),
            config_flags=asdict(self.config),
            overall_confidence=overall_confidence(
                causes, result.conflicts, used_rag=self.config.use_rag
            ),
        )

    # ------------------------------------------------------------------ phụ trợ
    def _layer_notes(self) -> list[str]:
        """Ghi rõ tầng nào bị tắt — để đầu ra ablation tự giải thích được chính nó."""
        notes: list[str] = []
        if not self.config.use_observation:
            notes.append("Tầng dữ liệu quan trắc bị TẮT trong lần chạy này.")
        if not self.config.use_shap:
            notes.append("Tầng attribution mô hình (SHAP) bị TẮT trong lần chạy này.")
        if not self.config.use_rules:
            notes.append("Tầng luật/Knowledge Graph bị TẮT trong lần chạy này.")
        if not self.config.use_rag:
            notes.append("Tầng tài liệu khoa học (RAG) bị TẮT — câu trả lời không có trích dẫn.")
        elif self.retriever is None:
            notes.append("Chưa cấu hình retriever — không có trích dẫn khoa học.")
        return notes
