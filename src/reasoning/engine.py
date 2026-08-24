"""Facade của lõi suy luận — gói toàn bộ các bước thành một API gọn.

Cố ý tách làm HAI giai đoạn:

    reason()  : quan sát + SHAP  → giả thuyết có rule_strength và verdict
    ── (pipeline chèn RAG vào giữa: gán citations và rag_support) ──
    rank()    : giả thuyết đã đủ ba nguồn → điểm số, xếp hạng, độ tin cậy

Tách như vậy vì RAG là tầng I/O (mạng, vector DB) mà `reasoning/` không được phép
phụ thuộc. Nhờ đó toàn bộ module này test được offline, và cấu hình ablation D
(tắt RAG) chỉ đơn giản là bỏ qua bước ở giữa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kb import KnowledgeBase, get_knowledge_base
from schemas import Attribution, DataEvidence, DerivedFeatures, Hypothesis, Observation

from .consistency import annotate_contributions, assess_consistency
from .derive import derive_features, feature_context, variable_lookup
from .hypotheses import build_hypotheses
from .rules import RuleEngine, RuleEvaluation
from .scoring import ScoringWeights, score_hypotheses


@dataclass
class ReasoningResult:
    """Toàn bộ trạng thái trung gian — giữ lại để chương đánh giá phân tích được."""

    derived: DerivedFeatures
    variables: dict[str, float | None]
    rule_evaluation: RuleEvaluation
    hypotheses: list[Hypothesis]
    conflicts: list[str] = field(default_factory=list)
    data_evidence: list[DataEvidence] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


class ReasoningEngine:
    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        weights: ScoringWeights | None = None,
    ) -> None:
        self.kb = kb or get_knowledge_base()
        self.weights = weights or ScoringWeights()
        self.rules = RuleEngine(self.kb)

    # ------------------------------------------------------------------ bước 1
    def reason(
        self,
        observation: Observation,
        attribution: Attribution | None = None,
        use_rules: bool = True,
        use_shap: bool = True,
    ) -> ReasoningResult:
        """Từ quan sát (+ SHAP) sinh giả thuyết cơ chế và phát hiện mâu thuẫn.

        `use_rules` / `use_shap` phục vụ ablation (docs/01 §6): tắt tầng nào thì
        tầng đó không đóng góp bằng chứng, chứ không phải giả vờ có bằng chứng.
        """
        inversion_km = self.kb.config.get("inversion_layer_km", 1.5)
        derived = derive_features(observation, inversion_layer_km=inversion_km)
        variables = variable_lookup(observation, derived)
        data_evidence = feature_context(observation, derived)

        rule_eval = self.rules.evaluate(variables)
        strength_of = (
            {rid: f.strength for rid, f in rule_eval.firings.items()}
            if use_rules
            else dict.fromkeys(rule_eval.firings, 0.0)
        )

        hypotheses = build_hypotheses(self.kb, strength_of, rule_eval.firings, variables)

        if use_shap and attribution is not None:
            attribution = annotate_contributions(self.kb, attribution)
            hypotheses, conflicts = assess_consistency(self.kb, hypotheses, attribution)
        else:
            hypotheses, conflicts = assess_consistency(self.kb, hypotheses, None)

        missing = self._describe_missing(rule_eval)

        return ReasoningResult(
            derived=derived,
            variables=variables,
            rule_evaluation=rule_eval,
            hypotheses=hypotheses,
            conflicts=conflicts,
            data_evidence=data_evidence,
            missing=missing,
        )

    # ------------------------------------------------------------------ bước 2
    def rank(self, hypotheses: list[Hypothesis]) -> tuple[list[Hypothesis], list[Hypothesis]]:
        """Chấm điểm và xếp hạng sau khi RAG đã (hoặc không) bổ sung tài liệu."""
        return score_hypotheses(hypotheses, self.weights)

    # ------------------------------------------------------------------ phụ trợ
    def _describe_missing(self, rule_eval: RuleEvaluation) -> list[str]:
        """Diễn đạt dữ liệu thiếu bằng tiếng Việt để narrator nêu thành hạn chế.

        INV-3: thiếu dữ liệu phải được NÓI RA, không được im lặng bỏ qua — người
        đọc cần biết cơ chế nào chưa được kiểm tra chứ không phải cơ chế nào đã bị
        loại trừ.
        """
        notes: list[str] = []
        blocked = [f for f in rule_eval.firings.values() if f.skipped_reason]
        for firing in blocked:
            mechs = self.kb.mechanisms_for_rule(firing.rule_id)
            names = ", ".join(m.name for m in mechs) or "cơ chế liên quan"
            notes.append(
                f"Không kiểm tra được '{firing.name}' ({firing.skipped_reason}) "
                f"→ chưa thể khẳng định hay loại trừ: {names}."
            )
        return notes
