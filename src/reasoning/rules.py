"""Rule engine — đánh giá ngưỡng trong `knowledge/rules.yaml` trên dữ liệu quan sát.

Thiết kế cố ý đơn giản: không dùng thư viện rule engine (experta, durable_rules).
Lý do: rule của bài toán này đều là ngưỡng đơn biến, không có chaining phức tạp;
việc kết hợp nhiều rule thành cơ chế do `hypotheses.py` lo. Thêm một DSL nữa chỉ
làm khó giải thích trước hội đồng.
"""

from __future__ import annotations

from dataclasses import dataclass

from kb import KnowledgeBase, RuleSpec
from schemas import RuleFiring


@dataclass(frozen=True)
class RuleEvaluation:
    """Kết quả đánh giá toàn bộ rule set trên một quan sát."""

    firings: dict[str, RuleFiring]
    missing_variables: list[str]

    def strength(self, rule_id: str) -> float:
        firing = self.firings.get(rule_id)
        return firing.strength if firing else 0.0

    def fired(self) -> list[RuleFiring]:
        return [f for f in self.firings.values() if f.fired]


class RuleEngine:
    """Đánh giá rule. Không giữ trạng thái giữa các lần gọi."""

    #: strength phải vượt mức này mới coi là "kích hoạt".
    #: Đặt > 0 để tránh nhiễu: giá trị chỉ vừa chạm ngưỡng không nên tạo ra
    #: một tuyên bố nhân quả trong câu trả lời.
    FIRE_THRESHOLD = 0.05

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    def evaluate(self, variables: dict[str, float | None]) -> RuleEvaluation:
        """Chạy toàn bộ rule trên bảng tra biến từ `derive.variable_lookup`."""
        firings: dict[str, RuleFiring] = {}
        missing: set[str] = set()

        for rule in self.kb.rules.values():
            skipped = self._missing_requirement(rule, variables)
            if skipped:
                missing.update(skipped)
                firings[rule.id] = self._skipped_firing(rule, skipped)
                continue

            value = variables.get(rule.variable)
            if value is None:
                missing.add(rule.variable)
                firings[rule.id] = self._skipped_firing(rule, [rule.variable])
                continue

            strength = rule.strength(value)
            firings[rule.id] = RuleFiring(
                rule_id=rule.id,
                name=rule.name,
                variable=rule.variable,
                observed_value=value,
                threshold=rule.threshold,
                direction=rule.direction,
                unit=rule.unit,
                strength=strength,
                fired=strength > self.FIRE_THRESHOLD,
            )

        return RuleEvaluation(firings=firings, missing_variables=sorted(missing))

    @staticmethod
    def _missing_requirement(rule: RuleSpec, variables: dict[str, float | None]) -> list[str]:
        """Biến bắt buộc (`requires` trong YAML) mà quan sát không có."""
        return [name for name in rule.requires if variables.get(name) is None]

    @staticmethod
    def _skipped_firing(rule: RuleSpec, missing: list[str]) -> RuleFiring:
        """Rule không đánh giá được → strength 0 và ghi rõ lý do.

        INV-3 fail-safe: không bao giờ giả định giá trị mặc định cho dữ liệu thiếu.
        Lý do bỏ qua sẽ được đưa lên `bundle.missing` để narrator nêu hạn chế.
        """
        return RuleFiring(
            rule_id=rule.id,
            name=rule.name,
            variable=rule.variable,
            observed_value=None,
            threshold=rule.threshold,
            direction=rule.direction,
            unit=rule.unit,
            strength=0.0,
            fired=False,
            skipped_reason=f"thiếu dữ liệu: {', '.join(missing)}",
        )
