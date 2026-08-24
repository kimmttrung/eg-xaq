"""Từ rule đã kích hoạt → giả thuyết cơ chế nhân quả.

Đây là bước ánh xạ "điều kiện quan sát → cơ chế vật lý" trong Knowledge Graph.
Sinh giả thuyết cho MỌI cơ chế (kể cả cơ chế không kích hoạt) vì `consistency.py`
còn cần phát hiện trường hợp MODEL_ONLY: SHAP nhấn mạnh một biến mà rule không
kích hoạt — một trong hai phát hiện đáng giá của đề tài (docs/00 §3, ô (c)).
"""

from __future__ import annotations

import re
from string import Formatter

from kb import KnowledgeBase, MechanismSpec
from schemas import Hypothesis

#: Co-trigger CỦNG CỐ giả thuyết nhưng không tự tạo ra nó từ con số 0.
#: rule_strength = min(1, primary × (1 + BOOST × co))
#: → primary = 0 thì kết quả vẫn 0, dù mọi co-trigger đều bằng 1.
#: Đây là ràng buộc có chủ đích: một cơ chế phải được rule CHÍNH của nó kích hoạt.
#: Dạng NHÂN (không phải cộng) là thứ đảm bảo điều đó — dạng cộng sẽ để co-trigger
#: tự sinh ra cơ chế mà điều kiện chính của nó chưa hề thỏa mãn.
CO_TRIGGER_BOOST = 0.25


def combine_rule_strengths(
    mechanism: MechanismSpec,
    strength_of: dict[str, float],
) -> tuple[float, list[str]]:
    """Gộp strength của các rule thành rule_strength của cơ chế.

    - mode `all`: lấy MIN — cơ chế chỉ mạnh bằng điều kiện yếu nhất của nó.
    - mode `any`: lấy MAX — bất kỳ điều kiện nào đủ mạnh cũng kích hoạt.

    Trả về (rule_strength, danh sách rule_id đã đóng góp).
    """
    primary_ids = mechanism.triggers.rules
    if not primary_ids:
        return 0.0, []

    primary_values = [strength_of.get(rid, 0.0) for rid in primary_ids]
    primary = min(primary_values) if mechanism.triggers.mode == "all" else max(primary_values)

    contributing = [
        rid for rid, val in zip(primary_ids, primary_values, strict=True) if val > 0.0
    ]

    co_values = [strength_of.get(rid, 0.0) for rid in mechanism.co_triggers]
    co = sum(co_values) / len(co_values) if co_values else 0.0
    contributing += [
        rid for rid, val in zip(mechanism.co_triggers, co_values, strict=True) if val > 0.0
    ]

    strength = primary * (1.0 + CO_TRIGGER_BOOST * co)
    return max(0.0, min(1.0, strength)), contributing


def build_hypotheses(
    kb: KnowledgeBase,
    strength_of: dict[str, float],
    firings: dict,
    variables: dict[str, float | None],
) -> list[Hypothesis]:
    """Sinh một `Hypothesis` cho mỗi cơ chế trong knowledge base."""
    hypotheses: list[Hypothesis] = []

    for mech in kb.mechanisms.values():
        strength, contributing = combine_rule_strengths(mech, strength_of)
        hypotheses.append(
            Hypothesis(
                mechanism_id=mech.id,
                name=mech.name,
                name_en=mech.name_en,
                category=mech.category,
                effect=mech.effect,
                rule_strength=strength,
                fired_rules=[firings[rid] for rid in contributing if rid in firings],
                confidence_prior=mech.confidence_prior,
                narrative=render_template(mech.explanation_template, variables),
                limitations=mech.limitations,
                rag_query=" ".join(mech.rag_query.split()),
            )
        )

    return hypotheses


_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::[^}]*)?\}")


def render_template(template: str, variables: dict[str, float | None]) -> str:
    """Điền giá trị quan sát vào `explanation_template` của cơ chế.

    Nếu một placeholder không có dữ liệu, câu chứa nó KHÔNG được điền giá trị bịa —
    placeholder được thay bằng "(không có dữ liệu)". Narrator sẽ thấy và xử lý.
    """
    if not template:
        return ""

    text = " ".join(template.split())
    available = {
        name: value
        for name, value in variables.items()
        if value is not None
    }

    missing = {
        name for name in _PLACEHOLDER.findall(text) if name not in available
    }
    for name in missing:
        text = re.sub(rf"\{{{name}(?::[^}}]*)?\}}", "(không có dữ liệu)", text)

    try:
        return Formatter().vformat(text, (), _SafeDict(available))
    except (ValueError, TypeError):
        return text


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - phòng thủ
        return "(không có dữ liệu)"
