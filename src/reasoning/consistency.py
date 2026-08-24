"""Đối chiếu Rule/KG ↔ SHAP — ĐÓNG GÓP NGHIÊN CỨU SỐ 2.

Bài toán: SHAP faithful với MÔ HÌNH, không faithful với THỰC TẠI (INV-2).
Rule/KG mã hóa tri thức vật lý, không biết mô hình nghĩ gì.
Chỗ hai nguồn này gặp nhau cho ta bốn tình huống:

    Rule kích hoạt?  SHAP nói gì?              → verdict
    ─────────────────────────────────────────────────────────────
    ✓                nhấn mạnh, ĐÚNG chiều     → CONFIRMED   (confidence cao)
    ✓                không nhắc tới            → PARTIAL     (confidence trung bình)
    ✓                nhấn mạnh, NGƯỢC chiều    → CONFLICT ⚠  (phát hiện đáng giá)
    ✗                nhấn mạnh                 → MODEL_ONLY  (nghi tương quan giả)
    ✗                không nhắc tới            → INACTIVE    (bỏ khỏi câu trả lời)

CONFLICT và MODEL_ONLY là thứ hiếm được báo cáo trong tài liệu XAI cho chất lượng
không khí. Chúng KHÔNG được nuốt lặng — phải nổi lên `bundle.conflicts` để narrator
nêu ra và để chương đánh giá thống kê.
"""

from __future__ import annotations

from kb import KnowledgeBase
from schemas import Attribution, FeatureContribution, Hypothesis, Verdict

# --- Ngưỡng phân loại verdict. Đặt ở đây, không rải rác trong code. -----------

#: rule_strength phải vượt mức này thì cơ chế mới coi là "được kích hoạt".
RULE_ACTIVE = 0.05

#: shap_agreement ≥ ngưỡng này → SHAP thực sự ủng hộ cơ chế.
#: 0.15 nghĩa là các biến của cơ chế chiếm ≥15% tổng |SHAP| và đúng chiều.
SHAP_SUPPORTS = 0.15

#: shap_agreement ≤ -ngưỡng này → SHAP mâu thuẫn rõ ràng với cơ chế.
#: Đặt nhạy hơn (0.10) vì mâu thuẫn là tín hiệu cần biết sớm, thà báo thừa còn hơn bỏ sót.
SHAP_CONFLICTS = -0.10

#: Ngưỡng để coi là MODEL_ONLY — SHAP nhấn mạnh trong khi rule im lặng.
#: Cao hơn SHAP_SUPPORTS để tránh báo động vì nhiễu.
SHAP_DOMINATES = 0.25


def annotate_contributions(kb: KnowledgeBase, attribution: Attribution) -> Attribution:
    """Gắn `canonical_variable` và cờ `is_non_mechanistic` cho từng đặc trưng.

    ⚠ Nếu `knowledge/feature_map.yaml` chưa cập nhật theo danh sách đặc trưng thật
    của lab, hàm này trả về `canonical_variable=None` cho mọi thứ và toàn bộ
    consistency check trở nên vô nghĩa (docs/02-data-contract.md §7.3).
    """
    fmap = kb.feature_map
    for contribution in attribution.contributions:
        contribution.canonical_variable = fmap.resolve(contribution.feature)
        contribution.is_non_mechanistic = fmap.is_non_mechanistic(contribution.feature)
    return attribution


def _agreement_for_mechanism(
    kb: KnowledgeBase,
    mechanism_id: str,
    attribution: Attribution,
) -> tuple[float, list[FeatureContribution]]:
    """Tính shap_agreement ∈ [-1, 1] cho một cơ chế.

    Công thức:
        agreement = Σ_v  share(v) · sign_match(v)

        share(v)      = |shap_v| / Σ|shap|      (tỉ trọng đóng góp)
        sign_match(v) = +1 nếu dấu SHAP đúng như cơ chế kỳ vọng, -1 nếu ngược,
                        +1 nếu cơ chế không kỳ vọng dấu cụ thể ('any')

    Vì Σ share ≤ 1, giá trị trả về tự nhiên nằm trong [-1, 1] và mang hai thông
    tin cùng lúc: ĐỘ PHỦ (biến của cơ chế chiếm bao nhiêu phần attribution) và
    HƯỚNG (đúng hay ngược chiều vật lý).
    """
    mech = kb.mechanisms[mechanism_id]
    if not mech.variables or not attribution.contributions:
        return 0.0, []

    total = attribution.total_abs()
    agreement = 0.0
    matched: list[FeatureContribution] = []

    for canonical, expectation in mech.variables.items():
        expected_sign = expectation.sign()
        for contribution in attribution.for_variable(canonical):
            share = abs(contribution.shap) / total
            if share == 0.0:
                continue
            if expected_sign == 0:
                sign_match = 1.0
            else:
                observed_sign = 1 if contribution.shap > 0 else -1
                sign_match = 1.0 if observed_sign == expected_sign else -1.0
            agreement += share * sign_match
            matched.append(contribution)

    return max(-1.0, min(1.0, agreement)), matched


def _classify(rule_strength: float, agreement: float, has_shap: bool) -> Verdict:
    """Bảng chân trị ở docstring đầu module, viết thành code."""
    active = rule_strength > RULE_ACTIVE

    if not has_shap:
        return Verdict.NO_SHAP if active else Verdict.INACTIVE

    if active:
        if agreement <= SHAP_CONFLICTS:
            return Verdict.CONFLICT
        if agreement >= SHAP_SUPPORTS:
            return Verdict.CONFIRMED
        return Verdict.PARTIAL

    return Verdict.MODEL_ONLY if agreement >= SHAP_DOMINATES else Verdict.INACTIVE


def assess_consistency(
    kb: KnowledgeBase,
    hypotheses: list[Hypothesis],
    attribution: Attribution | None,
) -> tuple[list[Hypothesis], list[str]]:
    """Gán `shap_agreement` và `verdict` cho từng giả thuyết.

    Trả về (giả thuyết đã đánh giá, danh sách cảnh báo dạng chữ).
    Cảnh báo được viết sẵn bằng tiếng Việt để narrator dùng nguyên văn — narrator
    không được tự diễn giải mâu thuẫn theo ý nó.
    """
    warnings: list[str] = []
    has_shap = attribution is not None and bool(attribution.contributions)

    for hypothesis in hypotheses:
        if has_shap:
            agreement, matched = _agreement_for_mechanism(kb, hypothesis.mechanism_id, attribution)
        else:
            agreement, matched = 0.0, []

        hypothesis.shap_agreement = agreement
        hypothesis.shap_features = matched
        hypothesis.verdict = _classify(hypothesis.rule_strength, agreement, has_shap)

        if hypothesis.verdict is Verdict.CONFLICT:
            warnings.append(
                f"MÂU THUẪN: điều kiện quan sát kích hoạt cơ chế '{hypothesis.name}' "
                f"(mức {hypothesis.rule_strength:.2f}), nhưng attribution của mô hình lại "
                f"đẩy theo chiều NGƯỢC LẠI (agreement {agreement:+.2f}). "
                f"Có thể mô hình đang dựa vào tương quan giả cho các biến này."
            )
        elif hypothesis.verdict is Verdict.MODEL_ONLY:
            features = ", ".join(sorted({c.feature for c in matched})) or "không rõ"
            warnings.append(
                f"CHỈ MÔ HÌNH: mô hình dựa nhiều vào các đặc trưng của cơ chế "
                f"'{hypothesis.name}' ({features}, agreement {agreement:+.2f}) trong khi "
                f"điều kiện khí tượng quan sát KHÔNG kích hoạt cơ chế này. "
                f"Không đủ căn cứ để coi đây là nguyên nhân thực tế."
            )

    if has_shap:
        share = attribution.non_mechanistic_share()
        if share > 0.30:
            warnings.append(
                f"CẢNH BÁO CƠ CHẾ: {share:.0%} attribution của mô hình đến từ đặc trưng "
                f"KHÔNG mang cơ chế vật lý (toạ độ, mã thời gian, biến tĩnh theo pixel). "
                f"Dự báo có thể đúng nhờ ghi nhớ mẫu không gian/thời gian thay vì học cơ chế."
            )

    return hypotheses, warnings
