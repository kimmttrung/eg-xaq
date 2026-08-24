"""Xếp hạng giả thuyết và sinh độ tin cậy từ đồng thuận ba nguồn.

    score(h) = (α·rule_strength + β·shap_support + γ·rag_support) · prior_weight(h)

Ba trọng số α, β, γ là tham số của mô hình, KHÔNG phải hằng số vật lý. Chúng phải
được chứng minh bằng ablation trong chương đánh giá (docs/06-evaluation.md §4),
nên chúng nằm trong một dataclass cấu hình được, không hardcode rải rác.

Vì sao α > β: rule mã hóa tri thức vật lý đã được kiểm chứng, còn SHAP chỉ phản
ánh mô hình — vốn có thể học tương quan giả (INV-2). Cho SHAP trọng số cao hơn
rule là mâu thuẫn với chính lập luận trung tâm của đề tài.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import Confidence, Hypothesis, Verdict

from .consistency import RULE_ACTIVE, SHAP_SUPPORTS


@dataclass(frozen=True)
class ScoringWeights:
    """Trọng số tổng hợp. Mặc định là điểm khởi đầu, cần hiệu chỉnh bằng ablation."""

    rule: float = 0.45
    shap: float = 0.35
    rag: float = 0.20

    #: Giả thuyết CONFLICT bị phạt điểm nhưng KHÔNG bị loại — nó vẫn phải xuất
    #: hiện trong câu trả lời như một điểm bất định được nêu rõ.
    conflict_penalty: float = 0.5

    #: Giả thuyết MODEL_ONLY bị phạt nặng hơn: không có căn cứ vật lý nào.
    model_only_penalty: float = 0.35

    #: Dưới ngưỡng này thì không đưa vào câu trả lời (tránh nhiễu).
    min_score: float = 0.08

    def __post_init__(self) -> None:
        total = self.rule + self.shap + self.rag
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"α+β+γ phải bằng 1.0, hiện là {total}")


def _prior_weight(confidence_prior: float) -> float:
    """Prior chỉ ĐIỀU CHỈNH điểm trong [0.6, 1.0], không tự quyết định thứ hạng.

    Nếu để prior nhân trực tiếp, một cơ chế có prior 0.9 nhưng không hề được kích
    hoạt vẫn có thể vượt một cơ chế prior 0.6 đang kích hoạt mạnh — sai về logic.
    """
    return 0.6 + 0.4 * max(0.0, min(1.0, confidence_prior))


def _confidence_label(hypothesis: Hypothesis) -> Confidence:
    """Nhãn tin cậy = số nguồn bằng chứng đồng thuận, có xét verdict.

    Đây là giá trị mà narrator được phép nói ra. Narrator KHÔNG tự chấm điểm
    tin cậy — nếu để LLM tự đánh giá, nó sẽ đánh giá theo độ trôi chảy của văn
    bản chứ không theo bằng chứng (docs/07 §7.4).
    """
    if hypothesis.verdict is Verdict.CONFLICT:
        return Confidence.LOW
    if hypothesis.verdict is Verdict.MODEL_ONLY:
        return Confidence.LOW

    sources = 0
    if hypothesis.rule_strength > RULE_ACTIVE:
        sources += 1
    if hypothesis.shap_agreement >= SHAP_SUPPORTS:
        sources += 1
    if hypothesis.rag_support > 0.0:
        sources += 1

    if sources >= 3:
        return Confidence.HIGH
    if sources == 2:
        return Confidence.MEDIUM
    if sources == 1:
        return Confidence.LOW
    return Confidence.INSUFFICIENT


def score_hypotheses(
    hypotheses: list[Hypothesis],
    weights: ScoringWeights | None = None,
) -> tuple[list[Hypothesis], list[Hypothesis]]:
    """Chấm điểm, lọc và xếp hạng.

    Trả về (nguyên nhân làm TĂNG PM2.5, cơ chế làm GIẢM / điều kiện loại trừ),
    cả hai đã sắp xếp giảm dần theo điểm.

    Tách hai nhóm vì chúng trả lời hai câu khác nhau: nhóm đầu giải thích "vì sao
    bẩn", nhóm sau giải thích "vì sao không được làm sạch" hoặc "vì sao hôm nay sạch".
    """
    w = weights or ScoringWeights()
    kept: list[Hypothesis] = []

    for hypothesis in hypotheses:
        if hypothesis.verdict is Verdict.INACTIVE:
            continue

        # Chỉ phần đồng thuận DƯƠNG mới cộng điểm; phần mâu thuẫn xử lý bằng
        # hệ số phạt, để không có chuyện hai sai số triệt tiêu nhau.
        shap_support = max(0.0, hypothesis.shap_agreement)

        base = (
            w.rule * hypothesis.rule_strength
            + w.shap * shap_support
            + w.rag * hypothesis.rag_support
        )
        score = base * _prior_weight(hypothesis.confidence_prior)

        if hypothesis.verdict is Verdict.CONFLICT:
            score *= w.conflict_penalty
        elif hypothesis.verdict is Verdict.MODEL_ONLY:
            score *= w.model_only_penalty

        hypothesis.score = round(score, 4)
        hypothesis.confidence = _confidence_label(hypothesis)

        if hypothesis.score >= w.min_score:
            kept.append(hypothesis)

    kept.sort(key=lambda h: h.score, reverse=True)
    causes = [h for h in kept if h.effect == "increase"]
    suppressors = [h for h in kept if h.effect == "decrease"]
    return causes, suppressors


def overall_confidence(
    causes: list[Hypothesis],
    conflicts: list[str],
    used_rag: bool,
) -> Confidence:
    """Độ tin cậy của cả câu trả lời.

    Quy tắc thận trọng: lấy độ tin cậy của giả thuyết mạnh nhất rồi HẠ một bậc
    nếu có mâu thuẫn, hoặc nếu tầng RAG bị tắt (cấu hình ablation A–D).
    """
    if not causes:
        return Confidence.INSUFFICIENT

    order = [Confidence.INSUFFICIENT, Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]
    level = order.index(causes[0].confidence)

    if conflicts:
        level -= 1
    if not used_rag:
        level -= 1

    return order[max(0, level)]
