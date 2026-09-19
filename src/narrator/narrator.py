"""Constrained narrator — tầng diễn đạt cuối cùng.

Hai cài đặt:

- ``DryRunNarrator``   — render bằng template Python. Xác định hoàn toàn, không cần
                         mạng, không tốn tiền. Dùng cho test, cho demo nhanh, và làm
                         ĐỐI CHỨNG trong chương đánh giá: nó cho thấy chính xác
                         những gì bundle chứa, nên mọi khác biệt so với đầu ra LLM
                         đều là do LLM thêm vào — đó chính là thứ cần đo.
- ``AnthropicNarrator`` — gọi Claude API với prompt ràng buộc.

Vì sao DryRun đáng giá hơn vẻ ngoài của nó: khi đo hallucination, câu hỏi luôn là
"câu này có trong bằng chứng không?". DryRun là hiện thân của "đúng những gì có
trong bằng chứng, không hơn". So sánh hai đầu ra cho ta một baseline sạch.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from config import get_settings
from schemas import Answer, Confidence, EvidenceBundle, Hypothesis, Verdict

from .prompts import SYSTEM_PROMPT, build_user_message


@runtime_checkable
class Narrator(Protocol):
    name: str

    def narrate(self, bundle: EvidenceBundle) -> Answer: ...


# =============================================================================
# DryRun — template thuần Python
# =============================================================================


def is_clean_day(bundle: EvidenceBundle) -> bool:
    """Có nên kể câu chuyện "vì sao sạch" thay vì "vì sao bẩn"?

    Hai điều kiện, thỏa một là đủ: AQI ở mức tốt/trung bình, hoặc cơ chế loại bỏ mạnh
    hơn nguyên nhân mạnh nhất. Dùng chung cho narrator và bộ chấm điểm
    (`evaluation.predicted_outcome`) — một định nghĩa duy nhất, để metric đo đúng thứ
    người dùng đọc được.
    """
    if not bundle.suppressors:
        return False
    aqi = bundle.derived.aqi_vn
    if aqi is not None and aqi <= 100:
        return True
    top_cause = bundle.hypotheses[0].score if bundle.hypotheses else 0.0
    return bundle.suppressors[0].score > top_cause


def story_of(bundle: EvidenceBundle) -> Literal["cause", "clean", "insufficient"]:
    """Câu chuyện câu trả lời kể: vì sao bẩn, vì sao sạch, hay không đủ căn cứ."""
    if not bundle.hypotheses and not bundle.suppressors:
        return "insufficient"
    return "clean" if is_clean_day(bundle) else "cause"


class DryRunNarrator:
    name = "dryrun"

    def narrate(self, bundle: EvidenceBundle) -> Answer:
        sections: list[str] = []

        sections.append(self._conclusion(bundle))
        sections.append(self._data_section(bundle))

        model_section = self._model_section(bundle)
        if model_section:
            sections.append(model_section)

        sections.append(self._mechanism_section(bundle))

        if bundle.conflicts:
            sections.append(self._conflict_section(bundle))

        sections.append(self._confidence_section(bundle))

        return Answer(text="\n\n".join(s for s in sections if s), bundle=bundle, backend=self.name)

    # ------------------------------------------------------------------ mục
    def _conclusion(self, bundle: EvidenceBundle) -> str:
        header = "**Kết luận**"
        if not bundle.hypotheses and not bundle.suppressors:
            return (
                f"{header}\nKhông đủ căn cứ để khẳng định nguyên nhân. "
                f"Điều kiện quan sát không kích hoạt cơ chế nào trong cơ sở tri thức, "
                f"hoặc dữ liệu cần thiết bị thiếu."
            )

        obs = bundle.observation
        pm25 = obs.pm25_pred_ugm3 if obs.pm25_pred_ugm3 is not None else obs.pm25_obs_ugm3
        kind = f"dự báo t+{obs.step}" if obs.pm25_pred_ugm3 is not None else "quan trắc"
        aqi = bundle.derived.aqi_vn
        level = f"PM2.5 ≈ {pm25:.0f} µg/m³" if pm25 is not None else "Mức PM2.5 không rõ"
        if aqi is not None:
            level += f" (AQI {aqi} — {bundle.derived.aqi_category})"

        lines = [
            header,
            f"{level} tại {bundle.place_label} ngày {obs.date} ({kind}).",
        ]

        # Ngày không khí tốt thì câu hỏi thực sự là "vì sao SẠCH", nên phải mở đầu
        # bằng cơ chế LOẠI BỎ. Mở đầu bằng "nguyên nhân chủ đạo" trong khi AQI 35
        # là sai về mặt diễn đạt, dù mọi con số bên dưới đều đúng.
        if is_clean_day(bundle):
            lead = bundle.suppressors[0]
            lines.append(
                f"Không khí ở mức tốt. Cơ chế chủ đạo: **{lead.name}** "
                f"{self._refs(lead)}. {lead.narrative}"
            )
            if bundle.hypotheses:
                names = ", ".join(h.name.lower() for h in bundle.hypotheses[:2])
                lines.append(
                    f"Vẫn có điều kiện theo hướng làm tăng nồng độ ({names}) nhưng "
                    f"không đủ mạnh để lấn át cơ chế trên."
                )
            return "\n".join(lines)

        top = bundle.hypotheses[0]
        lines.append(f"Nguyên nhân chủ đạo: **{top.name}** {self._refs(top)}. {top.narrative}")
        if len(bundle.hypotheses) > 1:
            others = ", ".join(h.name.lower() for h in bundle.hypotheses[1:3])
            lines.append(f"Cơ chế góp phần: {others}.")
        if bundle.suppressors:
            sup = bundle.suppressors[0]
            lines.append(f"Cơ chế đang làm giảm nồng độ: {sup.name.lower()} — {sup.narrative}")
        return "\n".join(lines)

    def _data_section(self, bundle: EvidenceBundle) -> str:
        if not bundle.data_evidence:
            return ""
        lines = ["**Bằng chứng dữ liệu**"]
        for e in bundle.data_evidence:
            suffix = f" — {e.context}" if e.context else ""
            lines.append(f"- [{e.evidence_id}] {e.label}: {e.value:g} {e.unit}{suffix}")
        return "\n".join(lines)

    def _model_section(self, bundle: EvidenceBundle) -> str:
        if not bundle.shap_summary:
            return ""
        lines = [
            "**Bằng chứng từ mô hình**",
            "_Các giá trị dưới đây giải thích DỰ BÁO CỦA MÔ HÌNH, không phải nhân quả thực tế._",
        ]
        for i, c in enumerate(bundle.shap_summary, start=1):
            arrow = "↑" if c.shap > 0 else "↓"
            tag = "" if not c.is_non_mechanistic else "  ⚠ không mang cơ chế vật lý"
            lines.append(f"- [S{i}] `{c.feature}` {arrow} {c.shap:+.2f} µg/m³{tag}")
        if bundle.non_mechanistic_share is not None:
            lines.append(
                f"- Tỉ lệ attribution từ đặc trưng không mang cơ chế: "
                f"{bundle.non_mechanistic_share:.0%}"
            )
        return "\n".join(lines)

    def _mechanism_section(self, bundle: EvidenceBundle) -> str:
        lines = ["**Cơ chế khoa học**"]
        every = [*bundle.hypotheses, *bundle.suppressors]
        if not every:
            return ""
        for h in every:
            lines.append(
                f"- **{h.name}** (điểm {h.score:.2f}, {h.verdict.value}, "
                f"tin cậy {h.confidence.value})"
            )
            lines.append(f"  {h.narrative}")
            lines.append(f"  {self._rule_trace(h)}")
            if h.citations:
                for c in h.citations:
                    lines.append(
                        f"  - [{c.evidence_id}] {c.short()} — {c.title} "
                        f"{'doi:' + c.doi if c.doi else ''}"
                    )
            else:
                lines.append("  - _Chưa có tài liệu trong kho hỗ trợ cơ chế này._")
            if h.limitations:
                lines.append(f"  - _Hạn chế:_ {' '.join(h.limitations.split())}")
        return "\n".join(lines)

    def _conflict_section(self, bundle: EvidenceBundle) -> str:
        lines = ["**Điểm bất định**"]
        lines += [f"- {c}" for c in bundle.conflicts]
        return "\n".join(lines)

    def _confidence_section(self, bundle: EvidenceBundle) -> str:
        lines = [
            "**Độ tin cậy & hạn chế**",
            f"- Độ tin cậy tổng thể: **{bundle.overall_confidence.value}**",
        ]
        off = [name for name, on in bundle.config_flags.items() if on is False]
        if off:
            lines.append(f"- Tầng bằng chứng bị tắt trong lần chạy này: {', '.join(off)}")
        for note in bundle.missing:
            lines.append(f"- {note}")
        if bundle.overall_confidence is Confidence.INSUFFICIENT:
            lines.append(
                "- Hệ thống KHÔNG khẳng định nguyên nhân vì không đủ bằng chứng đồng thuận."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ phụ trợ
    @staticmethod
    def _refs(h: Hypothesis) -> str:
        refs = [f.rule_id for f in h.fired_rules] + h.evidence_ids()
        return f"[{', '.join(refs)}]" if refs else "[không có nhãn bằng chứng]"

    @staticmethod
    def _rule_trace(h: Hypothesis) -> str:
        if not h.fired_rules:
            if h.verdict is Verdict.MODEL_ONLY:
                return "_Không rule nào kích hoạt — cơ chế này chỉ do attribution mô hình gợi ý._"
            return "_Không có dấu vết rule._"
        parts = []
        for f in h.fired_rules:
            op = "<" if f.direction == "below" else ">"
            observed = f"{f.observed_value:g}" if f.observed_value is not None else "n/a"
            parts.append(f"{f.rule_id}: {f.variable} = {observed} {op} {f.threshold:g} {f.unit}")
        return "_Dấu vết luật:_ " + "; ".join(parts)


# =============================================================================
# Claude API
# =============================================================================


class AnthropicNarrator:
    """Gọi Claude API với prompt ràng buộc.

    Lưu ý API (các model hiện tại):
    - KHÔNG truyền `temperature` — tham số này đã bị bỏ trên Opus 5 / Sonnet 5 và
      request sẽ bị trả 400. Tính nhất quán của đầu ra đến từ prompt và từ việc
      bundle là xác định.
    - KHÔNG dùng assistant prefill — cũng đã bị bỏ trên các model này.
    - Thinking mặc định đã bật (adaptive) trên Opus 5; không cần cấu hình gì thêm.
    """

    name = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 4000) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.max_tokens = max_tokens

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Narrator 'anthropic' cần SDK. Cài: pip install anthropic") from exc

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)

    def narrate(self, bundle: EvidenceBundle) -> Answer:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(bundle)}],
            )
        except self._anthropic.APIStatusError as exc:  # pragma: no cover - cần mạng
            raise RuntimeError(
                f"Claude API lỗi {exc.status_code}: {exc.message}. "
                f"Dùng EGXAQ_NARRATOR_BACKEND=dryrun để chạy offline."
            ) from exc

        text = "".join(b.text for b in response.content if b.type == "text")
        return Answer(
            text=text,
            bundle=bundle,
            backend=self.name,
            model=self.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )


def get_narrator(backend: str | None = None) -> Narrator:
    backend = backend or get_settings().narrator_backend
    if backend == "dryrun":
        return DryRunNarrator()
    if backend == "anthropic":
        return AnthropicNarrator()
    raise ValueError(f"Narrator backend không hợp lệ: {backend!r}")
