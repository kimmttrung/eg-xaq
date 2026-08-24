"""Prompt ràng buộc cho LLM narrator.

INV-1: LLM KHÔNG phải nguồn của nguyên nhân. Nó nhận `EvidenceBundle` đã hoàn
chỉnh và chỉ được diễn đạt lại bằng tiếng Việt.

Nguyên tắc viết prompt ở đây:

1. **Phủ định kiến thức nền.** Nói thẳng "bạn KHÔNG có kiến thức riêng về nguyên
   nhân ô nhiễm". Nếu không, mô hình sẽ bổ sung những điều nó biết về PM2.5 —
   thường là đúng, và vì đúng nên rất khó phát hiện, mà vẫn là hallucination vì
   không truy vết được.
2. **Độ tin cậy do backend quyết định.** Nếu để LLM tự chấm, nó chấm theo độ trôi
   chảy của văn bản chứ không theo bằng chứng.
3. **Mâu thuẫn phải nêu, không được hòa giải.** Khi Rule và SHAP xung đột, nhiệm
   vụ của narrator là trình bày xung đột, không phải chọn bên hay làm mượt nó đi.
4. **Không đặt `temperature`.** Các model hiện tại (Opus 5, Sonnet 5) đã bỏ tham
   số này và trả 400 nếu gửi. Tính ổn định của đầu ra đến từ ràng buộc của prompt
   và từ việc bundle là xác định, không đến từ temperature.
"""

from __future__ import annotations

import json

from schemas import EvidenceBundle

SYSTEM_PROMPT = """\
Bạn là bộ phận DIỄN ĐẠT của hệ thống EG-XAQ — hệ thống giải thích nguyên nhân ô nhiễm
PM2.5 dựa trên bằng chứng.

BẠN KHÔNG CÓ KIẾN THỨC NỀN RIÊNG VỀ NGUYÊN NHÂN Ô NHIỄM.
Mọi điều bạn viết phải lấy từ khối EVIDENCE được cung cấp. Kể cả khi bạn "biết" một
cơ chế khí quyển nào đó là đúng, nếu nó không có trong EVIDENCE thì không được nhắc tới.

QUY TẮC BẮT BUỘC
1. Mỗi tuyên bố nguyên nhân PHẢI kèm nhãn bằng chứng: [D#] cho số liệu quan trắc,
   [S#] cho attribution của mô hình, [E#] cho trích dẫn khoa học, [R#] cho luật.
2. Không đủ bằng chứng để kết luận điều gì → viết rõ "Không đủ căn cứ để khẳng định ...".
   Tuyệt đối không suy đoán bù, không thêm cơ chế ngoài EVIDENCE.
3. Độ tin cậy đã được tính sẵn ở trường `confidence`. Chép lại đúng giá trị đó.
   KHÔNG tự đánh giá lại theo cảm nhận của bạn.
4. Nếu `conflicts` không rỗng, PHẢI trình bày mâu thuẫn đó thành một mục riêng.
   Không được chọn bên, không được làm nhẹ đi, không được bỏ qua.
5. Nếu `missing` không rỗng, PHẢI nêu ở mục hạn chế. Dữ liệu thiếu nghĩa là cơ chế
   liên quan CHƯA ĐƯỢC KIỂM TRA — không phải đã bị loại trừ. Diễn đạt đúng sắc thái đó.
6. Phân biệt rõ hai loại phát biểu:
   - "Mô hình dự báo dựa nhiều vào biến X" (nói về MÔ HÌNH)
   - "Điều kiện khí quyển X gây tích tụ" (nói về THỰC TẠI)
   Không được đánh đồng. SHAP giải thích mô hình, không chứng minh nhân quả thực tế.
7. Cơ chế thuộc nhóm `enabling_condition` là điều kiện CHO PHÉP, không phải nguyên
   nhân chủ động. Ví dụ: viết "không mưa nên ô nhiễm không được rửa trôi", KHÔNG viết
   "không mưa gây ô nhiễm".

ĐỊNH DẠNG TRẢ LỜI (tiếng Việt, dùng đúng các tiêu đề sau)
**Kết luận**
Hai đến ba câu, nêu nguyên nhân chính kèm nhãn bằng chứng.

**Bằng chứng dữ liệu**
Các số liệu quan trắc/dự báo liên quan, kèm [D#].

**Bằng chứng từ mô hình**
Attribution nói gì, kèm [S#]. Nhắc rằng đây là lý do của MÔ HÌNH.

**Cơ chế khoa học**
Từng cơ chế kèm trích dẫn [E#]. Cơ chế không có trích dẫn → ghi rõ "chưa có tài liệu
trong kho hỗ trợ cơ chế này".

**Điểm bất định** (chỉ khi `conflicts` không rỗng)

**Độ tin cậy & hạn chế**
Chép `overall_confidence`, rồi liệt kê `missing`.

Viết gọn, chính xác, không hoa mỹ. Không mở đầu bằng lời chào hay tóm tắt lại câu hỏi.
"""


def build_evidence_json(bundle: EvidenceBundle) -> dict:
    """Chuyển bundle thành JSON gọn cho LLM.

    Cố ý KHÔNG gửi nguyên `bundle.model_dump()`: bundle chứa nhiều trạng thái trung
    gian phục vụ đánh giá (điểm số thô, toàn bộ rule firing) mà nếu đưa vào prompt
    chỉ làm loãng và tăng nguy cơ mô hình bám vào con số không nên nhắc tới.
    """
    obs = bundle.observation

    payload: dict = {
        "location": bundle.place_label,
        "coordinates": [obs.lat, obs.lon],
        "date": str(obs.date),
        "forecast_step": f"t+{obs.step}",
        "question": bundle.question.raw_question,
        "aqi_vn": bundle.derived.aqi_vn,
        "aqi_category": bundle.derived.aqi_category,
        "data_evidence": [
            {
                "id": e.evidence_id,
                "label": e.label,
                "value": e.value,
                "unit": e.unit,
                "context": e.context,
            }
            for e in bundle.data_evidence
        ],
        "model_attribution": _attribution_payload(bundle),
        "hypotheses": [_hypothesis_payload(h) for h in bundle.hypotheses],
        "suppressors": [_hypothesis_payload(h) for h in bundle.suppressors],
        "conflicts": bundle.conflicts,
        "missing": bundle.missing,
        "overall_confidence": bundle.overall_confidence.value,
        "enabled_layers": bundle.config_flags,
    }
    return payload


def _attribution_payload(bundle: EvidenceBundle) -> dict | None:
    if bundle.attribution is None:
        return None
    return {
        "model_id": bundle.attribution.model_id,
        "base_value": bundle.attribution.base_value,
        "prediction": bundle.attribution.prediction,
        "note": (
            "Giá trị SHAP giải thích DỰ BÁO CỦA MÔ HÌNH, không phải nhân quả thực tế."
        ),
        "non_mechanistic_share": bundle.non_mechanistic_share,
        "top_features": [
            {
                "id": f"S{i + 1}",
                "feature": c.feature,
                "value": c.value,
                "shap": c.shap,
                "direction": "đẩy PM2.5 lên" if c.shap > 0 else "kéo PM2.5 xuống",
                "mechanistic": not c.is_non_mechanistic,
            }
            for i, c in enumerate(bundle.shap_summary)
        ],
    }


def _hypothesis_payload(h) -> dict:
    return {
        "mechanism_id": h.mechanism_id,
        "name": h.name,
        "category": h.category,
        "effect": h.effect,
        "narrative": h.narrative,
        "score": h.score,
        "verdict": h.verdict.value,
        "confidence": h.confidence.value,
        "rule_strength": round(h.rule_strength, 3),
        "shap_agreement": round(h.shap_agreement, 3),
        "rag_support": round(h.rag_support, 3),
        "fired_rules": [
            {
                "id": f.rule_id,
                "name": f.name,
                "variable": f.variable,
                "observed": f.observed_value,
                "threshold": f.threshold,
                "direction": f.direction,
                "unit": f.unit,
            }
            for f in h.fired_rules
        ],
        "citations": [
            {
                "id": c.evidence_id,
                "authors": c.authors,
                "year": c.year,
                "title": c.title,
                "doi": c.doi,
                "quote": c.text[:600],
            }
            for c in h.citations
        ],
        "limitations": h.limitations,
    }


def build_user_message(bundle: EvidenceBundle) -> str:
    """Thông điệp user: câu hỏi + khối EVIDENCE dạng JSON."""
    evidence = json.dumps(build_evidence_json(bundle), ensure_ascii=False, indent=2)
    return (
        f"CÂU HỎI: {bundle.question.raw_question}\n\n"
        f"EVIDENCE (nguồn thông tin DUY NHẤT bạn được dùng):\n"
        f"```json\n{evidence}\n```\n\n"
        f"Viết câu trả lời theo đúng định dạng đã quy định."
    )
