"""Schema dữ liệu đi qua ranh giới các module.

Đây là "hợp đồng" giữa các tầng. Mọi provider (mock hay lab) đều phải trả về đúng
những cấu trúc này — xem docs/02-data-contract.md §7.

Quy ước đặt tên: tên biến LUÔN mang đơn vị (`blh_m`, `wind_speed_ms`, `precip_mm`)
để không bao giờ nhầm Kelvin/Celsius hay mét/milimét.
"""

from __future__ import annotations

from datetime import date as Date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Đầu vào
# =============================================================================


class QuestionContext(BaseModel):
    """Câu hỏi đã được chuẩn hóa về không gian – thời gian."""

    raw_question: str
    lat: float
    lon: float
    date: Date
    step: int = Field(0, ge=0, description="Bước dự báo t+step (0..9 theo mô hình lab)")
    place_name: str | None = None


# =============================================================================
# Tầng bằng chứng 1 — Dữ liệu quan trắc/dự báo
# =============================================================================


class Observation(BaseModel):
    """Điều kiện khí quyển tại một điểm, một thời điểm, một bước dự báo.

    Trường thiếu để `None` — KHÔNG điền giá trị mặc định thay cho dữ liệu thiếu.
    Rule phụ thuộc trường `None` sẽ không kích hoạt và được ghi vào `missing`.
    """

    model_config = ConfigDict(extra="allow")  # cho phép lab thêm biến riêng

    lat: float
    lon: float
    date: Date
    step: int = 0

    # --- mục tiêu ---
    pm25_pred_ugm3: float | None = None
    pm25_obs_ugm3: float | None = None

    # --- biến khí tượng tức thời ---
    blh_m: float | None = None
    wind_speed_ms: float | None = None
    wind_dir_deg: float | None = Field(None, description="Hướng gió THỔI TỚI TỪ, 0=Bắc")
    t2m_c: float | None = None
    t850_c: float | None = None
    rh_pct: float | None = None
    precip_mm: float | None = None
    mslp_hpa: float | None = None

    # --- biến tích lũy (khớp đặc trưng accum của mô hình lab) ---
    wind_speed_mean_2d_ms: float | None = None
    wind_speed_mean_3d_ms: float | None = None
    blh_min_2d_m: float | None = None
    blh_min_3d_m: float | None = None
    precip_sum_3d_mm: float | None = None

    # --- nguồn ngoài ---
    upwind_fire_count: int | None = None
    upwind_fire_distance_km: float | None = None
    source_sector_alignment: float | None = Field(None, ge=0.0, le=1.0)

    # --- khí hậu tham chiếu, để tính anomaly ---
    blh_climatology_m: float | None = None
    wind_speed_climatology_ms: float | None = None

    provenance: str = "unknown"


class DerivedFeatures(BaseModel):
    """Chỉ số dẫn xuất tính từ `Observation`. Rule có thể tham chiếu trực tiếp."""

    lapse_rate_c_per_km: float | None = None
    ventilation_index_m2s: float | None = None
    blh_anomaly_pct: float | None = None
    stagnation_days: float | None = None
    aqi_vn: int | None = None
    aqi_category: str | None = None


# =============================================================================
# Tầng bằng chứng 2 — Attribution của mô hình
# =============================================================================


class FeatureContribution(BaseModel):
    """Đóng góp SHAP của một đặc trưng.

    `shap > 0` = đặc trưng đó ĐẨY dự báo PM2.5 LÊN.
    ⚠ Phải kiểm chứng lại quy ước này với model thật: nếu target là log-PM2.5 thì
    đơn vị của `shap` là log chứ không phải µg/m³ (docs/02 §7.2).
    """

    feature: str
    value: float | None = None
    shap: float
    canonical_variable: str | None = Field(
        None, description="Biến chuẩn hóa sau khi tra knowledge/feature_map.yaml"
    )
    is_non_mechanistic: bool = False


class Attribution(BaseModel):
    """Kết quả SHAP cho một điểm / một bước dự báo."""

    step: int
    model_id: str
    base_value: float | None = None
    prediction: float | None = None
    contributions: list[FeatureContribution] = Field(default_factory=list)
    provenance: str = "unknown"

    def total_abs(self) -> float:
        return sum(abs(c.shap) for c in self.contributions) or 1.0

    def top(self, k: int = 5) -> list[FeatureContribution]:
        return sorted(self.contributions, key=lambda c: abs(c.shap), reverse=True)[:k]

    def for_variable(self, canonical: str) -> list[FeatureContribution]:
        return [c for c in self.contributions if c.canonical_variable == canonical]

    def non_mechanistic_share(self) -> float:
        """Tỉ lệ attribution đến từ đặc trưng KHÔNG mang cơ chế vật lý.

        Chỉ số này là một kết quả nghiên cứu: nó đo mức độ mô hình dựa vào mẫu
        không gian/thời gian thay vì cơ chế khí quyển (docs/02 §6 R2).
        """
        return (
            sum(abs(c.shap) for c in self.contributions if c.is_non_mechanistic) / self.total_abs()
        )


# =============================================================================
# Tầng bằng chứng 3 — Cơ chế (Rule + KG)
# =============================================================================


class RuleFiring(BaseModel):
    """Một rule đã được đánh giá trên dữ liệu quan sát."""

    rule_id: str
    name: str
    variable: str
    observed_value: float | None
    threshold: float
    direction: Literal["below", "above"]
    unit: str
    strength: float = Field(ge=0.0, le=1.0)
    fired: bool
    skipped_reason: str | None = Field(None, description="Vì sao không đánh giá được (thiếu biến)")


# =============================================================================
# Tầng bằng chứng 4 — Tài liệu khoa học
# =============================================================================


class Citation(BaseModel):
    """Một đoạn trích dẫn được từ corpus."""

    evidence_id: str = Field(description="Nhãn dùng trong câu trả lời, ví dụ E1")
    doi: str | None = None
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    section: str | None = None
    text: str
    score: float = 0.0
    url: str | None = None
    is_open_access: bool = False

    def short(self) -> str:
        who = (self.authors or "n/a").split(",")[0]
        return f"{who} ({self.year or 'n.d.'})"


# =============================================================================
# Tổng hợp — Giả thuyết nhân quả
# =============================================================================


class Verdict(str, Enum):
    """Kết quả đối chiếu Rule ↔ SHAP. Đây là đóng góp nghiên cứu số 2."""

    CONFIRMED = "CONFIRMED"  # rule kích hoạt + SHAP đồng thuận cùng chiều
    PARTIAL = "PARTIAL"  # rule kích hoạt, SHAP không nói gì rõ ràng
    CONFLICT = "CONFLICT"  # rule kích hoạt nhưng SHAP ngược chiều → cảnh báo
    MODEL_ONLY = "MODEL_ONLY"  # SHAP nhấn mạnh nhưng rule không kích hoạt
    NO_SHAP = "NO_SHAP"  # rule kích hoạt nhưng không có attribution để đối chiếu
    INACTIVE = "INACTIVE"  # cả rule lẫn SHAP đều không nói gì → loại khỏi câu trả lời


class Confidence(str, Enum):
    HIGH = "CAO"
    MEDIUM = "TRUNG BÌNH"
    LOW = "THẤP"
    INSUFFICIENT = "KHÔNG ĐỦ CĂN CỨ"


class Hypothesis(BaseModel):
    """Một giả thuyết nguyên nhân, kèm toàn bộ dấu vết bằng chứng."""

    mechanism_id: str
    name: str
    name_en: str
    category: str
    effect: Literal["increase", "decrease"]

    # --- ba nguồn bằng chứng ---
    rule_strength: float = Field(0.0, ge=0.0, le=1.0)
    fired_rules: list[RuleFiring] = Field(default_factory=list)

    shap_agreement: float = Field(0.0, ge=-1.0, le=1.0, description="[-1,1]; âm = mâu thuẫn")
    shap_features: list[FeatureContribution] = Field(default_factory=list)

    rag_support: float = Field(0.0, ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)

    # --- tổng hợp ---
    confidence_prior: float = 0.5
    score: float = 0.0
    verdict: Verdict = Verdict.NO_SHAP
    confidence: Confidence = Confidence.INSUFFICIENT

    # --- diễn đạt ---
    narrative: str = ""
    limitations: str | None = None
    rag_query: str = ""

    def evidence_ids(self) -> list[str]:
        return [c.evidence_id for c in self.citations]


# =============================================================================
# Gói bằng chứng gửi cho LLM narrator
# =============================================================================


class DataEvidence(BaseModel):
    """Một quan sát cụ thể, có nhãn để narrator trích dẫn."""

    evidence_id: str  # D1, D2, ...
    label: str
    value: float
    unit: str
    context: str | None = Field(None, description="So sánh với bình thường, nếu có")


class EvidenceBundle(BaseModel):
    """Đầu vào DUY NHẤT của LLM narrator.

    INV-1: narrator không được dùng bất cứ thông tin nào ngoài bundle này.
    """

    question: QuestionContext
    place_label: str

    observation: Observation
    derived: DerivedFeatures
    data_evidence: list[DataEvidence] = Field(default_factory=list)

    attribution: Attribution | None = None
    shap_summary: list[FeatureContribution] = Field(default_factory=list)
    non_mechanistic_share: float | None = None

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    suppressors: list[Hypothesis] = Field(default_factory=list)

    conflicts: list[str] = Field(default_factory=list, description="Cảnh báo Rule ↔ SHAP mâu thuẫn")
    missing: list[str] = Field(
        default_factory=list, description="Dữ liệu thiếu — narrator phải nêu"
    )
    config_flags: dict[str, Any] = Field(default_factory=dict, description="Cấu hình ablation")

    overall_confidence: Confidence = Confidence.INSUFFICIENT

    def all_citations(self) -> list[Citation]:
        seen: dict[str, Citation] = {}
        for h in [*self.hypotheses, *self.suppressors]:
            for c in h.citations:
                seen.setdefault(c.evidence_id, c)
        return list(seen.values())


class Answer(BaseModel):
    """Đầu ra cuối cùng."""

    text: str
    bundle: EvidenceBundle
    backend: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
