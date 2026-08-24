"""Nạp và xác thực tầng tri thức từ `knowledge/*.yaml`.

Nguyên tắc P5 (docs/01 §1): tri thức khí quyển KHÔNG nằm trong code Python.
Muốn đổi ngưỡng PBLH → sửa YAML. Module này chỉ đọc và kiểm tra tính nhất quán.

Xác thực ngay lúc nạp (fail fast) vì một tham chiếu rule sai chính tả trong
mechanisms.yaml sẽ làm cơ chế đó im lặng không bao giờ kích hoạt — lỗi kiểu này
rất khó phát hiện qua kết quả đầu ra.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from config import get_settings


class KnowledgeError(ValueError):
    """Tri thức trong YAML không nhất quán."""


# =============================================================================
# Rule
# =============================================================================


class RuleSpec(BaseModel):
    id: str
    name: str
    variable: str
    direction: Literal["below", "above"]
    threshold: float
    saturation: float
    unit: str = ""
    rationale: str = ""
    source: str = ""
    calibration: str = "none"
    requires: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_ramp(self) -> RuleSpec:
        if self.direction == "below" and self.saturation >= self.threshold:
            raise KnowledgeError(
                f"{self.id}: direction=below cần saturation < threshold "
                f"(được {self.saturation} >= {self.threshold})"
            )
        if self.direction == "above" and self.saturation <= self.threshold:
            raise KnowledgeError(
                f"{self.id}: direction=above cần saturation > threshold "
                f"(được {self.saturation} <= {self.threshold})"
            )
        return self

    def strength(self, value: float | None) -> float:
        """Hàm dốc tuyến tính từ threshold (0.0) tới saturation (1.0).

        Chọn tuyến tính thay vì sigmoid vì: (a) chỉ có 2 tham số, cả hai đều có ý
        nghĩa vật lý giải thích được trước hội đồng; (b) không tạo ảo giác về độ
        chính xác mà dữ liệu không hỗ trợ.
        """
        if value is None:
            return 0.0
        span = self.threshold - self.saturation if self.direction == "below" else self.saturation - self.threshold
        raw = (self.threshold - value) / span if self.direction == "below" else (value - self.threshold) / span
        return max(0.0, min(1.0, raw))


# =============================================================================
# Mechanism
# =============================================================================


class TriggerSpec(BaseModel):
    mode: Literal["all", "any"] = "all"
    rules: list[str] = Field(default_factory=list)


class VariableExpectation(BaseModel):
    expected_shap: Literal["positive", "negative", "any"] = "any"

    def sign(self) -> int:
        return {"positive": 1, "negative": -1, "any": 0}[self.expected_shap]


class MechanismSpec(BaseModel):
    id: str
    name: str
    name_en: str
    category: str
    effect: Literal["increase", "decrease"]
    triggers: TriggerSpec
    co_triggers: list[str] = Field(default_factory=list)
    variables: dict[str, VariableExpectation] = Field(default_factory=dict)
    confidence_prior: float = 0.5
    rag_query: str = ""
    explanation_template: str = ""
    limitations: str | None = None
    notes: str | None = None

    def is_suppressor(self) -> bool:
        return self.effect == "decrease"


# =============================================================================
# Feature map
# =============================================================================


class VariableAliases(BaseModel):
    description: str = ""
    unit: str = ""
    exact: list[str] = Field(default_factory=list)
    prefixes: list[str] = Field(default_factory=list)


class FeatureMap(BaseModel):
    version: str = "0"
    source: str = "mock"
    canonical_variables: dict[str, VariableAliases] = Field(default_factory=dict)
    non_mechanistic_exact: list[str] = Field(default_factory=list)
    non_mechanistic_prefixes: list[str] = Field(default_factory=list)

    def resolve(self, feature_name: str) -> str | None:
        """Tên đặc trưng của lab → biến chuẩn hóa. `None` nếu không thuộc cơ chế nào."""
        name = feature_name.strip().lower()
        for canonical, aliases in self.canonical_variables.items():
            if name in {a.lower() for a in aliases.exact}:
                return canonical
        for canonical, aliases in self.canonical_variables.items():
            if any(name.startswith(p.lower()) for p in aliases.prefixes):
                return canonical
        return None

    def is_non_mechanistic(self, feature_name: str) -> bool:
        name = feature_name.strip().lower()
        if name in {a.lower() for a in self.non_mechanistic_exact}:
            return True
        return any(name.startswith(p.lower()) for p in self.non_mechanistic_prefixes)


# =============================================================================
# Knowledge base
# =============================================================================


class KnowledgeBase(BaseModel):
    rules: dict[str, RuleSpec]
    mechanisms: dict[str, MechanismSpec]
    feature_map: FeatureMap
    config: dict[str, float] = Field(default_factory=dict)
    rules_version: str = "0"
    calibrated: bool = False

    def rule(self, rule_id: str) -> RuleSpec:
        try:
            return self.rules[rule_id]
        except KeyError as exc:
            raise KnowledgeError(f"Rule không tồn tại: {rule_id}") from exc

    def mechanisms_for_rule(self, rule_id: str) -> list[MechanismSpec]:
        return [m for m in self.mechanisms.values() if rule_id in m.triggers.rules]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise KnowledgeError(f"Không tìm thấy file tri thức: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_knowledge_base(
    rules_path: Path | None = None,
    mechanisms_path: Path | None = None,
    feature_map_path: Path | None = None,
) -> KnowledgeBase:
    """Nạp toàn bộ tri thức và xác thực tham chiếu chéo."""
    s = get_settings()
    rules_raw = _load_yaml(rules_path or s.rules_path)
    mech_raw = _load_yaml(mechanisms_path or s.mechanisms_path)
    fmap_raw = _load_yaml(feature_map_path or s.feature_map_path)

    rules = {r["id"]: RuleSpec(**r) for r in rules_raw.get("rules", [])}
    mechanisms = {m["id"]: MechanismSpec(**m) for m in mech_raw.get("mechanisms", [])}

    non_mech = fmap_raw.get("non_mechanistic", {}) or {}
    feature_map = FeatureMap(
        version=str(fmap_raw.get("version", "0")),
        source=fmap_raw.get("source", "mock"),
        canonical_variables={
            k: VariableAliases(**v) for k, v in (fmap_raw.get("canonical_variables") or {}).items()
        },
        non_mechanistic_exact=non_mech.get("exact", []),
        non_mechanistic_prefixes=non_mech.get("prefixes", []),
    )

    kb = KnowledgeBase(
        rules=rules,
        mechanisms=mechanisms,
        feature_map=feature_map,
        config={k: float(v) for k, v in (mech_raw.get("config") or {}).items()},
        rules_version=str(rules_raw.get("version", "0")),
        calibrated=bool(rules_raw.get("calibrated", False)),
    )
    _validate(kb)
    return kb


def _validate(kb: KnowledgeBase) -> None:
    """Bắt lỗi im lặng: rule sai chính tả, cơ chế không có rule, biến lạ."""
    problems: list[str] = []

    for mech in kb.mechanisms.values():
        if not mech.triggers.rules:
            problems.append(f"{mech.id}: không có trigger rule nào → không bao giờ kích hoạt")
        for rid in [*mech.triggers.rules, *mech.co_triggers]:
            if rid not in kb.rules:
                problems.append(f"{mech.id}: tham chiếu rule không tồn tại '{rid}'")
        for var in mech.variables:
            if var not in kb.feature_map.canonical_variables:
                problems.append(
                    f"{mech.id}: biến '{var}' không có trong feature_map.canonical_variables "
                    f"→ shap_agreement sẽ luôn = 0"
                )
        if not mech.rag_query.strip():
            problems.append(f"{mech.id}: thiếu rag_query → gated retrieval không chạy được")

    used = {rid for m in kb.mechanisms.values() for rid in [*m.triggers.rules, *m.co_triggers]}
    for rid in kb.rules:
        if rid not in used:
            problems.append(f"Rule {rid} không được cơ chế nào dùng (rule mồ côi)")

    if problems:
        raise KnowledgeError("Tri thức không nhất quán:\n  - " + "\n  - ".join(problems))


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return load_knowledge_base()
