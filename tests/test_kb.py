"""Tri thức trong YAML phải nhất quán — bắt lỗi im lặng trước khi nó thành lỗi logic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kb import KnowledgeError, RuleSpec, load_knowledge_base


def test_knowledge_base_loads_and_validates(kb):
    assert kb.rules, "rules.yaml rỗng"
    assert kb.mechanisms, "mechanisms.yaml rỗng"
    assert kb.feature_map.canonical_variables


def test_every_mechanism_has_a_rag_query(kb):
    """Cơ chế thiếu rag_query thì gated retrieval không chạy được cho nó."""
    for mech in kb.mechanisms.values():
        assert mech.rag_query.strip(), f"{mech.id} thiếu rag_query"


def test_every_threshold_documents_its_source(kb):
    """Hội đồng sẽ hỏi 'con số này ở đâu ra' — câu trả lời phải nằm trong YAML."""
    for rule in kb.rules.values():
        assert rule.rationale.strip(), f"{rule.id} thiếu rationale"
        assert rule.source.strip(), f"{rule.id} thiếu source"


def test_mechanism_variables_exist_in_feature_map(kb):
    """Biến không có trong feature_map → shap_agreement luôn 0, consistency vô dụng."""
    known = set(kb.feature_map.canonical_variables)
    for mech in kb.mechanisms.values():
        assert set(mech.variables) <= known, f"{mech.id} tham chiếu biến lạ"


def test_suppressor_mechanisms_exist(kb):
    """Phải có cơ chế LÀM GIẢM, nếu không hệ thống không trả lời được 'vì sao hôm nay sạch'."""
    assert any(m.is_suppressor() for m in kb.mechanisms.values())


def test_rules_yaml_marked_uncalibrated(kb):
    """Nhắc rằng ngưỡng chưa hiệu chỉnh theo phân phối GFS của lab.

    Khi chạy scripts/calibrate_thresholds.py xong thì đổi cờ và sửa test này.
    """
    assert kb.calibrated is False


# --------------------------------------------------------------------------- ramp


def test_rule_strength_ramp_below():
    rule = RuleSpec(
        id="X", name="t", variable="v", direction="below", threshold=500.0, saturation=200.0
    )
    assert rule.strength(600.0) == 0.0  # trên ngưỡng → không kích hoạt
    assert rule.strength(500.0) == 0.0  # đúng ngưỡng → biên
    assert rule.strength(350.0) == pytest.approx(0.5)
    assert rule.strength(200.0) == 1.0
    assert rule.strength(50.0) == 1.0  # bão hòa, không vượt 1
    assert rule.strength(None) == 0.0  # thiếu dữ liệu → 0, không đoán


def test_rule_strength_ramp_above():
    rule = RuleSpec(
        id="X", name="t", variable="v", direction="above", threshold=1.0, saturation=11.0
    )
    assert rule.strength(0.5) == 0.0
    assert rule.strength(6.0) == pytest.approx(0.5)
    assert rule.strength(20.0) == 1.0


def test_invalid_ramp_rejected():
    """Ngưỡng và điểm bão hòa đặt ngược nhau sẽ làm rule kích hoạt ngược — phải nổ ngay.

    Pydantic bọc mọi lỗi trong validator thành `ValidationError`, nên bắt lớp đó
    và kiểm nội dung thông điệp của `KnowledgeError` bên trong.
    """
    with pytest.raises(ValidationError, match="saturation < threshold"):
        RuleSpec(
            id="X", name="t", variable="v", direction="below", threshold=200.0, saturation=500.0
        )


def test_dangling_rule_reference_is_caught(tmp_path, kb):
    """Rule sai chính tả trong mechanisms.yaml phải nổ ngay lúc nạp."""
    rules = tmp_path / "rules.yaml"
    mechs = tmp_path / "mechanisms.yaml"
    fmap = tmp_path / "feature_map.yaml"

    rules.write_text(
        "version: '1'\nrules:\n"
        "  - {id: R1, name: n, variable: blh_m, direction: below,"
        " threshold: 500, saturation: 200, rationale: r, source: s}\n",
        encoding="utf-8",
    )
    mechs.write_text(
        "version: '1'\nmechanisms:\n"
        "  - id: M1\n    name: n\n    name_en: n\n    category: accumulation\n"
        "    effect: increase\n    triggers: {mode: all, rules: [R_TYPO]}\n"
        "    variables: {blh: {expected_shap: positive}}\n    rag_query: q\n",
        encoding="utf-8",
    )
    fmap.write_text(
        "version: '1'\ncanonical_variables:\n  blh: {exact: [blh]}\n", encoding="utf-8"
    )

    with pytest.raises(KnowledgeError, match="R_TYPO"):
        load_knowledge_base(rules, mechs, fmap)


# --------------------------------------------------------------------------- feature map


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("blh_min_2d", "blh"),  # tên có hậu tố accum của lab
        ("BLH_MEAN", "blh"),  # không phân biệt hoa thường
        ("wind_speed_mean_3d", "wind_speed"),
        ("precip_sum_3d", "precip"),
        ("rh_mean", "humidity"),
        ("mslp_mean", "pressure"),
        ("fire_count_upwind", "fire"),
        ("pm25_lag1", "persistence"),
        ("some_unknown_feature", None),
    ],
)
def test_feature_map_resolves_lab_naming(kb, feature, expected):
    assert kb.feature_map.resolve(feature) == expected


@pytest.mark.parametrize("feature", ["lat", "lon", "doy_sin", "elevation", "landuse_urban"])
def test_non_mechanistic_features_flagged(kb, feature):
    """Các biến này dự báo được nhưng không giải thích được gì về nhân quả."""
    assert kb.feature_map.is_non_mechanistic(feature)
