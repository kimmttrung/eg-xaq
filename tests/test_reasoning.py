"""Lõi suy luận: derive → rule → giả thuyết → consistency → scoring.

Các test ở đây bảo vệ đúng những gì làm nên đóng góp nghiên cứu. Nếu một thay đổi
làm vỡ chúng thì thay đổi đó đang phá luận điểm của khóa luận, không phải phá code.
"""

from __future__ import annotations

from datetime import date as Date

import pytest

from geoutils import angular_diff_deg, bearing_deg, is_upwind, uv_to_speed_dir
from reasoning.consistency import assess_consistency
from reasoning.derive import derive_features, feature_context, pm25_to_aqi_vn, variable_lookup
from reasoning.hypotheses import combine_rule_strengths, render_template
from reasoning.rules import RuleEngine
from reasoning.scoring import ScoringWeights, overall_confidence, score_hypotheses
from schemas import Attribution, Confidence, FeatureContribution, Observation, Verdict

# =============================================================================
# Chỉ số dẫn xuất
# =============================================================================


def test_ventilation_index_uses_same_day_values(observation_of):
    """VI là đại lượng theo ngày; khía cạnh nhiều ngày do stagnation_days lo (tránh đếm hai lần)."""
    obs = observation_of("winter_inversion")
    derived = derive_features(obs)
    assert derived.ventilation_index_m2s == pytest.approx(320.0 * 0.8)


def test_ventilation_index_separates_clean_day_from_polluted_day(observation_of):
    """Ngày gió mạnh phải cho VI cao hơn hẳn ngày tù đọng — nếu không, R7/R8 vô dụng."""
    dirty = derive_features(observation_of("winter_inversion")).ventilation_index_m2s
    clean = derive_features(observation_of("cold_surge_clean")).ventilation_index_m2s
    assert clean > 10 * dirty


def test_inversion_detected_when_t850_above_t2m(observation_of):
    obs = observation_of("winter_inversion")
    derived = derive_features(obs)
    assert derived.lapse_rate_c_per_km is not None
    assert derived.lapse_rate_c_per_km < 0, "t850 > t2m phải cho lapse rate âm"


def test_derived_features_are_none_when_input_missing(observation_of):
    """INV-3: thiếu đầu vào → None, tuyệt đối không có giá trị mặc định."""
    obs = observation_of("sparse_data")
    derived = derive_features(obs)
    assert derived.lapse_rate_c_per_km is None  # thiếu t850
    assert derived.ventilation_index_m2s is None  # thiếu gió
    assert derived.stagnation_days is None  # không có biến gió nào


def test_stagnation_days_is_none_without_any_wind_data():
    """INV-3: "không biết" khác "biết là không".

    Trả 0.0 ở đây từng sinh ra một mẩu bằng chứng có nhãn ("Số ngày tù đọng: 0
    ngày") tính từ chỗ không hề có dữ liệu — đúng thứ mà một khóa luận về
    evidence-grounding không được phép để lọt.
    """
    empty = Observation(lat=21.03, lon=105.85, date=Date(2024, 1, 15), step=0)
    assert derive_features(empty).stagnation_days is None
    assert feature_context(empty, derive_features(empty)) == []


def test_stagnation_days_is_zero_when_wind_is_measured_and_strong():
    """Có đo gió và gió mạnh → 0.0 là câu trả lời THẬT, không phải giá trị mặc định."""
    windy = Observation(lat=21.03, lon=105.85, date=Date(2024, 1, 15), step=0, wind_speed_ms=6.0)
    assert derive_features(windy).stagnation_days == 0.0


@pytest.mark.parametrize(
    ("pm25", "category"),
    [(10.0, "Tốt"), (40.0, "Trung bình"), (65.0, "Kém"), (95.0, "Xấu"), (200.0, "Rất xấu")],
)
def test_aqi_vn_categories(pm25, category):
    aqi, label = pm25_to_aqi_vn(pm25)
    assert label == category
    assert aqi is not None


def test_aqi_none_when_no_pm25():
    assert pm25_to_aqi_vn(None) == (None, None)


# =============================================================================
# Hình học gió — bẫy hay sai nhất
# =============================================================================


def test_uv_to_direction_uses_meteorological_convention():
    """Gió thổi VỀ hướng đông (u>0) là gió TỪ hướng tây → 270°."""
    speed, direction = uv_to_speed_dir(5.0, 0.0)
    assert speed == pytest.approx(5.0)
    assert direction == pytest.approx(270.0)

    _, from_south = uv_to_speed_dir(0.0, 5.0)
    assert from_south == pytest.approx(180.0)


def test_bearing_and_angular_diff():
    assert bearing_deg(21.0, 105.0, 22.0, 105.0) == pytest.approx(0.0, abs=0.5)
    assert angular_diff_deg(350.0, 10.0) == pytest.approx(20.0)


def test_upwind_detection():
    """Nguồn ở phía nam, gió thổi từ nam (180°) → nguồn nằm thượng nguồn."""
    upwind, dist = is_upwind(21.0, 105.85, 20.5, 105.85, wind_dir_deg=180.0)
    assert upwind is True
    assert dist == pytest.approx(55.6, rel=0.05)

    # cùng nguồn đó nhưng gió thổi từ bắc → không còn thượng nguồn
    upwind, _ = is_upwind(21.0, 105.85, 20.5, 105.85, wind_dir_deg=0.0)
    assert upwind is False


# =============================================================================
# Rule engine
# =============================================================================


def test_winter_episode_fires_accumulation_rules(kb, observation_of):
    obs = observation_of("winter_inversion")
    variables = variable_lookup(obs, derive_features(obs))
    evaluation = RuleEngine(kb).evaluate(variables)

    for rule_id in ("R1", "R2", "R3", "R7", "R11", "R13"):
        assert evaluation.strength(rule_id) > 0, f"{rule_id} lẽ ra phải kích hoạt"
    assert evaluation.strength("R4") == 0.0, "không mưa thì không có rửa trôi"
    assert evaluation.strength("R9") == 0.0, "gió lặng thì không có bình lưu mạnh"


def test_clean_episode_fires_removal_rules(kb, observation_of):
    obs = observation_of("cold_surge_clean")
    variables = variable_lookup(obs, derive_features(obs))
    evaluation = RuleEngine(kb).evaluate(variables)

    assert evaluation.strength("R9") > 0, "gió 6.4 m/s phải kích hoạt bình lưu mạnh"
    assert evaluation.strength("R8") > 0, "thông gió tốt"
    assert evaluation.strength("R1") == 0.0, "PBLH 1350 m không phải lớp xáo trộn thấp"


def test_missing_data_skips_rules_with_reason(kb, observation_of):
    """Rule không đánh giá được phải nói RÕ vì sao, không im lặng trả 0."""
    obs = observation_of("sparse_data")
    variables = variable_lookup(obs, derive_features(obs))
    evaluation = RuleEngine(kb).evaluate(variables)

    firing = evaluation.firings["R2"]  # cần wind_speed_ms, kịch bản này không có
    assert firing.fired is False
    assert firing.skipped_reason is not None
    assert "wind_speed_ms" in firing.skipped_reason
    assert "wind_speed_ms" in evaluation.missing_variables


# =============================================================================
# Gộp giả thuyết
# =============================================================================


def test_co_triggers_cannot_create_a_mechanism_from_nothing(kb):
    """Ràng buộc thiết kế: cơ chế phải được rule CHÍNH của nó kích hoạt."""
    mech = kb.mechanisms["MECH_LOW_PBLH"]
    strengths = dict.fromkeys(kb.rules, 1.0)
    strengths["R1"] = 0.0  # rule chính im lặng

    strength, _ = combine_rule_strengths(mech, strengths)
    assert strength == 0.0


def test_co_triggers_boost_but_do_not_dominate(kb):
    mech = kb.mechanisms["MECH_LOW_PBLH"]
    alone = combine_rule_strengths(mech, {"R1": 0.5})[0]
    boosted = combine_rule_strengths(mech, {"R1": 0.5, "R2": 1.0, "R11": 1.0, "R10": 1.0})[0]

    assert boosted > alone
    assert boosted < 1.0


def test_mode_all_uses_weakest_condition(kb):
    """`all` lấy MIN: cơ chế chỉ mạnh bằng điều kiện yếu nhất của nó."""
    mech = kb.mechanisms["MECH_POOR_VENTILATION"]
    assert mech.triggers.mode == "all"
    strength, _ = combine_rule_strengths(mech, {"R7": 0.3})
    assert strength == pytest.approx(0.3)


def test_template_marks_missing_values_instead_of_inventing_them():
    text = render_template("PBLH {blh_m:.0f} m, gió {wind_speed_ms:.1f} m/s", {"blh_m": 320.0})
    assert "320 m" in text
    assert "(không có dữ liệu)" in text


# =============================================================================
# Consistency — đóng góp nghiên cứu số 2
# =============================================================================


def _attribution(pairs: list[tuple[str, float]]) -> Attribution:
    return Attribution(
        step=0,
        model_id="test",
        base_value=40.0,
        prediction=90.0,
        contributions=[FeatureContribution(feature=f, value=None, shap=s) for f, s in pairs],
    )


def _hypothesis(kb, engine, obs, attribution):
    result = engine.reason(obs, attribution)
    return {h.mechanism_id: h for h in result.hypotheses}


def test_aligned_shap_confirms_fired_mechanism(kb, engine, observation_of):
    obs = observation_of("winter_inversion")
    attribution = _attribution([("blh_min_2d", 20.0), ("wind_speed_mean_2d", 15.0)])
    hyps = _hypothesis(kb, engine, obs, attribution)

    assert hyps["MECH_LOW_PBLH"].verdict is Verdict.CONFIRMED
    assert hyps["MECH_LOW_PBLH"].shap_agreement > 0


def test_opposite_shap_raises_conflict(kb, engine, observation_of):
    """Rule nói PBLH thấp gây tích tụ, SHAP nói ngược → CONFLICT, không được nuốt."""
    obs = observation_of("winter_inversion")
    attribution = _attribution([("blh_min_2d", -30.0), ("t2m_mean", 2.0)])
    result = engine.reason(obs, attribution)
    hyps = {h.mechanism_id: h for h in result.hypotheses}

    assert hyps["MECH_LOW_PBLH"].verdict is Verdict.CONFLICT
    assert hyps["MECH_LOW_PBLH"].shap_agreement < 0
    assert any("MÂU THUẪN" in c for c in result.conflicts)


def test_shap_without_rule_support_is_model_only(kb, engine, observation_of):
    """Mô hình nhấn mạnh mưa trong khi trời không mưa → không phải nguyên nhân thực tế."""
    obs = observation_of("winter_inversion")
    attribution = _attribution([("precip_sum_3d", -50.0)])
    result = engine.reason(obs, attribution)
    hyps = {h.mechanism_id: h for h in result.hypotheses}

    assert hyps["MECH_WET_DEPOSITION"].verdict is Verdict.MODEL_ONLY
    assert any("CHỈ MÔ HÌNH" in c for c in result.conflicts)


def test_no_attribution_gives_no_shap_verdict(kb, engine, observation_of):
    obs = observation_of("winter_inversion")
    result = engine.reason(obs, attribution=None)
    hyps = {h.mechanism_id: h for h in result.hypotheses}

    assert hyps["MECH_LOW_PBLH"].verdict is Verdict.NO_SHAP
    assert hyps["MECH_LOW_PBLH"].shap_agreement == 0.0


def test_non_mechanistic_dominance_raises_warning(kb, engine, observation_of):
    """Mô hình dựa vào toạ độ và ngày trong năm → cảnh báo, không im lặng."""
    obs = observation_of("winter_inversion")
    attribution = _attribution(
        [("lat", 30.0), ("lon", 25.0), ("doy_sin", 20.0), ("blh_min_2d", 5.0)]
    )
    result = engine.reason(obs, attribution)
    assert any("CẢNH BÁO CƠ CHẾ" in c for c in result.conflicts)
    assert attribution.non_mechanistic_share() > 0.3


def test_annotation_marks_canonical_variables(kb, observation_of, attribution_of):
    attribution = attribution_of("winter_inversion")
    assess_consistency(kb, [], attribution)  # không đủ; cần annotate trước
    from reasoning.consistency import annotate_contributions

    annotate_contributions(kb, attribution)
    by_name = {c.feature: c for c in attribution.contributions}
    assert by_name["blh_min_2d"].canonical_variable == "blh"
    assert by_name["lat"].is_non_mechanistic is True


# =============================================================================
# Scoring
# =============================================================================


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="α"):
        ScoringWeights(rule=0.5, shap=0.5, rag=0.5)


def test_confidence_high_needs_all_three_sources(kb, engine, observation_of):
    obs = observation_of("winter_inversion")
    attribution = _attribution([("blh_min_2d", 25.0), ("wind_speed_mean_2d", 15.0)])
    result = engine.reason(obs, attribution)

    target = next(h for h in result.hypotheses if h.mechanism_id == "MECH_LOW_PBLH")
    causes, _ = score_hypotheses(result.hypotheses)
    scored = next(h for h in causes if h.mechanism_id == "MECH_LOW_PBLH")
    assert scored.confidence is Confidence.MEDIUM  # rule + shap, chưa có RAG

    target.rag_support = 0.8
    causes, _ = score_hypotheses(result.hypotheses)
    scored = next(h for h in causes if h.mechanism_id == "MECH_LOW_PBLH")
    assert scored.confidence is Confidence.HIGH


def test_conflict_is_penalised_but_kept(kb, engine, observation_of):
    """Giả thuyết mâu thuẫn vẫn phải xuất hiện — nó là điểm bất định cần nêu."""
    obs = observation_of("winter_inversion")
    attribution = _attribution([("blh_min_2d", -40.0)])
    result = engine.reason(obs, attribution)
    causes, _ = score_hypotheses(result.hypotheses)

    conflicted = [h for h in causes if h.verdict is Verdict.CONFLICT]
    assert conflicted, "giả thuyết CONFLICT không được biến mất khỏi câu trả lời"
    assert all(h.confidence is Confidence.LOW for h in conflicted)


def test_removal_mechanisms_go_to_suppressors(kb, engine, observation_of):
    obs = observation_of("rain_washout")
    result = engine.reason(obs, attribution=None)
    causes, suppressors = score_hypotheses(result.hypotheses)

    ids = {h.mechanism_id for h in suppressors}
    assert "MECH_WET_DEPOSITION" in ids
    assert all(h.effect == "increase" for h in causes)


def test_overall_confidence_downgraded_by_conflicts_and_missing_rag(kb, engine, observation_of):
    obs = observation_of("winter_inversion")
    result = engine.reason(obs, attribution=None)
    causes, _ = score_hypotheses(result.hypotheses)

    with_rag = overall_confidence(causes, conflicts=[], used_rag=True)
    no_rag = overall_confidence(causes, conflicts=[], used_rag=False)
    with_conflict = overall_confidence(causes, conflicts=["x"], used_rag=True)

    order = [Confidence.INSUFFICIENT, Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]
    assert order.index(no_rag) < order.index(with_rag)
    assert order.index(with_conflict) < order.index(with_rag)


def test_empty_causes_gives_insufficient():
    assert overall_confidence([], [], used_rag=True) is Confidence.INSUFFICIENT
