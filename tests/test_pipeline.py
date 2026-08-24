"""Pipeline end-to-end trên các kịch bản, và các bất biến INV-1/2/3.

Đây là lưới an toàn quan trọng nhất: khi cắm data thật của lab vào, những test này
phải vẫn xanh. Nếu chúng đỏ, nghĩa là lõi đã lỡ phụ thuộc vào mock ở đâu đó
(docs/02-data-contract.md §8 bước 6).
"""

from __future__ import annotations

import pytest

from data.mock import HANOI_LAT, HANOI_LON, MockObservationProvider
from narrator.narrator import DryRunNarrator
from narrator.prompts import SYSTEM_PROMPT, build_evidence_json
from pipeline import ExplanationPipeline, PipelineConfig
from schemas import Confidence
from xai.mock import MockAttributionProvider


def build(episode: str, ablation: str = "E", xai_mode: str = "aligned"):
    pipeline = ExplanationPipeline(
        observation_provider=MockObservationProvider(episode=episode),
        attribution_provider=MockAttributionProvider(mode=xai_mode),
        narrator=DryRunNarrator(),
        config=PipelineConfig.ablation(ablation),
    )
    date = MockObservationProvider.default_date(episode)
    bundle = pipeline.build_bundle("Vì sao không khí hôm nay như vậy?", HANOI_LAT, HANOI_LON, date)
    return pipeline, bundle


# =============================================================================
# Từng kịch bản phải cho ra cơ chế ĐÚNG về mặt khí tượng
# =============================================================================


def test_winter_inversion_explained_by_accumulation():
    _, bundle = build("winter_inversion")
    top_ids = [h.mechanism_id for h in bundle.hypotheses[:4]]

    assert any(
        mid in top_ids
        for mid in ("MECH_LOW_PBLH", "MECH_POOR_VENTILATION", "MECH_INVERSION", "MECH_STAGNATION")
    ), f"kịch bản nghịch nhiệt lẽ ra phải ra cơ chế tích tụ, nhận được {top_ids}"


def test_biomass_episode_surfaces_transport():
    _, bundle = build("biomass_burning")
    ids = [h.mechanism_id for h in bundle.hypotheses]
    assert "MECH_BIOMASS_TRANSPORT" in ids


def test_clean_day_is_explained_by_removal_not_by_causes():
    """Hệ thống phải trả lời được cả câu 'vì sao hôm nay SẠCH'."""
    _, bundle = build("cold_surge_clean")
    suppressor_ids = {h.mechanism_id for h in bundle.suppressors}
    assert "MECH_ADVECTION_CLEANSING" in suppressor_ids
    assert bundle.derived.aqi_category == "Tốt"


def test_clean_day_narrative_leads_with_removal_mechanism():
    """Ngày AQI tốt thì câu hỏi thực sự là 'vì sao SẠCH' — không được mở đầu bằng 'nguyên nhân'."""
    _, bundle = build("cold_surge_clean")
    text = DryRunNarrator().narrate(bundle).text
    conclusion = text.split("**Bằng chứng dữ liệu**")[0]

    assert "Không khí ở mức tốt" in conclusion
    assert "Nguyên nhân chủ đạo" not in conclusion


def test_polluted_day_narrative_leads_with_cause():
    _, bundle = build("winter_inversion")
    conclusion = DryRunNarrator().narrate(bundle).text.split("**Bằng chứng dữ liệu**")[0]
    assert "Nguyên nhân chủ đạo" in conclusion


def test_rain_day_surfaces_wet_deposition():
    _, bundle = build("rain_washout")
    assert "MECH_WET_DEPOSITION" in {h.mechanism_id for h in bundle.suppressors}


def test_humid_stagnant_keeps_secondary_aerosol_uncertain():
    """Thiếu dữ liệu tiền chất → cơ chế sol khí thứ cấp không được lên tin cậy CAO."""
    _, bundle = build("humid_stagnant")
    secondary = [h for h in bundle.hypotheses if h.mechanism_id == "MECH_SECONDARY_AEROSOL"]
    if secondary:
        assert secondary[0].confidence is not Confidence.HIGH
        assert secondary[0].limitations


# =============================================================================
# INV-3 — fail-safe
# =============================================================================


def test_sparse_data_reports_limits_instead_of_guessing():
    _, bundle = build("sparse_data")
    assert bundle.missing, "dữ liệu thiếu phải được nêu, không được im lặng"
    assert any("Không kiểm tra được" in note for note in bundle.missing)


def test_no_evidence_gives_insufficient_confidence():
    """Ablation A: tắt hết → hệ thống phải nói không đủ căn cứ."""
    _, bundle = build("winter_inversion", ablation="A")
    assert bundle.overall_confidence is Confidence.INSUFFICIENT
    assert bundle.hypotheses == []


def test_disabled_layers_are_declared_in_bundle():
    _, bundle = build("winter_inversion", ablation="C")
    assert any("Knowledge Graph bị TẮT" in note for note in bundle.missing)
    assert bundle.config_flags["use_rules"] is False


# =============================================================================
# Ablation
# =============================================================================


@pytest.mark.parametrize("level", list("ABCDE"))
def test_every_ablation_level_runs(level):
    _, bundle = build("winter_inversion", ablation=level)
    answer = DryRunNarrator().narrate(bundle)
    assert answer.text.strip()


def test_evidence_grows_monotonically_with_ablation_level():
    """A→E: mỗi tầng bật thêm phải làm bằng chứng nhiều hơn, không ít đi."""
    counts = {}
    for level in "ABCDE":
        _, bundle = build("winter_inversion", ablation=level)
        counts[level] = (
            len(bundle.data_evidence) + len(bundle.shap_summary) + len(bundle.hypotheses)
        )
    assert counts["A"] < counts["B"] <= counts["C"] <= counts["D"] <= counts["E"]


# =============================================================================
# Consistency ở mức pipeline
# =============================================================================


def test_conflicting_model_produces_visible_warning():
    _, bundle = build("winter_inversion", xai_mode="conflicting")
    assert bundle.conflicts, "SHAP ngược chiều mà không có cảnh báo nào"
    answer = DryRunNarrator().narrate(bundle)
    assert "Điểm bất định" in answer.text


def test_spurious_model_flags_non_mechanistic_dominance():
    _, bundle = build("winter_inversion", xai_mode="spurious")
    assert bundle.non_mechanistic_share is not None
    assert bundle.non_mechanistic_share > 0.3
    assert any("CẢNH BÁO CƠ CHẾ" in c for c in bundle.conflicts)


# =============================================================================
# INV-1 — narrator chỉ được nhìn thấy bundle
# =============================================================================


def test_evidence_json_contains_only_bundle_content():
    _, bundle = build("winter_inversion")
    payload = build_evidence_json(bundle)

    assert set(payload) >= {
        "data_evidence",
        "model_attribution",
        "hypotheses",
        "conflicts",
        "missing",
        "overall_confidence",
    }
    ids = {h["mechanism_id"] for h in payload["hypotheses"]}
    assert ids == {h.mechanism_id for h in bundle.hypotheses}


def test_system_prompt_forbids_outside_knowledge():
    assert "KHÔNG CÓ KIẾN THỨC NỀN RIÊNG" in SYSTEM_PROMPT
    assert "Không đủ căn cứ" in SYSTEM_PROMPT


def test_dryrun_answer_labels_every_evidence_item():
    _, bundle = build("winter_inversion")
    text = DryRunNarrator().narrate(bundle).text
    for evidence in bundle.data_evidence:
        assert f"[{evidence.evidence_id}]" in text


def test_answer_states_when_no_citations_available():
    """RAG chưa có corpus → phải nói rõ, không được lặng lẽ bỏ mục trích dẫn."""
    _, bundle = build("winter_inversion")
    text = DryRunNarrator().narrate(bundle).text
    assert "Chưa có tài liệu trong kho hỗ trợ cơ chế này" in text


# =============================================================================
# Knowledge graph
# =============================================================================


def test_knowledge_graph_traces_variable_to_pollutant():
    pipeline, bundle = build("winter_inversion")
    from reasoning.kg import explain_path

    path = explain_path(pipeline.graph, "MECH_LOW_PBLH")
    relations = {relation for _, relation, _ in path}
    assert {"EVALUATED_BY", "TRIGGERS", "INCREASES"} <= relations


def test_mermaid_export_is_renderable():
    pipeline, bundle = build("winter_inversion")
    from reasoning.kg import to_mermaid

    diagram = to_mermaid(pipeline.graph, [h.mechanism_id for h in bundle.hypotheses])
    assert diagram.startswith("flowchart LR")
    assert "-->" in diagram
