"""Bộ đánh giá: schema nhãn, chấm điểm, chọn ngày ứng viên.

Các test ở đây bảo vệ con số chính của khóa luận. Một lỗi trong cách đếm tp/fp/fn,
hay một nhãn sai tên cơ chế bị bỏ qua lặng lẽ, sẽ đổi Cause F1 mà không ai biết.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pytest

from evaluation import (
    DailyRecord,
    EpisodeLabel,
    EpisodeLabelError,
    EpisodeScore,
    aggregate,
    load_episodes,
    render_worksheet,
    score_bundle,
    select_candidates,
    validate_against_kb,
)
from pipeline import ExplanationPipeline, PipelineConfig
from xai.mock import MockAttributionProvider

SMOKE = Path(__file__).resolve().parents[1] / "data" / "eval" / "smoke_episodes.yaml"


def _label(**overrides) -> dict:
    base = {
        "episode_id": "hn-2023-01-10",
        "date": Date(2023, 1, 10),
        "group": "polluted_winter",
        "expected_outcome": "cause",
        "questions": ["Vì sao hôm nay Hà Nội ô nhiễm?"],
        "primary_mechanisms": ["MECH_LOW_PBLH"],
        "rationale": "Bản tin nêu lớp xáo trộn thấp kéo dài.",
        "sources": [{"type": "cem_report", "citation": "Bản tin CEM 01/2023"}],
        "annotator": "A",
        "label_confidence": "high",
    }
    base.update(overrides)
    return base


# =============================================================================
# Schema nhãn
# =============================================================================


def test_valid_label_parses():
    assert EpisodeLabel.model_validate(_label()).episode_id == "hn-2023-01-10"


def test_same_mechanism_cannot_be_primary_and_excluded():
    with pytest.raises(ValueError, match="trùng nhau"):
        EpisodeLabel.model_validate(_label(excluded_mechanisms=["MECH_LOW_PBLH"]))


def test_cause_without_primary_is_rejected():
    with pytest.raises(ValueError, match="phải có primary_mechanisms"):
        EpisodeLabel.model_validate(_label(primary_mechanisms=[]))


def test_insufficient_cannot_name_a_cause():
    with pytest.raises(ValueError, match="insufficient"):
        EpisodeLabel.model_validate(_label(expected_outcome="insufficient"))


def test_real_episode_cannot_run_on_mock_data():
    """Dữ liệu giả cho một ngày thật cho ra số trông y hệt số thật — phải chặn ở schema."""
    with pytest.raises(ValueError, match="không được dùng mock_episode"):
        EpisodeLabel.model_validate(_label(mock_episode="winter_inversion"))


def test_design_source_only_for_synthetic_episodes():
    with pytest.raises(ValueError, match="synthetic_design"):
        EpisodeLabel.model_validate(
            _label(sources=[{"type": "synthetic_design", "citation": "mock.py"}])
        )


def test_label_needs_a_source():
    with pytest.raises(ValueError):
        EpisodeLabel.model_validate(_label(sources=[]))


def test_typo_in_field_name_is_rejected():
    """`primary_mechanism` (thiếu s) mà bị bỏ qua → episode không có nguyên nhân nào."""
    with pytest.raises(ValueError):
        EpisodeLabel.model_validate(_label(primary_mechanism=["MECH_LOW_PBLH"]))


def test_unknown_mechanism_is_reported(kb):
    label = EpisodeLabel.model_validate(_label(primary_mechanisms=["MECH_KHONG_CO"]))
    assert any("không tồn tại" in p for p in validate_against_kb(label, kb))


def test_clean_day_must_be_explained_by_removal(kb):
    label = EpisodeLabel.model_validate(_label(group="clean", expected_outcome="clean"))
    assert any("LÀM GIẢM" in p for p in validate_against_kb(label, kb))


# =============================================================================
# Nạp file
# =============================================================================


def _write(tmp_path, text: str) -> Path:
    path = tmp_path / "episodes.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loader_counts_todo_and_rejected(tmp_path, kb):
    path = _write(
        tmp_path,
        """
episodes:
  - episode_id: a
    status: labeled
    date: 2023-01-10
    group: polluted_winter
    expected_outcome: cause
    questions: ["q"]
    primary_mechanisms: [MECH_LOW_PBLH]
    rationale: "nguồn nói lớp xáo trộn thấp"
    sources: [{type: cem_report, citation: "Bản tin"}]
    annotator: A
    label_confidence: high
  - episode_id: b
    status: todo
    expected_outcome:
  - episode_id: c
    status: rejected
    reject_reason: "không tìm được nguồn"
""",
    )
    episodes = load_episodes(path, kb)
    assert [e.episode_id for e in episodes.labeled] == ["a"]
    assert (episodes.todo, episodes.rejected) == (1, 1)


def test_rejected_episode_needs_a_reason(tmp_path, kb):
    path = _write(tmp_path, "episodes:\n  - episode_id: c\n    status: rejected\n")
    with pytest.raises(EpisodeLabelError, match="reject_reason"):
        load_episodes(path, kb)


def test_loader_reports_every_error_at_once(tmp_path, kb):
    """Fail fast nhưng gom đủ lỗi — sửa một lỗi rồi mới thấy lỗi tiếp là mất thời gian."""
    path = _write(
        tmp_path,
        """
episodes:
  - {episode_id: x, status: labeled, date: 2023-01-10, group: polluted_winter,
     expected_outcome: cause, questions: [q], primary_mechanisms: [MECH_SAI_1],
     rationale: "abcdefghijk", sources: [{type: cem_report, citation: abc}],
     annotator: A, label_confidence: high}
  - {episode_id: y, status: labeled, date: 2023-01-11, group: polluted_winter,
     expected_outcome: cause, questions: [q], primary_mechanisms: [MECH_SAI_2],
     rationale: "abcdefghijk", sources: [{type: cem_report, citation: abc}],
     annotator: A, label_confidence: high}
""",
    )
    with pytest.raises(EpisodeLabelError) as exc:
        load_episodes(path, kb)
    assert "MECH_SAI_1" in str(exc.value) and "MECH_SAI_2" in str(exc.value)


def test_smoke_fixture_matches_knowledge_base(kb):
    episodes = load_episodes(SMOKE, kb)
    assert len(episodes.labeled) == 6
    assert all(e.synthetic for e in episodes.labeled)


# =============================================================================
# Chấm điểm
# =============================================================================


def _score(predicted, primary, contributing=(), excluded=(), outcome="cause") -> EpisodeScore:
    return EpisodeScore(
        episode_id="e",
        config="E",
        predicted=tuple(predicted),
        predicted_outcome=outcome,
        expected_outcome="cause",
        primary=frozenset(primary),
        contributing=frozenset(contributing),
        excluded=frozenset(excluded),
        top_confidence=None,
        overall_confidence="THẤP",
    )


def test_strict_counts_contributing_as_false_positive():
    score = _score(["A", "B", "C"], primary={"A", "D"}, contributing={"B"}, excluded={"C"})
    assert score.counts(lenient=False) == (1, 2, 1)


def test_lenient_does_not_penalise_contributing():
    score = _score(["A", "B", "C"], primary={"A", "D"}, contributing={"B"}, excluded={"C"})
    assert score.counts(lenient=True) == (1, 1, 1)


def test_asserting_an_excluded_mechanism_is_counted():
    score = _score(["A", "C"], primary={"A"}, excluded={"C"})
    assert score.excluded_hits == ("C",)
    assert score.top1_hit is True


def test_saying_nothing_misses_top1():
    assert _score([], primary={"A"}, outcome="insufficient").top1_hit is False


def test_aggregate_uses_micro_average():
    """Micro: cộng tp/fp/fn rồi mới tính — ổn định hơn với 30–50 episode."""
    summary = aggregate(
        [
            _score(["A"], primary={"A"}),  # 1/0/0
            _score(["X", "Y", "Z"], primary={"B"}),  # 0/3/1
        ]
    )
    assert summary["strict_precision"] == pytest.approx(1 / 4)
    assert summary["strict_recall"] == pytest.approx(1 / 2)
    assert summary["top1_accuracy"] == pytest.approx(1 / 2)


def _run_smoke(kb, episode_id: str, level: str):
    label = next(e for e in load_episodes(SMOKE, kb).labeled if e.episode_id == episode_id)
    from data.mock import MockObservationProvider

    pipeline = ExplanationPipeline(
        observation_provider=MockObservationProvider(label.mock_episode),
        attribution_provider=MockAttributionProvider(),
        kb=kb,
        config=PipelineConfig.ablation(level),
    )
    bundle = pipeline.build_bundle(label.questions[0], date=label.date)
    return score_bundle(label, bundle, level)


def test_full_system_finds_the_designed_causes(kb):
    score = _run_smoke(kb, "smoke-winter-inversion", "E")
    assert score.counts(lenient=True) == (3, 0, 0)
    assert score.excluded_hits == ()
    assert score.top1_hit is True
    assert score.outcome_correct


def test_clean_day_is_explained_as_clean(kb):
    score = _run_smoke(kb, "smoke-cold-surge-clean", "E")
    assert score.predicted_outcome == "clean"
    assert "MECH_ADVECTION_CLEANSING" in score.predicted


def test_ablation_without_rules_asserts_nothing(kb):
    score = _run_smoke(kb, "smoke-winter-inversion", "A")
    assert score.predicted == ()
    assert score.predicted_outcome == "insufficient"
    assert score.counts(lenient=False) == (0, 0, 3)


# =============================================================================
# Chọn ngày ứng viên
# =============================================================================


def _two_years() -> list[DailyRecord]:
    start = Date(2022, 1, 1)
    return [
        DailyRecord(start + timedelta(days=i), float((i * 37) % 150 + 5))  # xác định, có dao động
        for i in range(730)
    ]


def test_candidates_are_spread_out_and_in_the_right_season():
    candidates = select_candidates(_two_years(), per_group=10, min_gap_days=5)
    dates = sorted(c.date for c in candidates)
    assert all((b - a).days >= 5 for a, b in zip(dates, dates[1:], strict=False))
    assert all(c.date.month in {11, 12, 1, 2} for c in candidates if c.group == "polluted_winter")
    assert all(
        c.date.month in {3, 4, 9, 10} for c in candidates if c.group == "polluted_transition"
    )


def test_selection_is_deterministic():
    assert select_candidates(_two_years()) == select_candidates(_two_years())


def test_worksheet_loads_as_all_todo(tmp_path, kb):
    """Phiếu vừa sinh phải nạp được ngay — nếu không, người gán nhãn bắt đầu bằng lỗi."""
    candidates = select_candidates(_two_years(), per_group=4)
    path = _write(tmp_path, render_worksheet(candidates, kb, "test"))
    episodes = load_episodes(path, kb)
    assert episodes.labeled == []
    assert episodes.todo == len(candidates)


def test_worksheet_does_not_leak_meteorology(kb):
    """Phiếu không được in PBLH/gió/mưa — người gán nhãn sẽ tái tạo lại rule."""
    text = render_worksheet(select_candidates(_two_years(), per_group=2), kb, "test")
    body = text.split("episodes:", 1)[1].lower()
    assert "blh" not in body and "wind" not in body and "rain" not in body


# =============================================================================
# Người gán nhãn thứ hai và Cohen's κ
# =============================================================================


def test_kappa_extremes():
    from evaluation import cohens_kappa

    assert cohens_kappa([("a", "a"), ("b", "b")]) == pytest.approx(1.0)
    assert cohens_kappa([("a", "b"), ("b", "a")]) == pytest.approx(-1.0)
    assert cohens_kappa([("a", "a"), ("a", "a")]) is None  # p_e = 1: không xác định
    assert cohens_kappa([]) is None


def test_agreement_reports_role_disagreements(tmp_path, kb):
    from evaluation import label_agreement

    entry = """
  - episode_id: e1
    date: 2023-01-10
    group: polluted_winter
    expected_outcome: cause
    questions: [q]
    primary_mechanisms: [{primary}]
    rationale: "nguồn nêu nguyên nhân rõ ràng"
    sources: [{{type: cem_report, citation: "Bản tin"}}]
    annotator: X
    label_confidence: high
"""
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text("episodes:" + entry.format(primary="MECH_LOW_PBLH"), encoding="utf-8")
    second.write_text("episodes:" + entry.format(primary="MECH_STAGNATION"), encoding="utf-8")

    result = label_agreement(load_episodes(first, kb), load_episodes(second, kb), kb)

    assert result.common == 1
    assert result.outcome_kappa is None  # cả hai cùng "cause" → κ không xác định
    assert any("MECH_LOW_PBLH" in line for line in result.disagreements)
    assert result.mechanism_kappa is not None and result.mechanism_kappa < 1.0


def test_second_annotator_sample_is_stratified_and_reproducible():
    from evaluation import sample_for_second_annotator

    candidates = select_candidates(_two_years(), per_group=10)
    picked = sample_for_second_annotator(candidates, n=9, seed=7)

    assert picked == sample_for_second_annotator(candidates, n=9, seed=7)
    assert len(picked) == 9
    assert {c.group for c in picked} == {c.group for c in candidates}


def test_second_worksheet_demands_independence(kb):
    text = render_worksheet(
        select_candidates(_two_years(), per_group=2), kb, "test", second_annotator=True
    )
    assert "ĐỘC LẬP" in text
