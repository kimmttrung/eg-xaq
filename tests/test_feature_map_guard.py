"""Hàng rào cho `knowledge/feature_map.yaml`.

VÌ SAO CẦN RIÊNG MỘT FILE TEST CHO CHUYỆN NÀY
=============================================

Ánh xạ tên đặc trưng là chỗ nối duy nhất giữa attribution của mô hình lab và tầng
cơ chế. Khi nó hỏng, hệ thống KHÔNG ném lỗi — nó vẫn chấm điểm, vẫn xếp hạng, vẫn
sinh câu trả lời trông tự tin, chỉ có điều `shap_agreement` bằng 0 ở mọi cơ chế và
đóng góp nghiên cứu số 2 đã tắt lặng lẽ.

Các test dưới đây làm hai việc:

1. Khóa trạng thái ĐÚNG hiện tại, để lần cập nhật feature_map tiếp theo (khi nhận
   danh sách đặc trưng thật từ lab, mục M4 trong data contract) không âm thầm phá
   ánh xạ mà không ai biết.
2. Chứng minh chế độ hỏng CÓ THẬT và hàng rào bắt được nó — `test_renamed_features_
   silently_disable_consistency` cố ý tái hiện đúng thảm họa đó.
"""

from __future__ import annotations

import copy

import pytest

from data.mock import EPISODES, HANOI_LAT, HANOI_LON, MockObservationProvider
from schemas import Verdict
from xai.coverage import (
    FeatureMapCoverageError,
    analyze_coverage,
    assert_mapping_healthy,
    format_report,
    fully_blind_mechanisms,
    mechanism_blind_spots,
)
from xai.mock import MockAttributionProvider

#: Cơ chế được phép mù, kèm lý do. Danh sách này phải NGẮN và mỗi mục phải giải
#: thích được trước hội đồng.
#:
#: MECH_SOURCE_SECTOR_TRANSPORT chỉ kỳ vọng biến `wind_dir`, mà rule kích hoạt nó
#: (R12) lại cần `source_sector_alignment` — thứ đòi layer GIS nguồn thải hiện
#: CHƯA CÓ cho Hà Nội. Cơ chế này vì thế không kích hoạt trong bất kỳ tình huống
#: nào hôm nay, nên việc nó không kiểm chứng chéo được là hệ quả của dữ liệu
#: thiếu, không phải của ánh xạ hỏng. Khi có layer GIS thì phải bỏ khỏi danh sách
#: này và bổ sung đặc trưng hướng gió.
KNOWN_BLIND_MECHANISMS = ["MECH_SOURCE_SECTOR_TRANSPORT"]


@pytest.fixture
def model_feature_roster():
    """Toàn bộ tên đặc trưng mà mô hình CÓ THỂ sinh ra, gộp qua mọi kịch bản.

    Đây mới là thứ đúng để kiểm tra độ phủ. Attribution của MỘT lần dự báo chỉ
    chứa các đặc trưng có đóng góp khác 0 — ví dụ ngày không có điểm cháy thì
    `fire_count_upwind` vắng mặt, và nếu lấy đúng ngày đó đi đánh giá feature map
    thì sẽ kết luận nhầm là biến `fire` không được phủ.

    Với mô hình thật, danh sách tương ứng là `booster.feature_names` đọc từ
    checkpoint — xem scripts/check_feature_map.py.
    """
    names: set[str] = set()
    for episode, spec in EPISODES.items():
        obs = MockObservationProvider(episode).get(HANOI_LAT, HANOI_LON, spec["date"], 0)
        attribution = MockAttributionProvider("aligned").get(
            HANOI_LAT, HANOI_LON, obs.date, 0, observation=obs
        )
        names |= {c.feature for c in attribution.contributions}
    return sorted(names)


# =============================================================================
# Trạng thái đúng — khóa lại để không bị phá âm thầm
# =============================================================================


def test_mock_features_are_all_recognized(kb, model_feature_roster):
    """Mọi đặc trưng mô hình sinh ra phải được nhận ra, dù là cơ chế hay phi cơ chế.

    Tên đặc trưng trong mock cố ý bắt chước quy ước của lab (hậu tố `_2d`, `_3d`)
    để chính ánh xạ này được kiểm thử chứ không chỉ chạy cho có.
    """
    report = analyze_coverage(kb.feature_map, model_feature_roster)

    assert report.unmapped == (), (
        f"Đặc trưng chưa ánh xạ: {report.unmapped}. "
        f"Mỗi tên ở đây là một phần attribution mà consistency check bỏ qua."
    )
    assert report.recognized_ratio == 1.0


def test_only_documented_mechanisms_are_blind(kb, model_feature_roster):
    """Cơ chế mù = khóa cứng ở PARTIAL, không bao giờ phát hiện được CONFLICT.

    Chỉ chấp nhận đúng những cơ chế đã ghi lý do trong KNOWN_BLIND_MECHANISMS.
    Test này sẽ báo khi có cơ chế mới bị mù, VÀ khi một cơ chế hết mù mà danh sách
    chưa dọn — cả hai đều là lúc cần nhìn lại feature map.
    """
    report = analyze_coverage(kb.feature_map, model_feature_roster)

    assert fully_blind_mechanisms(kb, report) == KNOWN_BLIND_MECHANISMS


def test_non_mechanistic_features_are_classified_not_dropped(kb, model_feature_roster):
    """`lat`, `lon`, `doy_sin` phải vào nhóm phi cơ chế, KHÔNG phải nhóm chưa ánh xạ.

    Phân biệt này quan trọng: phi cơ chế là kết quả nghiên cứu (đo mức mô hình dựa
    vào mẫu không gian/thời gian), còn chưa ánh xạ là lỗi cấu hình. Gộp hai thứ lại
    sẽ làm `non_mechanistic_share` sai.
    """
    report = analyze_coverage(kb.feature_map, model_feature_roster)

    assert set(report.non_mechanistic) == {"lat", "lon", "doy_sin"}
    assert "lat" not in report.unmapped


def test_every_variable_expected_by_a_mechanism_is_reachable(kb, model_feature_roster):
    """Mỗi biến mà cơ chế kỳ vọng phải có ít nhất một đặc trưng trỏ tới.

    `kb._validate` đã bắt trường hợp biến không tồn tại trong feature_map. Test này
    chặt hơn một bậc: biến có tồn tại trong YAML, nhưng KHÔNG đặc trưng thật nào
    ánh xạ về nó — lúc đó cơ chế vẫn im lặng y hệt.
    """
    report = analyze_coverage(kb.feature_map, model_feature_roster)

    expected = {var for mech in kb.mechanisms.values() for var in mech.variables}
    unreachable = expected - report.covered_variables

    # Mock không sinh đặc trưng hướng gió nào. Khi nối model lab, nếu lab CÓ u10/v10
    # thì `wind_dir` sẽ tự được phủ và assert này phải siết lại thành rỗng.
    assert unreachable == {"wind_dir"}, f"Biến không có đặc trưng nào trỏ tới: {unreachable}"


def test_ratio_check_passes_for_the_mock_roster(kb, model_feature_roster):
    """Phần kiểm tra tỉ lệ phải sạch; chỉ có cơ chế mù đã ghi nhận là còn lại."""
    report = assert_mapping_healthy(kb, model_feature_roster, allow_fully_blind=True)

    assert report.total == len(model_feature_roster)
    assert report.mechanistic_ratio == pytest.approx(9 / 12)


# =============================================================================
# Chế độ hỏng — chứng minh hàng rào bắt được
# =============================================================================


def test_guard_rejects_unknown_lab_feature_names(kb):
    """Lab đặt tên khác mà feature_map chưa cập nhật → phải NÉM LỖI, không im lặng."""
    lab_names = [
        "HPBL_surface_min_48h",
        "UGRD_10m_mean_72h",
        "TMP_2m_max",
        "APCP_surface_sum_72h",
        "RH_2m_mean",
    ]

    with pytest.raises(FeatureMapCoverageError) as exc:
        assert_mapping_healthy(kb, lab_names)

    assert "CHƯA" in str(exc.value).upper() or "nhận ra" in str(exc.value)


def test_renamed_features_silently_disable_consistency(kb, engine, observation_of, attribution_of):
    """ĐÂY LÀ THẢM HỌA MÀ HÀNG RÀO TỒN TẠI ĐỂ NGĂN.

    Đổi tên đặc trưng → không ánh xạ được → shap_agreement = 0 ở mọi cơ chế →
    mọi verdict rơi về PARTIAL → KHÔNG CÒN CONFLICT NÀO ĐƯỢC PHÁT HIỆN.

    Và không có một exception nào. Test này khóa hành vi đó lại thành tài liệu
    sống: nếu ai đó "sửa" cho nó im lặng theo cách khác, test sẽ báo.
    """
    obs = observation_of("winter_inversion")
    attribution = copy.deepcopy(attribution_of("winter_inversion", mode="conflicting"))

    # Bình thường: chế độ conflicting phải sinh ra CONFLICT.
    normal = engine.reason(obs, copy.deepcopy(attribution))
    assert any(h.verdict is Verdict.CONFLICT for h in normal.hypotheses)
    assert normal.conflicts, "chế độ conflicting phải sinh cảnh báo"

    # Đổi tên toàn bộ đặc trưng sang quy ước lạ.
    for contribution in attribution.contributions:
        contribution.feature = f"X_{contribution.feature.upper()}_v3"

    broken = engine.reason(obs, attribution)

    assert all(h.shap_agreement == 0.0 for h in broken.hypotheses)
    assert not any(h.verdict is Verdict.CONFLICT for h in broken.hypotheses)
    assert broken.conflicts == [], "mâu thuẫn biến mất hoàn toàn — đúng như lo ngại"

    # Và hàng rào phải bắt được đúng tình huống này.
    with pytest.raises(FeatureMapCoverageError):
        assert_mapping_healthy(kb, [c.feature for c in attribution.contributions])


def test_guard_detects_a_single_blinded_variable(kb, model_feature_roster):
    """Bỏ hết đặc trưng PBLH → cơ chế dựa vào PBLH phải bị báo là mù.

    Tình huống thực tế: lab đặt tên PBLH theo kiểu ta chưa lường. Tỉ lệ phủ tổng
    thể vẫn cao (các biến khác vẫn ánh xạ tốt), nên chỉ nhìn tỉ lệ thì KHÔNG phát
    hiện ra — phải soi theo từng biến, đó là lý do `mechanism_blind_spots` tồn tại.
    """
    names = [n for n in model_feature_roster if kb.feature_map.resolve(n) != "blh"]

    report = analyze_coverage(kb.feature_map, names)
    blind = mechanism_blind_spots(kb, report)

    assert "blh" not in report.covered_variables
    assert blind["MECH_LOW_PBLH"] == ["blh"]
    assert "MECH_LOW_PBLH" in fully_blind_mechanisms(kb, report)

    # Tỉ lệ nhận ra vẫn 100% — bằng chứng cho thấy chỉ đo tỉ lệ là không đủ.
    assert report.recognized_ratio == 1.0

    with pytest.raises(FeatureMapCoverageError, match="mù hoàn toàn"):
        assert_mapping_healthy(kb, names)


def test_blind_mechanisms_can_be_allowed_but_only_explicitly(kb, model_feature_roster):
    """Có cờ bỏ qua, nhưng phải bật TƯỜNG MINH — mặc định luôn là chặn."""
    names = [n for n in model_feature_roster if kb.feature_map.resolve(n) != "blh"]

    with pytest.raises(FeatureMapCoverageError):
        assert_mapping_healthy(kb, names)

    assert_mapping_healthy(kb, names, allow_fully_blind=True)


# =============================================================================
# Báo cáo cho người đọc
# =============================================================================


def test_report_names_the_unmapped_features(kb):
    """Tên lạ phải được gọi đích danh trong báo cáo, không chỉ đếm số."""
    report = analyze_coverage(
        kb.feature_map, ["blh_min_2d", "lat", "UGRD_10m_mean_72h", "sea_ice_frac"]
    )
    text = format_report(kb, report)

    assert set(report.unmapped) == {"UGRD_10m_mean_72h", "sea_ice_frac"}
    assert "UGRD_10m_mean_72h" in text
    assert "sea_ice_frac" in text
    assert "⚠ CHƯA ÁNH XẠ" in text
    assert report.mechanistic_ratio == pytest.approx(1 / 4)
    assert report.recognized_ratio == pytest.approx(2 / 4)


def test_gfs_style_names_that_already_resolve(kb):
    """Một số quy ước GFS đã được prefix trong feature_map bắt sẵn.

    Ghi lại thành test để lần cập nhật feature_map sau không vô tình bỏ các prefix
    này đi — chúng là thứ giúp ánh xạ sống sót khi lab đổi hậu tố cửa sổ tích lũy.
    """
    assert kb.feature_map.resolve("HPBL_surface_min_48h") == "blh"
    assert kb.feature_map.resolve("APCP_surface_sum_72h") == "precip"
    assert kb.feature_map.resolve("RH_2m_mean") == "humidity"
