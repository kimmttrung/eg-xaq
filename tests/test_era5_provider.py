"""ERA5DailyObservationProvider — nguồn khí tượng thật cho bộ episode đánh giá.

Trọng tâm là INV-3 ở tầng dữ liệu: "chưa kiểm tra" phải ra `None`, không bao giờ ra 0
hay một giá trị tính trên nửa cửa sổ. Lỗi kiểu này không làm test nào khác đỏ — nó chỉ
lặng lẽ biến "không có dữ liệu" thành "không có điểm cháy".
"""

from __future__ import annotations

from datetime import date as Date

import pytest

from data.base import ObservationUnavailable
from data.era5 import ERA5DailyObservationProvider
from reasoning.derive import derive_features, feature_context

HEADER = (
    "date,blh_min,blh_mean,wind_speed_mean,wind_speed_min,rh_mean,t2m_mean,rain_sum,wind_dir_mean\n"
)
ROWS = [
    "2023-01-08,50,400,1.0,0.2,80,18,0.0,45\n",
    "2023-01-09,40,300,2.0,0.3,82,17,0.5,40\n",
    "2023-01-10,30,200,3.0,0.4,85,16,1.5,35\n",
]
HANOI = (21.03, 105.85)
DAY = Date(2023, 1, 10)


def _features(tmp_path, rows=ROWS):
    path = tmp_path / "features_daily.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def _get(provider):
    return provider.get(*HANOI, DAY)


def test_same_day_values_pass_through_with_units(tmp_path):
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path)))

    assert (obs.blh_m, obs.wind_speed_ms, obs.wind_dir_deg) == (200, 3.0, 35)
    assert (obs.t2m_c, obs.rh_pct, obs.precip_mm) == (16, 85, 1.5)
    assert obs.pm25_pred_ugm3 is None, "ERA5 không phải mô hình dự báo"
    assert obs.upwind_fire_count is None, "chưa khai FIRMS → chưa kiểm tra, không phải 0"


def test_multi_day_windows(tmp_path):
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path)))

    assert obs.wind_speed_mean_2d_ms == pytest.approx(2.5)
    assert obs.wind_speed_mean_3d_ms == pytest.approx(2.0)
    assert obs.blh_min_2d_m == 200
    assert obs.precip_sum_3d_mm == pytest.approx(2.0)


def test_gap_in_window_gives_none_not_partial_value(tmp_path):
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path, [ROWS[0], ROWS[2]])))

    assert obs.wind_speed_mean_2d_ms is None
    assert obs.precip_sum_3d_mm is None
    assert obs.wind_speed_ms == 3.0


def test_missing_supplement_is_reported_and_leaves_inversion_unchecked(tmp_path):
    provider = ERA5DailyObservationProvider(
        _features(tmp_path), supplement_csv=tmp_path / "chua_tai.csv"
    )
    obs = _get(provider)

    assert obs.t850_c is None and obs.mslp_hpa is None
    assert derive_features(obs).lapse_rate_c_per_km is None
    assert any("R3" in note for note in provider.missing_sources)


def test_supplement_enables_inversion_check(tmp_path):
    supplement = tmp_path / "supplement.csv"
    supplement.write_text(
        "date,t850_mean_c,t850_00utc_c,mslp_mean_hpa\n2023-01-10,18.5,19.0,1024.0\n",
        encoding="utf-8",
    )
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path), supplement_csv=supplement))

    assert (obs.t850_c, obs.mslp_hpa) == (18.5, 1024.0)
    assert derive_features(obs).lapse_rate_c_per_km < 0, "t850 > t2m phải là nghịch nhiệt"


def _firms(tmp_path, rows: str, dates: str):
    path = tmp_path / "firms.csv"
    path.write_text("latitude,longitude,acq_date\n" + rows, encoding="utf-8")
    (tmp_path / "firms.csv.dates").write_text(dates, encoding="utf-8")
    return path


def test_unqueried_fire_day_is_none_not_zero(tmp_path):
    firms = _firms(tmp_path, "", "2023-01-10\n")  # thiếu hôm trước 2023-01-09
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path), firms_csv=firms))
    assert obs.upwind_fire_count is None


def test_queried_day_without_fires_is_zero(tmp_path):
    firms = _firms(tmp_path, "", "2023-01-09\n2023-01-10\n")
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path), firms_csv=firms))
    assert obs.upwind_fire_count == 0


def test_only_upwind_fires_are_counted(tmp_path):
    # Gió từ 35° (Đông Bắc). Điểm (21.9, 106.6) nằm hướng ~39°, cách ~120 km → thượng nguồn.
    # Điểm (20.3, 105.2) nằm hướng Tây Nam → hạ nguồn, không tính.
    firms = _firms(
        tmp_path,
        "21.9,106.6,2023-01-10\n20.3,105.2,2023-01-09\n",
        "2023-01-09\n2023-01-10\n",
    )
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path), firms_csv=firms))

    assert obs.upwind_fire_count == 1
    assert obs.upwind_fire_distance_km == pytest.approx(124, abs=15)


def test_firms_without_dates_file_is_rejected(tmp_path):
    path = tmp_path / "firms.csv"
    path.write_text("latitude,longitude,acq_date\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ngày đã tra"):
        ERA5DailyObservationProvider(_features(tmp_path), firms_csv=path)


def test_rejects_other_places_forecast_steps_and_unknown_days(tmp_path):
    provider = ERA5DailyObservationProvider(_features(tmp_path))
    with pytest.raises(ObservationUnavailable):
        provider.get(10.8, 106.7, DAY)  # TP.HCM
    with pytest.raises(ObservationUnavailable):
        provider.get(*HANOI, DAY, step=1)
    with pytest.raises(ObservationUnavailable):
        provider.get(*HANOI, Date(2030, 1, 1))


def test_observed_pm25_drives_aqi_when_there_is_no_forecast(tmp_path):
    pm25 = tmp_path / "pm25.csv"
    pm25.write_text("date,pm25\n2023-01-10,95.0\n", encoding="utf-8")
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path), pm25_csv=pm25))
    derived = derive_features(obs)

    assert obs.pm25_obs_ugm3 == 95.0
    assert derived.aqi_category == "Xấu"
    labels = [item.label for item in feature_context(obs, derived)]
    assert "PM2.5 quan trắc" in labels and "PM2.5 dự báo" not in labels


def test_climatology_is_monthly_mean(tmp_path):
    obs = _get(ERA5DailyObservationProvider(_features(tmp_path)))
    assert obs.blh_climatology_m == pytest.approx(300.0)
    assert derive_features(obs).blh_anomaly_pct == pytest.approx(-100 / 3)


def test_real_era5_series_loads(tmp_path):
    """Chuỗi thật ở ../era5 — bỏ qua nếu máy không có thư mục đó."""
    from config import PROJECT_ROOT

    real = PROJECT_ROOT.parent / "era5" / "features_daily.csv"
    if not real.exists():
        pytest.skip("không có ../era5/features_daily.csv")
    obs = ERA5DailyObservationProvider(real).get(*HANOI, Date(2023, 1, 10))
    assert obs.blh_m is not None and 0 < obs.blh_m < 4000
    assert obs.t2m_c is not None and -5 < obs.t2m_c < 45, "t2m phải là °C, không phải Kelvin"
