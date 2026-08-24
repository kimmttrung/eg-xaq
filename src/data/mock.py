"""Dữ liệu quan sát tổng hợp — thay thế tạm cho data lab.

⚠ Đây KHÔNG phải dữ liệu thật. Mục đích duy nhất là để lõi suy luận, RAG và
narrator chạy được và có test trước khi nhận data từ lab (docs/02 §8).
Mọi con số trong khóa luận PHẢI đến từ backend `lab`.

Các kịch bản dưới đây dựng theo những dạng episode điển hình của Hà Nội, với giá
trị lấy ở khoảng hợp lý về mặt vật lý để rule kích hoạt đúng như mong đợi. Chúng
đóng vai trò "test fixture có ý nghĩa domain": nếu sửa ngưỡng trong rules.yaml mà
làm hỏng các kịch bản này thì test sẽ báo ngay.
"""

from __future__ import annotations

from datetime import date as Date

from schemas import Observation

HANOI_LAT, HANOI_LON = 21.03, 105.85


#: Kịch bản → các trường của Observation.
#: Khóa `_note` chỉ để đọc, bị loại trước khi dựng Observation.
EPISODES: dict[str, dict] = {
    # -------------------------------------------------------------------------
    "winter_inversion": {
        "_note": "Đợt ô nhiễm mùa đông kinh điển: nghịch nhiệt + PBLH thấp + gió lặng.",
        "date": Date(2024, 1, 15),
        "pm25_pred_ugm3": 95.2,
        "blh_m": 320.0,
        "wind_speed_ms": 0.8,
        "wind_dir_deg": 25.0,
        "t2m_c": 16.4,
        "t850_c": 18.1,  # t850 > t2m → nghịch nhiệt
        "rh_pct": 78.0,
        "precip_mm": 0.0,
        "mslp_hpa": 1024.0,
        "wind_speed_mean_2d_ms": 1.1,
        "wind_speed_mean_3d_ms": 1.2,
        "blh_min_2d_m": 290.0,
        "blh_min_3d_m": 305.0,
        "precip_sum_3d_mm": 0.0,
        "upwind_fire_count": 0,
        "blh_climatology_m": 820.0,
        "wind_speed_climatology_ms": 2.4,
    },
    # -------------------------------------------------------------------------
    "biomass_burning": {
        "_note": "Đốt rơm rạ sau thu hoạch ở đồng bằng sông Hồng, gió Đông Nam mang khói về.",
        "date": Date(2024, 4, 8),
        "pm25_pred_ugm3": 88.0,
        "blh_m": 780.0,
        "wind_speed_ms": 2.2,
        "wind_dir_deg": 135.0,
        "t2m_c": 27.5,
        "t850_c": 18.5,
        "rh_pct": 72.0,
        "precip_mm": 0.0,
        "mslp_hpa": 1010.0,
        "wind_speed_mean_2d_ms": 2.4,
        "wind_speed_mean_3d_ms": 2.6,
        "blh_min_2d_m": 700.0,
        "precip_sum_3d_mm": 0.2,
        "upwind_fire_count": 42,
        "upwind_fire_distance_km": 65.0,
        "blh_climatology_m": 900.0,
        "wind_speed_climatology_ms": 2.6,
    },
    # -------------------------------------------------------------------------
    "cold_surge_clean": {
        "_note": "Không khí lạnh tràn về: gió mạnh thổi sạch. Dùng để test cơ chế LOẠI BỎ.",
        "date": Date(2024, 12, 3),
        "pm25_pred_ugm3": 17.5,
        "blh_m": 1350.0,
        "wind_speed_ms": 6.4,
        "wind_dir_deg": 40.0,
        "t2m_c": 14.2,
        "t850_c": 6.8,
        "rh_pct": 55.0,
        "precip_mm": 0.0,
        "mslp_hpa": 1026.0,
        "wind_speed_mean_2d_ms": 5.1,
        "wind_speed_mean_3d_ms": 4.2,
        "blh_min_2d_m": 1100.0,
        "precip_sum_3d_mm": 0.0,
        "upwind_fire_count": 3,
        "blh_climatology_m": 820.0,
        "wind_speed_climatology_ms": 2.4,
    },
    # -------------------------------------------------------------------------
    "rain_washout": {
        "_note": "Mưa rào rửa trôi. Test cơ chế wet deposition.",
        "date": Date(2024, 5, 20),
        "pm25_pred_ugm3": 21.0,
        "blh_m": 610.0,
        "wind_speed_ms": 3.1,
        "wind_dir_deg": 160.0,
        "t2m_c": 28.0,
        "t850_c": 19.0,
        "rh_pct": 94.0,
        "precip_mm": 18.5,
        "mslp_hpa": 1004.0,
        "wind_speed_mean_2d_ms": 2.9,
        "wind_speed_mean_3d_ms": 2.7,
        "blh_min_2d_m": 520.0,
        "precip_sum_3d_mm": 34.0,
        "upwind_fire_count": 0,
        "blh_climatology_m": 900.0,
        "wind_speed_climatology_ms": 2.6,
    },
    # -------------------------------------------------------------------------
    "humid_stagnant": {
        "_note": "Nồm ẩm + tù đọng: nghi sol khí thứ cấp, nhưng thiếu dữ liệu tiền chất.",
        "date": Date(2024, 3, 5),
        "pm25_pred_ugm3": 76.0,
        "blh_m": 430.0,
        "wind_speed_ms": 1.1,
        "wind_dir_deg": 110.0,
        "t2m_c": 21.0,
        "t850_c": 15.5,
        "rh_pct": 93.0,
        "precip_mm": 0.3,
        "mslp_hpa": 1016.0,
        "wind_speed_mean_2d_ms": 1.3,
        "wind_speed_mean_3d_ms": 1.4,
        "blh_min_2d_m": 400.0,
        "precip_sum_3d_mm": 0.6,
        "upwind_fire_count": 6,
        "upwind_fire_distance_km": 190.0,
        "blh_climatology_m": 880.0,
        "wind_speed_climatology_ms": 2.5,
    },
    # -------------------------------------------------------------------------
    "sparse_data": {
        "_note": (
            "Thiếu nhiều biến — dùng để kiểm tra INV-3 fail-safe: hệ thống phải nói "
            "'không đủ căn cứ' chứ không được đoán bù."
        ),
        "date": Date(2024, 2, 10),
        "pm25_pred_ugm3": 64.0,
        "blh_m": 480.0,
        "wind_speed_ms": None,
        "wind_dir_deg": None,
        "t2m_c": 18.0,
        "t850_c": None,
        "rh_pct": None,
        "precip_mm": None,
        "mslp_hpa": None,
        "upwind_fire_count": None,
        "blh_climatology_m": 820.0,
    },
}


class MockObservationProvider:
    """Provider trả dữ liệu kịch bản. Cài đặt `ObservationProvider`."""

    name = "mock"

    def __init__(self, episode: str = "winter_inversion") -> None:
        if episode not in EPISODES:
            raise KeyError(f"Kịch bản không tồn tại: {episode!r}. Có: {sorted(EPISODES)}")
        self.episode = episode

    def get(self, lat: float, lon: float, date: Date, step: int = 0) -> Observation:
        payload = {k: v for k, v in EPISODES[self.episode].items() if not k.startswith("_")}
        payload.pop("date", None)
        return Observation(
            lat=lat,
            lon=lon,
            date=date,
            step=step,
            provenance=f"mock:{self.episode}",
            **payload,
        )

    @staticmethod
    def default_date(episode: str) -> Date:
        """Ngày "hợp mùa" của kịch bản — dùng khi người dùng không nêu ngày cụ thể."""
        return EPISODES[episode]["date"]

    @staticmethod
    def describe(episode: str) -> str:
        return EPISODES[episode].get("_note", "")
