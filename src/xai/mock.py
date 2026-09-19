"""SHAP giả lập — thay thế tạm cho 10 checkpoint XGBoost của lab.

⚠ KHÔNG phải kết quả SHAP thật. Mục đích: (a) để consistency check có đầu vào mà
chạy và có test; (b) dựng sẵn các tình huống CONFLICT / MODEL_ONLY để chứng minh
cơ chế phát hiện hoạt động — những tình huống mà data thật có thể hiếm khi cho ta
đúng lúc cần demo.

Tên đặc trưng ở đây cố ý bắt chước quy ước của lab (có hậu tố accum `_2d`, `_3d`)
để kiểm tra luôn `knowledge/feature_map.yaml` có ánh xạ đúng không.
"""

from __future__ import annotations

import random
from datetime import date as Date
from typing import Literal

from schemas import Attribution, FeatureContribution, Observation

Mode = Literal["aligned", "conflicting", "spurious"]


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class MockAttributionProvider:
    """Sinh attribution suy ra một cách xác định từ quan sát.

    Ba chế độ:

    - ``aligned``     — SHAP đồng thuận với cơ chế vật lý. Đây là "mô hình tốt".
    - ``conflicting`` — đảo dấu các biến cơ chế → tạo verdict CONFLICT. Mô phỏng
                        mô hình học phải tương quan giả (INV-2).
    - ``spurious``    — toạ độ và mã ngày trong năm chiếm phần lớn attribution →
                        tạo cảnh báo "mô hình ghi nhớ mẫu không gian/thời gian".
    """

    name = "mock"

    def __init__(self, mode: Mode = "aligned", seed: int = 42, noise: float = 0.05) -> None:
        self.mode = mode
        self.seed = seed
        self.noise = noise

    def get(
        self,
        lat: float,
        lon: float,
        date: Date,
        step: int = 0,
        observation: Observation | None = None,
    ) -> Attribution:
        if observation is None:
            raise ValueError(
                "MockAttributionProvider cần `observation` để suy ra attribution. "
                "Provider thật sẽ đọc trực tiếp từ raster đặc trưng."
            )

        rng = random.Random(f"{self.seed}:{date}:{step}:{lat}:{lon}")
        obs = observation

        # --- Đóng góp cơ chế: dấu suy ra từ vật lý, độ lớn từ mức vượt ngưỡng ---
        signals: list[tuple[str, float | None, float]] = [
            # (tên đặc trưng, giá trị, đóng góp thô ∈ [-1, 1])
            ("blh_min_2d", obs.blh_min_2d_m or obs.blh_m, self._blh_signal(obs)),
            ("blh_mean", obs.blh_m, 0.6 * self._blh_signal(obs)),
            (
                "wind_speed_mean_2d",
                obs.wind_speed_mean_2d_ms or obs.wind_speed_ms,
                self._wind_signal(obs),
            ),
            ("wind_speed_max_3d", obs.wind_speed_mean_3d_ms, 0.5 * self._wind_signal(obs)),
            ("precip_sum_3d", obs.precip_sum_3d_mm, self._precip_signal(obs)),
            ("rh_mean", obs.rh_pct, self._rh_signal(obs)),
            ("mslp_mean", obs.mslp_hpa, self._pressure_signal(obs)),
            ("t2m_mean", obs.t2m_c, self._temperature_signal(obs)),
            ("fire_count_upwind", obs.upwind_fire_count, self._fire_signal(obs)),
        ]

        if self.mode == "conflicting":
            signals = [(name, value, -raw) for name, value, raw in signals]

        # --- Đặc trưng KHÔNG mang cơ chế (biến tĩnh theo pixel, mã thời gian) ---
        spurious_scale = 1.8 if self.mode == "spurious" else 0.15
        signals += [
            ("lat", lat, spurious_scale * 0.9),
            ("lon", lon, spurious_scale * 0.6),
            ("doy_sin", float(date.timetuple().tm_yday), spurious_scale * 0.8),
        ]

        # --- Quy về thang µg/m³ để narrator diễn đạt được ---
        base_value = 45.0
        prediction = obs.pm25_pred_ugm3 if obs.pm25_pred_ugm3 is not None else base_value
        gap = prediction - base_value
        total_raw = sum(abs(raw) for _, _, raw in signals) or 1.0
        scale = abs(gap) / total_raw if gap else 1.0

        contributions = [
            FeatureContribution(
                feature=name,
                value=float(value) if value is not None else None,
                shap=round(raw * scale * (1.0 + rng.uniform(-self.noise, self.noise)), 3),
            )
            for name, value, raw in signals
            if abs(raw) > 1e-6
        ]

        return Attribution(
            step=step,
            model_id=f"mock_step{step}",
            base_value=base_value,
            prediction=prediction,
            contributions=contributions,
            provenance=f"mock:{self.mode}",
        )

    # ------------------------------------------------------------------ tín hiệu
    # Dấu DƯƠNG = đẩy PM2.5 lên. Các công thức dưới đây mã hóa đúng chiều vật lý
    # mà `knowledge/mechanisms.yaml` kỳ vọng ở trường `expected_shap`.

    @staticmethod
    def _blh_signal(obs: Observation) -> float:
        """PBLH càng thấp so với 800 m thì càng đẩy PM2.5 lên."""
        blh = obs.blh_min_2d_m if obs.blh_min_2d_m is not None else obs.blh_m
        return 0.0 if blh is None else _clip((800.0 - blh) / 600.0)

    @staticmethod
    def _wind_signal(obs: Observation) -> float:
        """Gió càng yếu so với 2.5 m/s thì càng đẩy PM2.5 lên."""
        wind = (
            obs.wind_speed_mean_2d_ms
            if obs.wind_speed_mean_2d_ms is not None
            else obs.wind_speed_ms
        )
        return 0.0 if wind is None else _clip((2.5 - wind) / 2.5)

    @staticmethod
    def _precip_signal(obs: Observation) -> float:
        """Không mưa → đẩy lên (vắng rửa trôi); mưa nhiều → kéo xuống. Đổi dấu ở ~5 mm."""
        precip = obs.precip_sum_3d_mm if obs.precip_sum_3d_mm is not None else obs.precip_mm
        return 0.0 if precip is None else _clip((5.0 - precip) / 5.0)

    @staticmethod
    def _rh_signal(obs: Observation) -> float:
        """RH trên ~75% bắt đầu đẩy lên (hút ẩm + phản ứng pha lỏng)."""
        return 0.0 if obs.rh_pct is None else _clip((obs.rh_pct - 75.0) / 20.0)

    @staticmethod
    def _pressure_signal(obs: Observation) -> float:
        """Cao áp → chìm lún → đẩy lên."""
        return 0.0 if obs.mslp_hpa is None else _clip((obs.mslp_hpa - 1013.0) / 15.0)

    @staticmethod
    def _temperature_signal(obs: Observation) -> float:
        """Nghịch nhiệt (t850 > t2m) → đẩy lên. Không có t850 → tín hiệu yếu."""
        if obs.t2m_c is None:
            return 0.0
        if obs.t850_c is None:
            return _clip((20.0 - obs.t2m_c) / 20.0) * 0.3
        lapse = (obs.t2m_c - obs.t850_c) / 1.5
        return _clip((4.0 - lapse) / 6.0)

    @staticmethod
    def _fire_signal(obs: Observation) -> float:
        return 0.0 if not obs.upwind_fire_count else _clip(obs.upwind_fire_count / 50.0)
