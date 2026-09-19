"""Chỉ số dẫn xuất — biến quan trắc thô thành đại lượng mà rule tham chiếu được.

Tách riêng khỏi rule engine vì đây là công thức VẬT LÝ (ổn định), còn ngưỡng là
tri thức CÓ THỂ HIỆU CHỈNH (nằm trong YAML).
"""

from __future__ import annotations

from schemas import DataEvidence, DerivedFeatures, Observation

# Bảng AQI Việt Nam cho PM2.5 trung bình 24h (QĐ 1459/QĐ-TCMT).
# ⚠ Phải xác nhận web PopGIS dùng thang này hay thang US EPA
#    (docs/02-data-contract.md W2) — hai thang lệch nhau đáng kể.
_AQI_VN_PM25: list[tuple[float, float, int, int, str]] = [
    (0.0, 25.0, 0, 50, "Tốt"),
    (25.0, 50.0, 51, 100, "Trung bình"),
    (50.0, 80.0, 101, 150, "Kém"),
    (80.0, 150.0, 151, 200, "Xấu"),
    (150.0, 250.0, 201, 300, "Rất xấu"),
    (250.0, 500.0, 301, 500, "Nguy hại"),
]


def pm25_to_aqi_vn(pm25: float | None) -> tuple[int | None, str | None]:
    """PM2.5 (µg/m³) → (AQI Việt Nam, nhãn chất lượng)."""
    if pm25 is None:
        return None, None
    if pm25 >= 500.0:
        return 500, "Nguy hại"
    for c_lo, c_hi, i_lo, i_hi, label in _AQI_VN_PM25:
        if c_lo <= pm25 < c_hi:
            aqi = (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
            return round(aqi), label
    return None, None


def _lapse_rate(obs: Observation, layer_km: float) -> float | None:
    """Gradient nhiệt theo độ cao, °C/km. Âm = nghịch nhiệt.

    Bình thường ~6.5 °C/km. Càng gần 0 hoặc âm thì tầng khí càng bền vững.
    """
    if obs.t2m_c is None or obs.t850_c is None or layer_km <= 0:
        return None
    return (obs.t2m_c - obs.t850_c) / layer_km


def _ventilation_index(obs: Observation) -> float | None:
    """PBLH × tốc độ gió, m²/s — khả năng phát tán tổng hợp của khí quyển.

    Cố ý dùng giá trị CÙNG NGÀY, không dùng cửa sổ tích lũy. Hai lý do:

    1. Ventilation index theo định nghĩa là đại lượng tức thời/theo ngày. Ghép
       PBLH cực tiểu 2 ngày (một thống kê ban đêm, rất thấp) với gió trung bình 2
       ngày sẽ hạ thấp chỉ số một cách hệ thống, khiến ngày trong lành cũng bị xếp
       là "kém thông thoáng".
    2. Khía cạnh cộng dồn nhiều ngày đã có `stagnation_days` (R10) và cơ chế
       MECH_MULTIDAY_ACCUMULATION lo. Nhồi nó vào đây là đếm hai lần.

    Chỉ lùi về giá trị tích lũy khi không có giá trị cùng ngày.
    """
    blh = obs.blh_m if obs.blh_m is not None else obs.blh_min_2d_m
    wind = obs.wind_speed_ms if obs.wind_speed_ms is not None else obs.wind_speed_mean_2d_ms
    if blh is None or wind is None:
        return None
    return blh * wind


def _blh_anomaly_pct(obs: Observation) -> float | None:
    """Độ lệch PBLH so với khí hậu tham chiếu, %. Âm = thấp hơn bình thường.

    Anomaly có sức thuyết phục hơn giá trị tuyệt đối: "PBLH 320 m" không nói lên
    gì với người đọc, "thấp hơn bình thường 60%" thì có.
    """
    if obs.blh_m is None or not obs.blh_climatology_m:
        return None
    return (obs.blh_m - obs.blh_climatology_m) / obs.blh_climatology_m * 100.0


def _stagnation_days(obs: Observation) -> float | None:
    """Ước lượng số ngày tù đọng liên tiếp từ các biến accum 2/3 ngày.

    Đây là XẤP XỈ, không phải đếm thật: ta chỉ có giá trị trung bình cửa sổ, không
    có chuỗi ngày. Nếu gió trung bình 3 ngày đã dưới ngưỡng thì gần như chắc chắn
    cả 3 ngày đều tù đọng. Khi có dữ liệu chuỗi đầy đủ từ lab, thay bằng phép đếm
    thật (docs/02-data-contract.md F3).

    ⚠ Hệ quả của phép xấp xỉ: giá trị trả về chỉ nhận được 0/1/2/3, nên R10
    (threshold 1.0, saturation 4.0) có trần thực tế strength = 0.667 và không bao
    giờ bão hòa. Đừng diễn giải điểm số của MECH_MULTIDAY_ACCUMULATION như thể nó
    được chấm trên thang đầy đủ.

    KHÔNG có biến gió nào → `None`, tuyệt đối không phải 0.0. INV-3: "không biết"
    khác "biết là không". Trả 0.0 ở đây sẽ sinh ra một mẩu bằng chứng
    ("Số ngày tù đọng: 0 ngày") tính từ chỗ không hề có dữ liệu.
    """
    threshold = 1.5  # m/s, khớp ngưỡng R2
    winds = (obs.wind_speed_mean_3d_ms, obs.wind_speed_mean_2d_ms, obs.wind_speed_ms)
    if all(wind is None for wind in winds):
        return None

    if obs.wind_speed_mean_3d_ms is not None and obs.wind_speed_mean_3d_ms < threshold:
        return 3.0
    if obs.wind_speed_mean_2d_ms is not None and obs.wind_speed_mean_2d_ms < threshold:
        return 2.0
    if obs.wind_speed_ms is not None and obs.wind_speed_ms < threshold:
        return 1.0
    return 0.0


def derive_features(obs: Observation, inversion_layer_km: float = 1.5) -> DerivedFeatures:
    """Tính toàn bộ chỉ số dẫn xuất. Thiếu đầu vào → trường tương ứng là None."""
    # Giải thích một ngày đã qua (không có dự báo) thì dùng PM2.5 quan trắc.
    pm25 = obs.pm25_pred_ugm3 if obs.pm25_pred_ugm3 is not None else obs.pm25_obs_ugm3
    aqi, category = pm25_to_aqi_vn(pm25)
    return DerivedFeatures(
        lapse_rate_c_per_km=_lapse_rate(obs, inversion_layer_km),
        ventilation_index_m2s=_ventilation_index(obs),
        blh_anomaly_pct=_blh_anomaly_pct(obs),
        stagnation_days=_stagnation_days(obs),
        aqi_vn=aqi,
        aqi_category=category,
    )


def variable_lookup(obs: Observation, derived: DerivedFeatures) -> dict[str, float | None]:
    """Bảng tra phẳng `tên_biến -> giá trị` cho rule engine.

    Rule trong YAML tham chiếu tên biến ở đây. Thêm biến mới → thêm vào dict này
    rồi mới dùng được trong rules.yaml.
    """
    table: dict[str, float | None] = {}
    for name, value in obs.model_dump().items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            table[name] = float(value)
        elif value is None:
            table[name] = None
    for name, value in derived.model_dump().items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            table[name] = float(value)
        elif value is None and name not in table:
            table[name] = None
    return table


# =============================================================================
# Bằng chứng dữ liệu để narrator trích dẫn
# =============================================================================

_EVIDENCE_SPECS: list[tuple[str, str, str]] = [
    # (tên trường, nhãn tiếng Việt, đơn vị)
    ("pm25_pred_ugm3", "PM2.5 dự báo", "µg/m³"),
    ("pm25_obs_ugm3", "PM2.5 quan trắc", "µg/m³"),
    ("blh_m", "Chiều cao lớp xáo trộn (PBLH)", "m"),
    ("wind_speed_ms", "Tốc độ gió 10 m", "m/s"),
    ("wind_dir_deg", "Hướng gió", "°"),
    ("t2m_c", "Nhiệt độ 2 m", "°C"),
    ("rh_pct", "Độ ẩm tương đối", "%"),
    ("precip_mm", "Lượng mưa", "mm"),
    ("mslp_hpa", "Áp suất mực biển", "hPa"),
    ("precip_sum_3d_mm", "Tổng mưa 3 ngày", "mm"),
    ("wind_speed_mean_3d_ms", "Gió trung bình 3 ngày", "m/s"),
    ("upwind_fire_count", "Điểm cháy thượng nguồn gió", "điểm"),
]


def feature_context(obs: Observation, derived: DerivedFeatures) -> list[DataEvidence]:
    """Đóng gói quan sát thành các mẩu bằng chứng có nhãn D1, D2, ...

    Chỉ đưa vào những trường CÓ giá trị. Trường thiếu sẽ được pipeline ghi vào
    `bundle.missing` thay vì đưa vào đây với giá trị bịa.
    """
    items: list[DataEvidence] = []
    data = obs.model_dump()

    for field, label, unit in _EVIDENCE_SPECS:
        value = data.get(field)
        if value is None:
            continue
        items.append(
            DataEvidence(
                evidence_id=f"D{len(items) + 1}",
                label=label,
                value=float(value),
                unit=unit,
                context=_context_for(field, obs, derived),
            )
        )

    for field, label, unit in [
        ("ventilation_index_m2s", "Chỉ số thông gió (PBLH × gió)", "m²/s"),
        ("lapse_rate_c_per_km", "Gradient nhiệt theo độ cao", "°C/km"),
        ("stagnation_days", "Số ngày tù đọng liên tiếp", "ngày"),
    ]:
        value = getattr(derived, field)
        if value is None:
            continue
        items.append(
            DataEvidence(
                evidence_id=f"D{len(items) + 1}",
                label=label,
                value=float(value),
                unit=unit,
                context=_context_for(field, obs, derived),
            )
        )
    return items


def _context_for(field: str, obs: Observation, derived: DerivedFeatures) -> str | None:
    """Câu ngắn giúp người đọc hiểu con số là cao hay thấp."""
    if field == "blh_m" and derived.blh_anomaly_pct is not None:
        sign = "thấp hơn" if derived.blh_anomaly_pct < 0 else "cao hơn"
        return f"{sign} trung bình khí hậu {abs(derived.blh_anomaly_pct):.0f}%"
    if field == "pm25_obs_ugm3" and obs.pm25_pred_ugm3 is None and derived.aqi_vn is not None:
        return f"AQI Việt Nam ≈ {derived.aqi_vn} ({derived.aqi_category})"
    if field == "pm25_pred_ugm3" and derived.aqi_vn is not None:
        return f"AQI Việt Nam ≈ {derived.aqi_vn} ({derived.aqi_category})"
    if field == "lapse_rate_c_per_km":
        return "khí quyển bình thường ≈ 6.5 °C/km; ≤0 là nghịch nhiệt"
    if field == "ventilation_index_m2s":
        return "dưới 4000 m²/s là kém thông thoáng, trên 6000 là tốt"
    if field == "wind_dir_deg" and obs.wind_dir_deg is not None:
        from geoutils import compass_label_vi

        return f"gió từ hướng {compass_label_vi(obs.wind_dir_deg)}"
    return None
