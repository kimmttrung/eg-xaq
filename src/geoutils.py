"""Tiện ích hình học địa lý dùng chung cho tầng dữ liệu và tầng suy luận.

Đặt ở đây (không trong `data/`) vì `reasoning/` cũng cần, mà `reasoning/` không
được phép phụ thuộc vào `data/`.
"""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Khoảng cách vòng lớn giữa hai điểm, tính bằng km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Phương vị từ điểm 1 tới điểm 2, độ, 0 = Bắc, tăng theo chiều kim đồng hồ."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlam)
    return math.degrees(math.atan2(y, x)) % 360.0


def angular_diff_deg(a: float, b: float) -> float:
    """Chênh lệch góc nhỏ nhất giữa hai phương vị, trong [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def uv_to_speed_dir(u: float, v: float) -> tuple[float, float]:
    """Thành phần gió (u, v) → (tốc độ m/s, hướng gió THỔI TỚI TỪ, độ).

    ⚠ Đây là chỗ hay sai nhất trong toàn bộ pipeline (xem CLAUDE.md §8 bẫy #3).
    Quy ước khí tượng: hướng gió là hướng gió ĐẾN TỪ, không phải hướng gió thổi đi.
    Gió Đông Bắc (45°) nghĩa là gió thổi TỪ hướng Đông Bắc VỀ phía Tây Nam.
    """
    speed = math.hypot(u, v)
    direction = (270.0 - math.degrees(math.atan2(v, u))) % 360.0
    return speed, direction


def is_upwind(
    receptor_lat: float,
    receptor_lon: float,
    source_lat: float,
    source_lon: float,
    wind_dir_deg: float,
    sector_deg: float = 90.0,
    max_distance_km: float = 300.0,
) -> tuple[bool, float]:
    """Nguồn có nằm trong quạt thượng nguồn gió của điểm nhận không?

    `wind_dir_deg` là hướng gió ĐẾN TỪ, nên phương vị từ receptor tới nguồn phải
    nằm trong ±sector_deg/2 quanh chính giá trị đó.

    Trả về (có_thượng_nguồn, khoảng_cách_km).
    """
    dist = haversine_km(receptor_lat, receptor_lon, source_lat, source_lon)
    if dist > max_distance_km:
        return False, dist
    brg = bearing_deg(receptor_lat, receptor_lon, source_lat, source_lon)
    return angular_diff_deg(brg, wind_dir_deg) <= sector_deg / 2.0, dist


def compass_label_vi(deg: float | None) -> str:
    """Hướng gió dạng chữ tiếng Việt, để narrator diễn đạt tự nhiên."""
    if deg is None:
        return "không rõ"
    labels = [
        "Bắc",
        "Bắc Đông Bắc",
        "Đông Bắc",
        "Đông Đông Bắc",
        "Đông",
        "Đông Đông Nam",
        "Đông Nam",
        "Nam Đông Nam",
        "Nam",
        "Nam Tây Nam",
        "Tây Nam",
        "Tây Tây Nam",
        "Tây",
        "Tây Tây Bắc",
        "Tây Bắc",
        "Bắc Tây Bắc",
    ]
    return labels[int((deg % 360.0) / 22.5 + 0.5) % 16]
