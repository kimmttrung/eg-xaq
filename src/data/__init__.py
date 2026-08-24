"""Tầng dữ liệu — mọi truy cập ra thế giới bên ngoài đi qua đây.

Hôm nay: `MockObservationProvider` (dữ liệu tổng hợp có kiểm soát).
Khi có data lab: thêm `LabRasterObservationProvider`, không sửa gì ở `reasoning/`.
"""

from .base import ObservationProvider, get_observation_provider
from .mock import EPISODES, MockObservationProvider

__all__ = [
    "EPISODES",
    "MockObservationProvider",
    "ObservationProvider",
    "get_observation_provider",
]
