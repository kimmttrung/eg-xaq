"""Interface của tầng dữ liệu — "phích cắm" cho data của lab.

Xem docs/02-data-contract.md §7.1 để biết chính xác từng trường phải chứa gì.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Protocol, runtime_checkable

from config import get_settings
from schemas import Observation


@runtime_checkable
class ObservationProvider(Protocol):
    """Trả về điều kiện khí quyển tại một điểm / một thời điểm / một bước dự báo.

    Cài đặt PHẢI tuân thủ:
    - Đơn vị đúng như tên trường (`_c` là Celsius, `_mm` là milimét, `_ms` là m/s).
    - Dữ liệu không có → để `None`. TUYỆT ĐỐI không điền giá trị mặc định (INV-3).
    - Đặt `provenance` để truy vết được nguồn của mỗi câu trả lời.
    """

    name: str

    def get(self, lat: float, lon: float, date: Date, step: int = 0) -> Observation: ...


class ProviderNotAvailable(RuntimeError):
    """Backend được yêu cầu nhưng chưa cài đặt được (thiếu data, thiếu thư viện)."""


def get_observation_provider(backend: str | None = None) -> ObservationProvider:
    """Chọn provider theo cấu hình `EGXAQ_DATA_BACKEND`."""
    backend = backend or get_settings().data_backend

    if backend == "mock":
        from .mock import MockObservationProvider

        return MockObservationProvider()

    if backend == "lab":
        try:
            from .lab_raster import LabRasterObservationProvider
        except ImportError as exc:  # pragma: no cover - cần data lab mới chạy tới
            raise ProviderNotAvailable(
                "Backend 'lab' cần requirements-lab.txt và checkpoint từ lab. "
                "Xem docs/02-data-contract.md §8."
            ) from exc
        return LabRasterObservationProvider()

    raise ProviderNotAvailable(f"Backend dữ liệu không hợp lệ: {backend!r}")
