"""Interface tầng attribution — "phích cắm" cho 10 checkpoint XGBoost của lab.

Ghi chú thiết kế cho khi nối vào model thật (docs/02-data-contract.md §2):

- Mô hình lab có **10 checkpoint riêng cho 10 bước** t → t+9. Phải chọn đúng
  checkpoint theo `step`, và đọc `feature_names` TỪ CHECKPOINT, không hardcode.
- `shap.TreeExplainer` với XGBoost chạy rất nhanh (thuật toán TreeSHAP), nên
  giải thích từng pixel là khả thi; nhưng giải thích cả bản đồ thì không —
  chỉ giải thích tại điểm người dùng hỏi.
- Nếu target được biến đổi (log1p), giá trị SHAP nằm trên thang đã biến đổi.
  Phải ghi rõ điều đó vào `Attribution.provenance` để narrator không nói sai
  đơn vị µg/m³.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Protocol, runtime_checkable

from config import get_settings
from schemas import Attribution, Observation


@runtime_checkable
class AttributionProvider(Protocol):
    """Trả về đóng góp SHAP của từng đặc trưng cho dự báo tại (điểm, thời điểm, step)."""

    name: str

    def get(
        self,
        lat: float,
        lon: float,
        date: Date,
        step: int = 0,
        observation: Observation | None = None,
    ) -> Attribution: ...


class AttributionNotAvailable(RuntimeError):
    """Backend attribution chưa dùng được."""


def get_attribution_provider(backend: str | None = None) -> AttributionProvider:
    """Chọn provider theo cấu hình `EGXAQ_XAI_BACKEND`."""
    backend = backend or get_settings().xai_backend

    if backend == "mock":
        from .mock import MockAttributionProvider

        return MockAttributionProvider()

    if backend == "lab":
        try:
            from .lab_xgb import LabXGBAttributionProvider
        except ImportError as exc:  # pragma: no cover - cần checkpoint lab
            raise AttributionNotAvailable(
                "Backend 'lab' cần requirements-lab.txt và 10 checkpoint XGBoost. "
                "Xem docs/02-data-contract.md §2."
            ) from exc
        return LabXGBAttributionProvider()

    raise AttributionNotAvailable(f"Backend attribution không hợp lệ: {backend!r}")
