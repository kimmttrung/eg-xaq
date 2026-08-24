"""Tầng attribution — SHAP trên mô hình dự báo của lab.

INV-2: kết quả ở đây giải thích MÔ HÌNH, không giải thích THỰC TẠI. Không bao giờ
đưa thẳng ra người dùng như "nguyên nhân ô nhiễm".
"""

from .base import AttributionProvider, get_attribution_provider
from .mock import MockAttributionProvider

__all__ = [
    "AttributionProvider",
    "MockAttributionProvider",
    "get_attribution_provider",
]
