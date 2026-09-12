"""Tầng attribution — SHAP trên mô hình dự báo của lab.

INV-2: kết quả ở đây giải thích MÔ HÌNH, không giải thích THỰC TẠI. Không bao giờ
đưa thẳng ra người dùng như "nguyên nhân ô nhiễm".
"""

from .base import AttributionProvider, get_attribution_provider
from .coverage import (
    CoverageReport,
    FeatureMapCoverageError,
    analyze_coverage,
    assert_mapping_healthy,
    format_report,
    fully_blind_mechanisms,
    mechanism_blind_spots,
)
from .mock import MockAttributionProvider

__all__ = [
    "AttributionProvider",
    "CoverageReport",
    "FeatureMapCoverageError",
    "MockAttributionProvider",
    "analyze_coverage",
    "assert_mapping_healthy",
    "format_report",
    "fully_blind_mechanisms",
    "get_attribution_provider",
    "mechanism_blind_spots",
]
