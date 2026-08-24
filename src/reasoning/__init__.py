"""Lõi suy luận: rule → cơ chế → giả thuyết → điểm số → độ tin cậy.

Package này KHÔNG được import xgboost, shap, rasterio, qdrant_client hay anthropic.
Nó chỉ nhận Pydantic schema và trả về Pydantic schema — nhờ vậy toàn bộ phần
"ăn điểm" của khóa luận test được mà không cần data lab.
"""

from .consistency import assess_consistency
from .derive import derive_features, feature_context
from .engine import ReasoningEngine
from .hypotheses import build_hypotheses
from .kg import build_knowledge_graph, explain_path
from .rules import RuleEngine
from .scoring import ScoringWeights, score_hypotheses

__all__ = [
    "ReasoningEngine",
    "RuleEngine",
    "ScoringWeights",
    "assess_consistency",
    "build_hypotheses",
    "build_knowledge_graph",
    "derive_features",
    "explain_path",
    "feature_context",
    "score_hypotheses",
]
