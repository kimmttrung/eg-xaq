"""Cấu hình toàn hệ thống, đọc từ biến môi trường / file .env.

Nguyên tắc: KHÔNG có giá trị bí mật nào được hardcode. Mọi thứ đọc từ .env
(xem .env.example). Mặc định luôn là chế độ chạy được offline, không cần API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Cấu hình runtime. Prefix `EGXAQ_` cho các biến riêng của dự án."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # ---------- nguồn dữ liệu ----------
    data_backend: Literal["mock", "lab", "era5"] = Field("mock", alias="EGXAQ_DATA_BACKEND")
    xai_backend: Literal["mock", "lab"] = Field("mock", alias="EGXAQ_XAI_BACKEND")
    lab_model_dir: str | None = Field(None, alias="EGXAQ_LAB_MODEL_DIR")
    lab_raster_dir: str | None = Field(None, alias="EGXAQ_LAB_RASTER_DIR")

    # ---------- ERA5 (phương án dự phòng khi chưa có GFS của lab) ----------
    era5_daily_csv: str | None = Field(None, alias="EGXAQ_ERA5_DAILY_CSV")
    era5_supplement_csv: str | None = Field(None, alias="EGXAQ_ERA5_SUPPLEMENT_CSV")
    firms_csv: str | None = Field(None, alias="EGXAQ_FIRMS_CSV")
    pm25_obs_csv: str | None = Field(None, alias="EGXAQ_PM25_OBS_CSV")

    # ---------- narrator ----------
    narrator_backend: Literal["dryrun", "anthropic"] = Field(
        "dryrun", alias="EGXAQ_NARRATOR_BACKEND"
    )
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    llm_model: str = Field("claude-opus-5", alias="EGXAQ_LLM_MODEL")

    # ---------- vector DB ----------
    qdrant_url: str = Field("http://localhost:6333", alias="EGXAQ_QDRANT_URL")
    qdrant_api_key: str | None = Field(None, alias="EGXAQ_QDRANT_API_KEY")
    qdrant_collection: str = Field("egxaq_papers", alias="EGXAQ_QDRANT_COLLECTION")

    # ---------- embedding ----------
    # hash        = băm offline, chỉ để test
    # hf          = sentence-transformers tại chỗ (cần torch, ~2GB)
    # precomputed = tra vector truy vấn đã tính sẵn trên GPU ngoài (Kaggle/Colab).
    #               Tài liệu đã được embed và nạp Qdrant từ phía GPU; máy local
    #               không cần torch. Xem notebooks/kaggle_index_corpus.py
    embedding_backend: Literal["hash", "hf", "precomputed"] = Field(
        "hash", alias="EGXAQ_EMBEDDING_BACKEND"
    )
    query_vectors_file: str | None = Field(None, alias="EGXAQ_QUERY_VECTORS")
    embedding_model: str = Field("BAAI/bge-m3", alias="EGXAQ_EMBEDDING_MODEL")
    embedding_dim: int = Field(1024, alias="EGXAQ_EMBEDDING_DIM")
    reranker_model: str = Field("BAAI/bge-reranker-v2-m3", alias="EGXAQ_RERANKER_MODEL")
    use_reranker: bool = Field(False, alias="EGXAQ_USE_RERANKER")

    # ---------- corpus ----------
    openalex_mailto: str | None = Field(None, alias="OPENALEX_MAILTO")
    semantic_scholar_api_key: str | None = Field(None, alias="SEMANTIC_SCHOLAR_API_KEY")
    firms_map_key: str | None = Field(None, alias="FIRMS_MAP_KEY")

    # ---------- đường dẫn ----------
    @property
    def rules_path(self) -> Path:
        return KNOWLEDGE_DIR / "rules.yaml"

    @property
    def mechanisms_path(self) -> Path:
        return KNOWLEDGE_DIR / "mechanisms.yaml"

    @property
    def feature_map_path(self) -> Path:
        return KNOWLEDGE_DIR / "feature_map.yaml"

    @property
    def corpus_dir(self) -> Path:
        return DATA_DIR / "corpus"

    @property
    def query_vectors_path(self) -> Path:
        """Bảng vector truy vấn tính sẵn (backend `precomputed`)."""
        if self.query_vectors_file:
            return Path(self.query_vectors_file)
        return DATA_DIR / "corpus" / "query_vectors.json"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton — tránh đọc lại .env ở mỗi lần gọi."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Dùng trong test khi cần đổi biến môi trường giữa chừng."""
    global _settings
    _settings = None
