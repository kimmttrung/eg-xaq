"""Fixture dùng chung.

Toàn bộ test chạy offline: không mạng, không Docker, không API key, không data lab.
Đó là ràng buộc có chủ đích — test nào cần hạ tầng ngoài thì test đó sẽ không được
chạy, và phần lõi của khóa luận sẽ mất lưới an toàn.
"""

from __future__ import annotations

from datetime import date as Date

import pytest

from data.mock import HANOI_LAT, HANOI_LON, MockObservationProvider
from kb import get_knowledge_base
from reasoning.engine import ReasoningEngine
from xai.mock import MockAttributionProvider


@pytest.fixture(scope="session")
def kb():
    return get_knowledge_base()


@pytest.fixture
def engine(kb):
    return ReasoningEngine(kb)


@pytest.fixture
def observation_of():
    """`observation_of("winter_inversion")` → Observation của kịch bản đó."""

    def _make(episode: str, step: int = 0):
        provider = MockObservationProvider(episode=episode)
        date = MockObservationProvider.default_date(episode)
        return provider.get(HANOI_LAT, HANOI_LON, date, step)

    return _make


@pytest.fixture
def attribution_of(observation_of):
    def _make(episode: str, mode: str = "aligned", step: int = 0):
        obs = observation_of(episode, step)
        provider = MockAttributionProvider(mode=mode)
        return provider.get(HANOI_LAT, HANOI_LON, obs.date, step, observation=obs)

    return _make


@pytest.fixture
def hanoi():
    return HANOI_LAT, HANOI_LON


@pytest.fixture
def some_date():
    return Date(2024, 1, 15)
