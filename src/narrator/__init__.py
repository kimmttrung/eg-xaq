"""Tầng diễn đạt — LLM chỉ được nói lại những gì có trong EvidenceBundle (INV-1)."""

from .narrator import AnthropicNarrator, DryRunNarrator, Narrator, get_narrator
from .prompts import SYSTEM_PROMPT, build_evidence_json, build_user_message

__all__ = [
    "SYSTEM_PROMPT",
    "AnthropicNarrator",
    "DryRunNarrator",
    "Narrator",
    "build_evidence_json",
    "build_user_message",
    "get_narrator",
]
