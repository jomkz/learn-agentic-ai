"""Tests for prompt_engineering.py — module-level constants."""

from __future__ import annotations

from prompt_engineering import TRANSCRIPT, V1_USER, V2_SYSTEM, V3_SYSTEM, V3_USER


def test_transcript_is_non_empty() -> None:
    assert isinstance(TRANSCRIPT, str)
    assert len(TRANSCRIPT.strip()) > 0


def test_v1_user_contains_transcript() -> None:
    assert "action items" in V1_USER.lower()


def test_v2_system_mentions_owner() -> None:
    assert "owner" in V2_SYSTEM.lower()


def test_v3_user_has_examples() -> None:
    assert "Example" in V3_USER


def test_v3_system_same_as_v2() -> None:
    assert V3_SYSTEM is V2_SYSTEM
