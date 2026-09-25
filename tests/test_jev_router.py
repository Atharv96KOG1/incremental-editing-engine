"""Offline tests for retrieval/jev_router.py -- Jev is a real, optional,
network-dependent third-party model, so every test here mocks
typesafe_sdk's client (never a live call) and confirms the "not
configured / low confidence / any failure" cases all degrade to None,
which run_pipeline.py's own dispatch treats as "don't shortcut."
"""

import pytest

from incremental_editing.retrieval import jev_router


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    jev_router.get_settings.cache_clear()
    yield
    jev_router.get_settings.cache_clear()


def test_classify_request_kind_none_when_not_configured(monkeypatch):
    """No TYPESAFE_API_KEY set -- the client is never even constructed,
    same as VectorRetriever's own opt-in-only design."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert jev_router.classify_request_kind("rename foo to bar") is None


def test_classify_request_kind_returns_kind_above_confidence_threshold(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    class _FakeAnswer:
        choice = "question"
        confidence = 0.92

    class _FakeResponse:
        answers = {"kind": _FakeAnswer()}

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def system_one(self, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", _FakeClient)
    assert jev_router.classify_request_kind("what does this file do") == "question"


def test_classify_request_kind_none_below_confidence_threshold(monkeypatch):
    """A genuinely unsure answer must not be trusted -- falls through to
    the existing pipeline exactly like an unconfigured/failed call."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    class _FakeAnswer:
        choice = "whole_file"
        confidence = 0.4

    class _FakeResponse:
        answers = {"kind": _FakeAnswer()}

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def system_one(self, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", _FakeClient)
    assert jev_router.classify_request_kind("do something ambiguous") is None


def test_classify_request_kind_none_on_api_failure(monkeypatch):
    """A broken gateway/timeout/rate limit must degrade to None, not
    raise -- the same "signal unavailable" contract every other
    retrieval signal in this project already holds itself to."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    from typesafe_sdk import TypeSafeError

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def system_one(self, **kwargs):
            raise TypeSafeError("boom")

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", _FakeClient)
    assert jev_router.classify_request_kind("rename foo to bar") is None


def test_classify_request_kind_respects_configured_threshold(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("JEV_CONFIDENCE_THRESHOLD", "0.99")

    class _FakeAnswer:
        choice = "question"
        confidence = 0.9  # would pass the default 0.8 bar, not this stricter one

    class _FakeResponse:
        answers = {"kind": _FakeAnswer()}

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def system_one(self, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", _FakeClient)
    assert jev_router.classify_request_kind("what does this do") is None
