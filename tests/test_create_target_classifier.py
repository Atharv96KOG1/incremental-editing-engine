"""Offline tests for strategies/create_target_classifier.py -- a real,
optional structured-output model call, so every test here mocks
openai.OpenAI directly (never a live call)."""

import json

import pytest

from incremental_editing.strategies.create_target_classifier import classify_create_intent


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from incremental_editing.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 20
    total_tokens = 120
    prompt_tokens_details = None


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, payload):
        self.choices = [_FakeChoice(json.dumps(payload))]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, payload):
        self._payload = payload

    def create(self, **kwargs):
        return _FakeResponse(self._payload)


class _FakeChat:
    def __init__(self, payload):
        self.completions = _FakeCompletions(payload)


class _FakeClient:
    def __init__(self, payload):
        self.chat = _FakeChat(payload)


def _mock_openai(monkeypatch, payload):
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: _FakeClient(payload))


def test_classify_create_intent_returns_a_single_file_path(monkeypatch):
    _mock_openai(monkeypatch, {"wants_new_file": True, "paths": ["utils.py"]})
    result = classify_create_intent("make a new file called utils.py", "")
    assert result["wants_new_file"] is True
    assert result["paths"] == ["utils.py"]
    assert result["total_tokens"] == 120


def test_classify_create_intent_returns_multiple_paths_across_folders(monkeypatch):
    """Real motivating case: "frontend and backend in separate folders"
    must come back as real file paths under both, never merged into one
    file and never bare, empty folder names."""
    paths = ["frontend/index.html", "frontend/app.js", "backend/server.py"]
    _mock_openai(monkeypatch, {"wants_new_file": True, "paths": paths})
    result = classify_create_intent("write the RAG system, frontend and backend in separate folders", "")
    assert result["paths"] == paths


def test_classify_create_intent_returns_false_for_an_edit_shaped_request(monkeypatch):
    _mock_openai(monkeypatch, {"wants_new_file": False, "paths": []})
    result = classify_create_intent("fix the bug in the login handler", "app.py\nutils.py")
    assert result["wants_new_file"] is False
    assert result["paths"] == []
