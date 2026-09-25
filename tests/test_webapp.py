"""Offline tests for the FastAPI layer's chat-history persistence
(GET/POST /api/chat_history) -- no LLM call, no real MinIO/docker
required (storage is monkeypatched to a tmp-path LocalStorage exactly
like the run_pipeline tests do).
"""

from fastapi.testclient import TestClient

from incremental_editing import webapp
from incremental_editing.storage.minio_client import LocalStorage


def _client(tmp_path, monkeypatch):
    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr(webapp, "get_storage", lambda: storage)
    return TestClient(webapp.app)


def test_chat_history_is_empty_before_anything_is_saved(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.get("/api/chat_history", params={"project_dir": str(tmp_path / "proj")})
    assert resp.status_code == 200
    assert resp.json() == {"messages": []}


def test_chat_history_round_trips_through_save_and_fetch(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    project_dir = str(tmp_path / "proj")
    messages = [{"id": 1, "role": "user", "text": "hello"}, {"id": 2, "role": "assistant", "text": "hi"}]

    save_resp = client.post("/api/chat_history", json={"project_dir": project_dir, "messages": messages})
    assert save_resp.status_code == 200

    fetch_resp = client.get("/api/chat_history", params={"project_dir": project_dir})
    assert fetch_resp.json() == {"messages": messages}


def test_chat_history_is_namespaced_per_project(tmp_path, monkeypatch):
    """Two different projects must never see each other's history --
    keyed by the same project_id derivation run_pipeline.run_edit uses."""
    client = _client(tmp_path, monkeypatch)
    proj_a = str(tmp_path / "proj_a")
    proj_b = str(tmp_path / "proj_b")

    client.post("/api/chat_history", json={"project_dir": proj_a, "messages": [{"id": 1, "text": "from a"}]})
    client.post("/api/chat_history", json={"project_dir": proj_b, "messages": [{"id": 1, "text": "from b"}]})

    assert client.get("/api/chat_history", params={"project_dir": proj_a}).json()["messages"][0]["text"] == "from a"
    assert client.get("/api/chat_history", params={"project_dir": proj_b}).json()["messages"][0]["text"] == "from b"


def test_chat_history_overwrites_rather_than_appends(tmp_path, monkeypatch):
    """A save must replace the prior history, not accumulate onto it --
    the frontend always sends the full current message list."""
    client = _client(tmp_path, monkeypatch)
    project_dir = str(tmp_path / "proj")

    client.post("/api/chat_history", json={"project_dir": project_dir, "messages": [{"id": 1, "text": "first"}]})
    client.post("/api/chat_history", json={"project_dir": project_dir, "messages": [{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]})

    resp = client.get("/api/chat_history", params={"project_dir": project_dir})
    assert [m["text"] for m in resp.json()["messages"]] == ["first", "second"]
