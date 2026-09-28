"""Offline tests for api/pending_confirmations.py's storage backend
switch -- Redis is a real, optional, network-dependent service, so every
test here mocks its client (never a live server) and confirms both the
default in-memory path and the Redis-backed path behave identically from
stash()/resolve()'s own callers' point of view.
"""

import json

import pytest

from incremental_editing.api import pending_confirmations as pc


@pytest.fixture(autouse=True)
def _reset_state():
    pc.get_settings.cache_clear()
    pc._pending.clear()
    pc._redis_client = None
    yield
    pc.get_settings.cache_clear()
    pc._pending.clear()
    pc._redis_client = None


def test_stash_and_resolve_reject_uses_in_memory_dict_by_default(monkeypatch):
    # An explicit empty override, not delenv: pydantic-settings also
    # reads .env itself (see config.py's own `env_file=".env"`), so a
    # real REDIS_URL set there for actual `iee serve` use would otherwise
    # leak into this test the moment the OS env var alone is unset.
    monkeypatch.setenv("REDIS_URL", "")
    pc.get_settings.cache_clear()
    pc.stash("run-1", "create", files={"a.py": "x"}, request="r", project_id="p", metadata={"result": {"status": "pending"}})
    assert "run-1" in pc._pending
    metadata = pc.resolve("run-1", accept=False)
    assert metadata["result"]["status"] == "rejected"
    assert "run-1" not in pc._pending


def test_resolve_raises_when_nothing_pending(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "")  # explicit: don't leak a real .env REDIS_URL into this test
    pc.get_settings.cache_clear()
    with pytest.raises(KeyError, match="run-missing"):
        pc.resolve("run-missing", accept=False)


class _FakeRedis:
    """Minimal fake matching the exact three calls this module makes:
    set(key, value, ex=ttl), get(key), delete(key) -- a plain dict is a
    faithful enough stand-in for what those calls actually need."""

    def __init__(self):
        self.store = {}
        self.ttls = {}

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.ttls[key] = ex

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)
        self.ttls.pop(key, None)


def test_stash_writes_to_redis_with_the_configured_ttl_when_redis_url_is_set(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("PENDING_CONFIRMATION_TTL_SECONDS", "3600")
    pc.get_settings.cache_clear()

    fake = _FakeRedis()
    monkeypatch.setattr(pc, "_redis", lambda: fake)

    pc.stash("run-2", "create", files={"a.py": "x"}, request="r", project_id="p", metadata={"result": {"status": "pending"}})

    assert "run-2" not in pc._pending  # never touches the in-memory dict once Redis is configured
    key = pc._redis_key("run-2")
    assert json.loads(fake.store[key])["request"] == "r"
    assert fake.ttls[key] == 3600


def test_resolve_reads_and_deletes_from_redis_when_redis_url_is_set(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    pc.get_settings.cache_clear()

    fake = _FakeRedis()
    monkeypatch.setattr(pc, "_redis", lambda: fake)

    pc.stash("run-3", "create", files={"a.py": "x"}, request="r", project_id="p", metadata={"result": {"status": "pending"}})
    metadata = pc.resolve("run-3", accept=False)
    assert metadata["result"]["status"] == "rejected"
    assert pc._redis_key("run-3") not in fake.store  # popped, not left behind

    with pytest.raises(KeyError):
        pc.resolve("run-3", accept=False)  # already resolved -- gone from Redis too
