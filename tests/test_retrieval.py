"""Offline tests for the hybrid-retrieval subsystem (PHOENIX doc sections
1 and 4) -- everything deterministic and local: repo indexing, symbol/
BM25 retrieval, RRF fusion, the native call graph, risk classification,
and Semgrep structural matching (a real local subprocess, no network, so
safe to run automatically -- unlike vector retrieval which needs the
Bifrost embeddings endpoint and is exercised manually instead).
"""

from incremental_editing.retrieval.bm25_retriever import BM25Retriever
from incremental_editing.retrieval.dependency_graph import build_call_graph
from incremental_editing.retrieval.fusion import fuse
from incremental_editing.retrieval.joern_graph import build_call_graph_via_joern, is_available
from incremental_editing.retrieval.repo_index import RepoSymbol, build_repo_index
from incremental_editing.retrieval.risk import classify_risk
from incremental_editing.retrieval.semgrep_refs import structural_match_score
from incremental_editing.retrieval.symbol_retriever import SymbolRetriever


def _write_repo(tmp_path):
    (tmp_path / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def _internal_helper(x):\n"
        "    return x * 2\n"
    )
    (tmp_path / "auth.py").write_text(
        "def check_password(password, hashed):\n"
        "    \"\"\"Verify a user's password against a stored hash.\"\"\"\n"
        "    return hashed == password\n"
        "\n"
        "\n"
        "class SessionManager:\n"
        "    def create_session(self, user):\n"
        "        token = check_password(user, 'x')\n"
        "        return token\n"
    )
    return tmp_path


def test_build_repo_index_finds_symbols_across_multiple_files(tmp_path):
    repo = _write_repo(tmp_path)
    symbols = build_repo_index(str(repo))
    names = {s.name for s in symbols}
    assert {"add", "_internal_helper", "check_password", "SessionManager", "create_session"} <= names
    files = {s.file for s in symbols}
    assert "calculator.py" in files
    assert "auth.py" in files


def test_symbol_retriever_finds_literal_name_across_repo(tmp_path):
    repo = _write_repo(tmp_path)
    symbols = build_repo_index(str(repo))
    ranked = SymbolRetriever(symbols).rank("fix a bug in check_password")
    assert ranked[0][0].name == "check_password"


def test_bm25_retriever_scores_docstring_overlap(tmp_path):
    repo = _write_repo(tmp_path)
    symbols = build_repo_index(str(repo))
    ranked = BM25Retriever(symbols).rank("verify a user's password against a hash")
    assert ranked
    assert ranked[0][0].name == "check_password"


def test_fusion_prefers_symbol_agreed_on_by_multiple_retrievers():
    sym_a = RepoSymbol("a.py", "target", "function", 1, 3, None, "def target(): pass")
    sym_b = RepoSymbol("b.py", "other", "function", 1, 3, None, "def other(): pass")

    # "target" wins in both rankings; "other" only shows up in one.
    ranking_1 = [(sym_a, 5.0), (sym_b, 1.0)]
    ranking_2 = [(sym_a, 3.0)]

    fused = fuse([ranking_1, ranking_2], top_k=5)
    assert fused[0][0].name == "target"


def test_build_call_graph_finds_real_call_edges(tmp_path):
    repo = _write_repo(tmp_path)
    symbols = build_repo_index(str(repo))
    graph = build_call_graph(symbols)
    assert "check_password" in graph["calls"]["auth.py::create_session"]
    assert "auth.py::create_session" in graph["called_by"]["check_password"]


def test_classify_risk_flags_auth_keywords_and_private_helpers(tmp_path):
    repo = _write_repo(tmp_path)
    symbols = build_repo_index(str(repo))
    by_name = {s.name: s for s in symbols}

    assert classify_risk(by_name["check_password"]) == "HIGH"  # keyword match
    assert classify_risk(by_name["SessionManager"]) == "HIGH"  # public class
    assert classify_risk(by_name["_internal_helper"]) == "LOW"  # private helper
    assert classify_risk(by_name["add"]) == "MEDIUM"  # ordinary public function


def test_structural_match_score_confirms_real_validation_code():
    """"Meaning-wise" matching: a request mentioning "validate types" should
    only score a candidate whose body *actually* validates types (a real
    isinstance/raise pattern), not one that merely shares that vocabulary."""
    validating_fn = (
        "def add(a, b):\n"
        "    if not isinstance(a, (int, float)):\n"
        "        raise TypeError('not a number')\n"
        "    return a + b\n"
    )
    non_validating_fn = "def add(a, b):\n    return a + b\n"

    assert structural_match_score(validating_fn, "validate the argument types") > 0
    assert structural_match_score(non_validating_fn, "validate the argument types") == 0


def test_structural_match_score_is_zero_when_request_names_no_known_intent():
    source = "def add(a, b):\n    return a + b\n"
    assert structural_match_score(source, "make this function a bit faster") == 0


def test_locate_degrades_gracefully_when_vector_retrieval_fails(tmp_path, monkeypatch):
    """Real incident: the embeddings gateway cold-starting after idle threw
    APITimeoutError straight out of locate() with no fallback -- every
    other optional signal here (Semgrep, the locator's semantic tiebreak)
    already degrades to "signal unavailable" on failure; vector retrieval
    was the one path that didn't, and it crashed the whole find/locate
    request instead of just proceeding on symbol+BM25 alone."""
    from incremental_editing.retrieval.locate_repo import locate, locate_best_file

    _write_repo(tmp_path)

    def _boom(self, query, top_k=10):
        raise RuntimeError("embeddings gateway unreachable")

    monkeypatch.setattr("incremental_editing.retrieval.vector_retriever.VectorRetriever.rank", _boom)

    result = locate(str(tmp_path), "check the user's password", use_vector=True, use_semgrep=False)
    assert result["candidates"]  # symbol+BM25 still found something -- not a crash, not an empty result
    assert result["candidates"][0]["symbol"] == "check_password"

    best_file = locate_best_file(str(tmp_path), "check the user's password", use_vector=True)
    assert best_file == "auth.py"


def test_is_available_never_raises():
    """Must work identically whether or not Joern is actually installed
    on the machine running this -- CI and most dev machines won't have
    it (it's a separate ~1.7GB system install, not a pip dependency)."""
    assert isinstance(is_available(), bool)


def test_build_call_graph_via_joern_returns_none_when_unavailable(tmp_path, monkeypatch):
    """Joern is an opt-in enhancement, not a requirement -- if it isn't
    installed, this must return None (a clean, checkable "didn't run"),
    not raise, so callers can fall back to the native call graph."""
    _write_repo(tmp_path)
    symbols = build_repo_index(str(tmp_path))
    monkeypatch.setattr("incremental_editing.retrieval.joern_graph.is_available", lambda: False)
    assert build_call_graph_via_joern(str(tmp_path), symbols) is None


def test_build_call_graph_via_joern_returns_none_on_subprocess_failure(tmp_path, monkeypatch):
    """Joern reports available (on PATH) but the actual parse/query pass
    fails (bad frontend, corrupt CPG, timeout, crash) -- must still
    return None rather than propagate the subprocess error, same
    degrade-gracefully contract as every other optional signal here."""
    _write_repo(tmp_path)
    symbols = build_repo_index(str(tmp_path))
    monkeypatch.setattr("incremental_editing.retrieval.joern_graph.is_available", lambda: True)

    def _boom(*args, **kwargs):
        raise RuntimeError("joern-parse crashed")

    monkeypatch.setattr("subprocess.run", _boom)
    assert build_call_graph_via_joern(str(tmp_path), symbols) is None


def test_locate_falls_back_to_native_graph_when_joern_unavailable(tmp_path, monkeypatch):
    """use_joern=True on a machine without Joern installed must not
    crash `locate()` or silently drop call-graph enrichment -- it should
    fall back to the native AST-based graph exactly as if use_joern had
    been False."""
    from incremental_editing.retrieval.locate_repo import locate

    _write_repo(tmp_path)
    monkeypatch.setattr("incremental_editing.retrieval.joern_graph.is_available", lambda: False)

    result = locate(str(tmp_path), "check the user's password", use_vector=False, use_semgrep=False, use_joern=True)
    assert result["candidates"]
    top = result["candidates"][0]
    assert top["symbol"] == "check_password"
    # native fallback still ran and found the real call edge (create_session -> check_password)
    assert top["called_by_count"] >= 1
    assert result["used_joern"] is False  # fallback used, not the real thing -- must say so honestly


def test_locate_auto_joern_skips_when_confidence_is_high(tmp_path, monkeypatch):
    """The confidence/need gate's whole point: use_joern="auto" must NOT
    pay Joern's real build cost (~12-45s+) when the ranking already has a
    clear, uncontested winner -- that's exactly the case where the extra
    accuracy buys nothing."""
    from incremental_editing.retrieval import locate_repo

    _write_repo(tmp_path)
    monkeypatch.setattr(locate_repo, "retrieval_confidence", lambda fused: 0.9)

    calls = []
    monkeypatch.setattr(locate_repo, "build_call_graph_via_joern", lambda *a, **k: calls.append(1) or None)

    result = locate_repo.locate(str(tmp_path), "check the user's password", use_vector=False, use_joern="auto")
    assert calls == []  # never even attempted
    assert result["used_joern"] is False


def test_locate_auto_joern_escalates_when_confidence_is_low(tmp_path, monkeypatch):
    """The other half of the gate: a near-tie between the top candidate
    and the runner-up is exactly when the more expensive, more accurate
    evidence is worth its cost -- use_joern="auto" must actually attempt
    it in that case, not just when forced with use_joern=True."""
    from incremental_editing.retrieval import locate_repo

    _write_repo(tmp_path)
    monkeypatch.setattr(locate_repo, "retrieval_confidence", lambda fused: 0.3)

    fake_graph = {"calls": {}, "called_by": {}}
    calls = []
    monkeypatch.setattr(
        locate_repo, "build_call_graph_via_joern", lambda *a, **k: calls.append(1) or fake_graph
    )

    result = locate_repo.locate(str(tmp_path), "check the user's password", use_vector=False, use_joern="auto")
    assert len(calls) == 1  # attempted exactly once, not skipped
    assert result["used_joern"] is True
