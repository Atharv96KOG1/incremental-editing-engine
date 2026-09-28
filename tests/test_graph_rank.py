"""Offline tests for retrieval/graph_rank.py -- personalized PageRank
over the native call graph, adapted from Aider's own repo-map ranking.
Pure NumPy power iteration, no network, no external graph library.
"""

from incremental_editing.retrieval.graph_rank import _personalized_pagerank, graph_centrality_ranking
from incremental_editing.retrieval.repo_index import RepoSymbol


def _sym(file, name, source):
    return RepoSymbol(
        file=file,
        name=name,
        symbol_type="function",
        start_line=1,
        end_line=len(source.splitlines()),
        docstring_first_line=None,
        source=source,
        language="python",
    )


def _repo():
    # helper() is called by both main() and other_caller() -- structurally
    # the most central symbol here. unrelated() calls and is called by
    # nothing.
    return [
        _sym("a.py", "main", "def main():\n    helper()\n    helper()\n"),
        _sym("b.py", "helper", "def helper():\n    pass\n"),
        _sym("c.py", "other_caller", "def other_caller():\n    helper()\n"),
        _sym("d.py", "unrelated", "def unrelated():\n    pass\n"),
    ]


def test_personalized_pagerank_ranks_the_most_referenced_node_highest():
    nodes = ["a", "b", "c"]
    edges = [("a", "b", 1.0), ("c", "b", 1.0)]  # b is referenced by both a and c
    ranked = _personalized_pagerank(nodes, edges)
    assert ranked["b"] > ranked["a"]
    assert ranked["b"] > ranked["c"]


def test_personalized_pagerank_handles_an_empty_graph():
    assert _personalized_pagerank([], []) == {}


def test_graph_centrality_ranking_ranks_the_most_called_symbol_highest():
    """Real motivation this closes: a vague request that doesn't
    textually match any symbol name still has a real, different signal
    available -- how structurally central a symbol is in the call
    graph. helper() is called from two different places; unrelated(),
    main(), and other_caller() are not called at all."""
    ranked = graph_centrality_ranking(_repo(), "zzzqqqxyz nonword tokens only")
    names_in_order = [sym.name for sym, _ in ranked]
    assert names_in_order[0] == "helper"


def test_graph_centrality_ranking_personalization_boosts_a_mentioned_symbol():
    """Same role Aider's own "mentioned_idents" plays in its repo-map
    ranking: a symbol the request actually names should outrank an
    equally-uncalled symbol it doesn't."""
    baseline = {sym.name: score for sym, score in graph_centrality_ranking(_repo(), "zzzqqqxyz nonword tokens only")}
    boosted = {sym.name: score for sym, score in graph_centrality_ranking(_repo(), "fix a bug in other_caller")}
    assert boosted["other_caller"] > baseline["other_caller"]


def test_graph_centrality_ranking_returns_empty_for_a_single_symbol_repo():
    assert graph_centrality_ranking([_sym("a.py", "solo", "def solo():\n    pass\n")], "anything") == []


def test_graph_centrality_ranking_reuses_a_precomputed_call_graph(monkeypatch):
    """locate_repo.locate() already builds the native call graph for its
    own candidate evidence -- passing it in must skip a second,
    identical AST/Tree-sitter pass rather than silently recomputing."""
    import incremental_editing.retrieval.graph_rank as graph_rank_module

    calls = []
    monkeypatch.setattr(
        graph_rank_module,
        "build_call_graph",
        lambda symbols: (_ for _ in ()).throw(AssertionError("must not recompute when call_graph is given")),
    )
    prebuilt = {"calls": {"a.py::main": ["helper"], "b.py::helper": [], "c.py::other_caller": ["helper"], "d.py::unrelated": []}, "called_by": {}}
    ranked = graph_centrality_ranking(_repo(), "anything", call_graph=prebuilt)
    assert ranked  # didn't raise -- confirms the prebuilt graph was actually used
