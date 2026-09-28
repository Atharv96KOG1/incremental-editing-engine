"""Offline tests for the static, zero-LLM-call code metadata builder --
every field here comes from AST/Tree-sitter parsing, never a model call.
"""

import json
import os

from incremental_editing.analyzer.metadata_builder import (
    build_project_metadata,
    delete_file_metadata,
    extract_symbol_metadata,
    per_file_metadata_path,
    write_file_metadata,
)


def test_extract_symbol_metadata_reads_parameters_decorators_and_return_type():
    source = (
        "@app.route('/login')\n"
        "@staticmethod\n"
        "async def handler(request, *args, timeout=30, **kwargs) -> dict:\n"
        "    \"\"\"Handles login.\"\"\"\n"
        "    return {}\n"
    )
    syms = extract_symbol_metadata(source, "python")
    assert len(syms) == 1
    sym = syms[0]
    assert sym.name == "handler"
    assert sym.parameters == ["request", "*args", "timeout", "**kwargs"]
    assert sym.decorators == ["app.route('/login')", "staticmethod"]
    assert sym.return_type_hint == "dict"
    assert sym.is_async is True
    assert sym.docstring_first_line == "Handles login."
    assert sym.line_count == sym.end_line - sym.start_line + 1
    assert sym.content_hash  # non-empty, deterministic given the same source


def test_extract_symbol_metadata_content_hash_is_stable_and_sensitive():
    """Same symbol text -> same hash (so an unchanged symbol is
    detectable as unchanged); a real edit to the body -> a different
    hash (so a changed symbol is detectable as changed). Both directions
    matter for this to be useful as a cache-invalidation signal."""
    a = extract_symbol_metadata("def f(x):\n    return x + 1\n", "python")[0]
    b = extract_symbol_metadata("def f(x):\n    return x + 1\n", "python")[0]
    c = extract_symbol_metadata("def f(x):\n    return x + 2\n", "python")[0]
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


def test_extract_symbol_metadata_tracks_parent_class_and_keywords():
    source = (
        "class Trig:\n"
        "    def tan(self, x):\n"
        "        \"\"\"Compute the tangent.\"\"\"\n"
        "        return x\n"
    )
    syms = extract_symbol_metadata(source, "python")
    cls, method = syms
    assert cls.symbol_type == "class" and cls.parent_class is None
    assert method.symbol_type == "function" and method.parent_class == "Trig"
    assert "tan" in method.keywords
    assert "compute" in method.keywords  # from the docstring, stopwords already excluded
    assert "the" not in method.keywords


def test_extract_symbol_metadata_non_python_gets_the_honest_subset():
    """Non-Python languages get name/type/lines/keywords/hash from the
    same generic Tree-sitter extraction used elsewhere in this project
    -- not parameters/decorators/return-type, which would need a
    per-language grammar query this project deliberately doesn't build
    (same tradeoff as docstring_first_line/parent_class being
    Python-only in analyzer/locator.py)."""
    source = "public class Foo {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}\n"
    syms = extract_symbol_metadata(source, "java")
    add = next(s for s in syms if s.name == "add")
    assert add.parameters == []
    assert add.decorators == []
    assert add.return_type_hint is None
    assert add.content_hash


def test_extract_symbol_metadata_covers_markdown_toml_and_yaml():
    """Real ask this closes: markdown/TOML/YAML (and any other format
    with no function/class concept) previously got NO metadata at all --
    detect_language finds a real language for them, but Tree-sitter's
    generic function/class node-type heuristic naturally finds nothing
    in their grammars, so extract_symbol_metadata fell straight to
    empty. Same block-level fallback as the unrecognized-language case,
    just reached via a different route (a real language, zero symbols),
    covering every format this project's own locator already knows how
    to section, not just files with no detected language."""
    md = extract_symbol_metadata("# Title\n\nSome text.\n\n## Section\n\nMore.\n", "markdown")
    assert [s.symbol_type for s in md] == ["block", "block"]
    assert {s.name for s in md} == {"Title", "Section"}

    toml = extract_symbol_metadata('[server]\nhost = "localhost"\n\n[db]\nurl = "x"\n', "toml")
    assert {s.name for s in toml} == {"server", "db"}
    assert all(s.symbol_type == "block" for s in toml)

    yaml = extract_symbol_metadata("service:\n  name: api\n\ndatabase:\n  url: x\n", "yaml")
    assert {s.name for s in yaml} == {"service", "database"}


def test_extract_symbol_metadata_falls_back_to_a_block_for_unrecognized_language():
    """Real gap this closes: an unrecognized extension (or any format
    with no function/class concept at all -- markdown, TOML, YAML,
    HTML, Dockerfile, plain text) used to mean empty metadata, even
    though analyzer/text_blocks.py's own generic locator already finds
    real, addressable regions for exactly these files."""
    symbols = extract_symbol_metadata("whatever text", None)
    assert len(symbols) == 1
    assert symbols[0].symbol_type == "block"
    assert symbols[0].language == "text"


def test_extract_symbol_metadata_returns_empty_only_for_a_truly_empty_file():
    assert extract_symbol_metadata("", None) == []


def test_build_project_metadata_merges_call_graph_and_caches(tmp_path, monkeypatch):
    # Real bug this caught: build_project_metadata's cache dir was hardcoded
    # to ~/iee-metadata with no way to override it, so every run of this
    # test (a real caching test, needs use_cache=True to exercise the cache
    # path at all) left a permanent stray file in the user's actual home
    # directory -- 30 of them accumulated there over the course of this
    # project's own test runs before this was caught. Point the cache dir
    # at tmp_path itself so this test's caching behavior is still fully
    # exercised without ever touching the real one.
    monkeypatch.setattr("incremental_editing.analyzer.metadata_builder._CACHE_DIR", str(tmp_path / ".metadata_cache"))

    (tmp_path / "calc.py").write_text(
        "def helper(x):\n    return x + 1\n\n\ndef main():\n    return helper(2)\n"
    )
    doc = build_project_metadata(str(tmp_path))
    helper = next(s for s in doc["files"]["calc.py"]["symbols"] if s["name"] == "helper")
    main = next(s for s in doc["files"]["calc.py"]["symbols"] if s["name"] == "main")
    assert helper["called_by_count"] == 1  # main() calls it
    assert main["calls"] == ["helper"]

    # cache hit: unchanged files reuse the prior document rather than reparsing
    doc2 = build_project_metadata(str(tmp_path))
    assert doc2["generated_at"] == doc["generated_at"]

    # cache invalidation: touching the file's content forces a real rebuild
    (tmp_path / "calc.py").write_text(
        "def helper(x):\n    return x + 1\n\n\ndef main():\n    return helper(2)\n\n\ndef extra():\n    return 0\n"
    )
    doc3 = build_project_metadata(str(tmp_path))
    assert {s["name"] for s in doc3["files"]["calc.py"]["symbols"]} == {"helper", "main", "extra"}


def test_write_file_metadata_writes_one_json_per_file_with_a_real_block(tmp_path):
    """The per-commit metadata folder (distinct from build_project_metadata's
    one-JSON-per-*project* snapshot above) -- one file, mirroring the
    source file's own relative path, carrying a real "block" (the exact
    source lines) alongside name/start_line/end_line/keywords so a
    symbol can be found and read back exactly, not just located."""
    source = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n"
    out_path = write_file_metadata(str(tmp_path), "calculator.py", source)

    assert out_path == per_file_metadata_path(str(tmp_path), "calculator.py")
    with open(out_path) as f:
        doc = json.load(f)

    assert doc["file"] == "calculator.py"
    assert doc["language"] == "python"
    assert doc["total_lines"] == len(source.splitlines())
    add = next(s for s in doc["symbols"] if s["name"] == "add")
    assert add["start_line"] == 1
    assert add["end_line"] == 2
    assert add["block"] == "def add(a, b):\n    return a + b"
    assert "add" in add["keywords"]


def test_write_file_metadata_reflects_the_files_current_real_content(tmp_path):
    """Re-writing must fully replace the prior document -- a symbol
    deleted from the real file must not linger in its metadata."""
    write_file_metadata(str(tmp_path), "calc.py", "def add(a, b):\n    return a + b\n")
    write_file_metadata(str(tmp_path), "calc.py", "def multiply(a, b):\n    return a * b\n")

    with open(per_file_metadata_path(str(tmp_path), "calc.py")) as f:
        doc = json.load(f)
    names = {s["name"] for s in doc["symbols"]}
    assert names == {"multiply"}


def test_write_file_metadata_still_writes_for_a_file_with_no_detectable_language(tmp_path):
    """A file with an unrecognized extension is no longer "no symbols" --
    extract_symbol_metadata's block-level fallback still finds a real
    region (here, the whole file as one blank-line paragraph)."""
    path = write_file_metadata(str(tmp_path), "notes.xyzabc123", "just some plain text\n")
    assert path is not None
    assert os.path.exists(per_file_metadata_path(str(tmp_path), "notes.xyzabc123"))


def test_write_file_metadata_skips_a_truly_empty_file(tmp_path):
    assert write_file_metadata(str(tmp_path), "empty.xyzabc123", "") is None
    assert not os.path.exists(per_file_metadata_path(str(tmp_path), "empty.xyzabc123"))


def test_delete_file_metadata_removes_the_json_and_tolerates_a_missing_one(tmp_path):
    write_file_metadata(str(tmp_path), "calc.py", "def add(a, b):\n    return a + b\n")
    assert os.path.exists(per_file_metadata_path(str(tmp_path), "calc.py"))

    delete_file_metadata(str(tmp_path), "calc.py")
    assert not os.path.exists(per_file_metadata_path(str(tmp_path), "calc.py"))
    delete_file_metadata(str(tmp_path), "calc.py")  # must not raise when already gone
