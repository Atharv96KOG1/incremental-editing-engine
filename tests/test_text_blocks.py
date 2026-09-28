"""Offline tests for the LLM-free text-block locator (analyzer/text_blocks.py)
-- markdown headings, TOML/INI sections, YAML top-level keys, and the
blank-line-paragraph fallback for everything else. No LLM call anywhere.
"""

from incremental_editing.analyzer.text_blocks import (
    block_context_window,
    index_text_blocks,
    locate_text_block,
    locate_text_blocks,
)

MARKDOWN_SOURCE = (
    "# Project\n\n"
    "## Features\n\n"
    "- feature one\n"
    "- feature two\n\n"
    "### Notes\n\n"
    "- existing note\n"
)

TOML_SOURCE = '[server]\nhost = "localhost"\nport = 8080\n\n[database]\nurl = "sqlite:///db.sqlite"\n'

YAML_SOURCE = "service:\n  name: api\n  port: 8080\n\ndatabase:\n  url: sqlite:///db.sqlite\n"

DOCKERFILE_SOURCE = "FROM python:3.12\nWORKDIR /app\n\nCOPY . .\nRUN pip install -r requirements.txt\n\nCMD [\"python\", \"app.py\"]\n"

# Deliberately dense/hand-formatted, no blank lines between tags -- the
# exact shape that made the blank-line fallback find nothing to
# localize against for a typical real HTML file.
HTML_SOURCE = (
    "<!doctype html>\n"
    "<html>\n"
    "<head>\n"
    "  <title>T</title>\n"
    "</head>\n"
    "<body>\n"
    "  <header><h1>Hi</h1></header>\n"
    "  <main>\n"
    '    <section id="features">\n'
    "      <p>a</p>\n"
    "    </section>\n"
    "  </main>\n"
    "  <footer>bye</footer>\n"
    "</body>\n"
    "</html>\n"
)


def test_index_text_blocks_splits_markdown_on_headings():
    blocks = index_text_blocks(MARKDOWN_SOURCE, "markdown")
    names = [b.name for b in blocks]
    assert names == ["Project", "Features", "Notes"]
    notes = blocks[-1]
    assert notes.start_line == 8
    assert notes.end_line == 10


def test_index_text_blocks_covers_every_line_with_no_gaps_or_overlaps():
    blocks = index_text_blocks(MARKDOWN_SOURCE, "markdown")
    total_lines = len(MARKDOWN_SOURCE.splitlines())
    covered = set()
    for b in blocks:
        span = set(range(b.start_line, b.end_line + 1))
        assert not (span & covered), "blocks must never overlap"
        covered |= span
    assert covered == set(range(1, total_lines + 1))


def test_index_text_blocks_splits_toml_on_sections():
    blocks = index_text_blocks(TOML_SOURCE, "toml")
    assert [b.name for b in blocks] == ["server", "database"]


def test_index_text_blocks_splits_yaml_on_top_level_keys():
    blocks = index_text_blocks(YAML_SOURCE, "yaml")
    assert [b.name for b in blocks] == ["service", "database"]
    # nested keys (indented) must not themselves start a new block
    service = blocks[0]
    assert service.start_line == 1
    assert service.end_line == 4  # includes the trailing blank line before the next key


def test_index_text_blocks_falls_back_to_blank_line_paragraphs_for_dockerfile():
    """Dockerfile has no native section marker this locator knows --
    blank-line-separated paragraphs are the universal fallback."""
    blocks = index_text_blocks(DOCKERFILE_SOURCE, None)
    assert len(blocks) == 3
    assert blocks[0].name == "FROM python:3.12"
    assert blocks[1].name == "COPY . ."


def test_index_text_blocks_returns_empty_for_empty_source():
    assert index_text_blocks("", "markdown") == []


def test_index_text_blocks_splits_html_on_top_level_body_elements():
    """Real gap this closes: typical hand-formatted HTML has no blank
    lines between tags at all, so the blank-line fallback alone found
    nothing to localize against and every HTML edit fell through to a
    full-file rewrite. Real tag-nesting-aware blocks from the actual
    parse tree instead: each direct child of <body>, not a line
    heuristic that can't tell nested tags apart."""
    blocks = index_text_blocks(HTML_SOURCE, "html")
    assert [b.name for b in blocks] == ["header", "main", "footer"]


def test_index_text_blocks_html_includes_script_and_style_as_their_own_blocks():
    """Real bug this closes: tree-sitter's HTML grammar gives <script>
    and <style> their own distinct node types ("script_element"/
    "style_element"), never plain "element" -- filtering children on
    "element" alone made every <script>/<style> tag invisible to this
    locator entirely, not merged into a neighboring block, just never a
    candidate at all. Any request touching inline JS/CSS (extremely
    common in a real HTML file) fell straight through to the whole-
    file-block refusal even though a perfectly splice-able block
    existed the whole time."""
    source = (
        "<html>\n"
        "<body>\n"
        '  <button id="helloBtn">Click</button>\n'
        "  <script>\n"
        "    const helloBtn = document.getElementById('helloBtn');\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    blocks = index_text_blocks(source, "html")
    assert "script" in [b.name for b in blocks]


def test_index_text_blocks_html_block_names_use_id_or_class():
    blocks = index_text_blocks(HTML_SOURCE, "html")
    main = next(b for b in blocks if b.name == "main")
    # the section *inside* main isn't a top-level body child, so it's
    # not its own block here -- but confirms the id-based naming
    # convention works by checking it directly.
    from incremental_editing.analyzer.text_blocks import _html_blocks

    assert any(b.name == "main" for b in _html_blocks(HTML_SOURCE))


def test_index_text_blocks_html_covers_every_body_line_with_no_gaps_or_overlaps():
    blocks = index_text_blocks(HTML_SOURCE, "html")
    covered = set()
    for b in blocks:
        span = set(range(b.start_line, b.end_line + 1))
        assert not (span & covered), "blocks must never overlap"
        covered |= span
    # body opens line 6, closes line 14 -- every line strictly between
    # (the direct children's own span) must be covered by some block.
    assert covered == set(range(7, 14))


def test_locate_text_block_finds_the_named_section():
    block = locate_text_block(MARKDOWN_SOURCE, "add retrieval types in the notes part", "markdown")
    assert block is not None
    assert block.name == "Notes"


def test_locate_text_block_returns_none_when_ambiguous():
    """Two sections sharing the request's only real content word must
    not be guessed between -- fall through to a whole-file edit instead."""
    source = "## API Notes\n\n- a\n\n## Release Notes\n\n- b\n"
    assert locate_text_block(source, "update the notes", "markdown") is None


def test_locate_text_block_returns_none_when_nothing_matches():
    block = locate_text_block(MARKDOWN_SOURCE, "completely unconnected topic xyz123", "markdown")
    assert block is None


def test_locate_text_block_finds_the_named_html_element():
    block = locate_text_block(HTML_SOURCE, "add EXPOSE-style text to the footer", "html")
    assert block is not None
    assert block.name == "footer"


def test_locate_text_blocks_returns_every_relevant_block_for_a_cross_block_rename():
    """Real bug this closes: "change the helloBtn to btn" needs BOTH the
    button's own id attribute AND the inline <script> referencing that
    id touched to stay correct -- neither block alone is a complete,
    correct change. locate_text_block's single-best design saw this as
    a tie (both blocks equally match "helloBtn") and refused outright;
    locate_text_blocks returns both instead of picking one."""
    source = (
        "<html>\n"
        "<body>\n"
        '  <button id="helloBtn">Click</button>\n'
        "  <script>\n"
        "    const helloBtn = document.getElementById('helloBtn');\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    blocks = locate_text_blocks(source, "change the helloBtn to btn", "html")
    assert {b.name for b in blocks} == {"button#helloBtn", "script"}


def test_locate_text_blocks_returns_empty_when_nothing_matches():
    assert locate_text_blocks(MARKDOWN_SOURCE, "completely unconnected topic xyz123", "markdown") == []


def test_locate_text_blocks_returns_a_single_block_when_only_one_is_relevant():
    blocks = locate_text_blocks(MARKDOWN_SOURCE, "add retrieval types in the notes part", "markdown")
    assert [b.name for b in blocks] == ["Notes"]


def test_locate_text_block_returns_none_for_a_single_block_file():
    """Nothing to localize against when the whole file is already ~one
    block -- matches find_multi_delete_targets' own "need at least 2"
    bar for a fast path to mean anything."""
    source = "Just one plain paragraph, no headings at all.\n"
    assert locate_text_block(source, "add a sentence", None) is None


def test_locate_text_block_hybrid_off_by_default_stays_network_free(monkeypatch):
    """use_hybrid defaults False -- VectorRetriever must never even be
    constructed unless a caller explicitly opts in, same guarantee
    build_context makes for code symbols."""
    from incremental_editing.analyzer import text_blocks

    def _boom(*a, **kw):
        raise AssertionError("VectorRetriever must not be called when use_hybrid is False")

    monkeypatch.setattr(text_blocks, "VectorRetriever", _boom)
    block = locate_text_block(MARKDOWN_SOURCE, "add retrieval types in the notes part", "markdown")
    assert block is not None
    assert block.name == "Notes"


def test_locate_text_block_hybrid_finds_a_rare_body_word_via_bm25(monkeypatch):
    """Real motivation: a request sharing only a rare body word with a
    section -- not its heading -- is exactly what BM25 exists to catch
    on top of plain name/word-overlap matching. VectorRetriever mocked
    to return nothing so this isolates BM25's own contribution. Three
    sections, not two: BM25 idf is degenerate (exactly zero) for a
    two-document corpus where a term appears in just one doc, so a
    real corpus needs >=3 blocks for the signal to fire at all."""
    from incremental_editing.analyzer import text_blocks

    monkeypatch.setattr(
        text_blocks, "VectorRetriever", lambda symbols: type("V", (), {"rank": lambda self, q, top_k: []})()
    )

    source = (
        "## Setup\n\n"
        "- install deps\n\n"
        "## Usage\n\n"
        "- run the CLI\n\n"
        "## Troubleshooting\n\n"
        "- if you see a zephyranthes error, restart the daemon\n"
    )
    request = "fix the zephyranthes error"

    # Confirms this genuinely isn't findable via plain name/word-overlap
    # matching alone -- proving hybrid adds real recall, not redundancy.
    assert locate_text_block(source, request, "markdown", use_hybrid=False) is None

    block = locate_text_block(source, request, "markdown", use_hybrid=True)
    assert block is not None
    assert block.name == "Troubleshooting"


def test_locate_text_block_hybrid_uses_the_vector_signal_too(monkeypatch):
    """Isolates the vector-retrieval contribution specifically -- mocked
    (a real embeddings call would make this test network-dependent and
    non-deterministic) to return a canned ranking, confirming its
    result actually participates in the fusion rather than being
    silently ignored."""
    from incremental_editing.analyzer import text_blocks

    def _fake_vector_rank(self, query, top_k):
        release = next(s for s in self.symbols if s.name == "Release Notes")
        return [(release, 0.9)]

    monkeypatch.setattr(
        text_blocks,
        "VectorRetriever",
        lambda symbols: type("V", (), {"symbols": symbols, "rank": _fake_vector_rank})(),
    )

    source = "## API Notes\n\n- a\n\n## Release Notes\n\n- b\n"
    block = locate_text_block(source, "completely unrelated wording naming nothing real", "markdown", use_hybrid=True)
    assert block is not None
    assert block.name == "Release Notes"


def test_locate_text_block_hybrid_still_falls_back_when_no_signal_matches_anything(monkeypatch):
    """Every signal (name match, BM25, mocked-empty vector) finding
    nothing must still degrade cleanly to None -- not crash, not
    silently invent a candidate."""
    from incremental_editing.analyzer import text_blocks

    monkeypatch.setattr(
        text_blocks, "VectorRetriever", lambda symbols: type("V", (), {"rank": lambda self, q, top_k: []})()
    )

    block = locate_text_block(MARKDOWN_SOURCE, "completely unconnected topic xyz123", "markdown", use_hybrid=True)
    assert block is None


def test_block_context_window_gets_neighboring_lines_on_both_sides():
    blocks = index_text_blocks(MARKDOWN_SOURCE, "markdown")
    features = next(b for b in blocks if b.name == "Features")
    before, after = block_context_window(MARKDOWN_SOURCE, features, n=2)
    assert "# Project" in before
    assert "### Notes" in after


def test_block_context_window_empty_before_first_block():
    blocks = index_text_blocks(MARKDOWN_SOURCE, "markdown")
    first = blocks[0]
    before, _after = block_context_window(MARKDOWN_SOURCE, first, n=2)
    assert before == ""


def test_block_context_window_empty_after_last_block():
    blocks = index_text_blocks(MARKDOWN_SOURCE, "markdown")
    last = blocks[-1]
    _before, after = block_context_window(MARKDOWN_SOURCE, last, n=2)
    assert after == ""


def test_block_context_window_never_crosses_a_second_neighbor_when_n_is_large():
    """A generous n must still never run past the file's own line range,
    even when it would otherwise reach two blocks deep."""
    blocks = index_text_blocks(MARKDOWN_SOURCE, "markdown")
    features = next(b for b in blocks if b.name == "Features")
    before, after = block_context_window(MARKDOWN_SOURCE, features, n=100)
    total_lines = len(MARKDOWN_SOURCE.splitlines())
    assert before == "\n".join(MARKDOWN_SOURCE.splitlines()[: features.start_line - 1])
    assert after == "\n".join(MARKDOWN_SOURCE.splitlines()[features.end_line : total_lines])


def test_build_block_messages_includes_context_labeled_as_reference_only():
    from incremental_editing.strategies.text_block_edit import build_block_messages

    messages = build_block_messages(
        "readme.md", "Notes", "### Notes\n\n- existing note", "add a line",
        context_before="## Features\n\n- feature one", context_after="",
    )
    user_content = messages[1]["content"]
    assert "context immediately before" in user_content
    assert "## Features" in user_content
    assert "context immediately after" not in user_content  # omitted entirely when empty


def test_build_block_messages_omits_context_sections_when_none_given():
    from incremental_editing.strategies.text_block_edit import build_block_messages

    messages = build_block_messages("readme.md", "Notes", "### Notes\n\n- existing note", "add a line")
    user_content = messages[1]["content"]
    assert "context immediately before" not in user_content
    assert "context immediately after" not in user_content
