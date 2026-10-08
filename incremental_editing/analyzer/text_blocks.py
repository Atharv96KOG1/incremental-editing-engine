"""LLM-free locator for files with no function/class concept at all --
markdown, YAML, TOML, INI, Dockerfile, plain text, dotenv, etc. Treats a
file's own natural section structure as pseudo-symbols (reusing
locator.SymbolInfo, symbol_type="block") instead of sending the whole
file: a markdown heading, TOML/INI [section], or YAML top-level key each
become one block; anything without a native marker falls back to
blank-line-separated paragraphs.

Same matching technique analyzer/locator.py uses for functions
(literal-name-in-request + word overlap), so context_builder-style
localization and hybrid retrieval apply to blocks the same way they do
to real symbols.
"""

import re
from typing import List, Optional

from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.fusion import fuse
from ..retrieval.repo_index import RepoSymbol
from ..retrieval.vector_retriever import VectorRetriever
from .locator import _GENERIC_LEADING_VERBS, _GENERIC_TYPE_WORDS, SymbolInfo, _words

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOML_INI_SECTION_RE = re.compile(r"^\[+([^\]]+)\]+\s*$")
_YAML_TOP_KEY_RE = re.compile(r"^([A-Za-z0-9_.\-]+):(?:\s|$)")


def _html_child_name(element, source_bytes: bytes) -> str:
    """A short, human-identifying label for an HTML element -- tag name,
    plus #id or .first-class-name when present (e.g. "section#features"),
    not a raw excerpt of markup."""
    start_tag = next((c for c in element.children if c.type == "start_tag"), None)
    if start_tag is None:
        return element.type
    tag_name_node = next((c for c in start_tag.children if c.type == "tag_name"), None)
    tag_name = source_bytes[tag_name_node.start_byte : tag_name_node.end_byte].decode("utf-8") if tag_name_node else "element"
    for attr in start_tag.children:
        if attr.type != "attribute":
            continue
        name_node = next((c for c in attr.children if c.type == "attribute_name"), None)
        attr_name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8") if name_node else ""
        if attr_name not in ("id", "class"):
            continue
        value_wrapper = next((c for c in attr.children if c.type in ("quoted_attribute_value", "attribute_value")), None)
        if value_wrapper is None:
            continue
        value_node = next((c for c in value_wrapper.children if c.type == "attribute_value"), value_wrapper)
        value = source_bytes[value_node.start_byte : value_node.end_byte].decode("utf-8").split()[0:1]
        if value:
            return f"{tag_name}#{value[0]}" if attr_name == "id" else f"{tag_name}.{value[0]}"
    return tag_name


def _find_element_by_tag(node, tag_name: str, source_bytes: bytes):
    for child in node.children:
        if child.type == "element":
            start_tag = next((c for c in child.children if c.type == "start_tag"), None)
            name_node = start_tag and next((c for c in start_tag.children if c.type == "tag_name"), None)
            if name_node and source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8") == tag_name:
                return child
        found = _find_element_by_tag(child, tag_name, source_bytes)
        if found is not None:
            return found
    return None


def _html_blocks(source: str) -> List[SymbolInfo]:
    """Tag-nesting-aware blocks for HTML -- each direct child element of
    <body> (or the document root for a bare fragment), extended to cover
    whitespace/text up to the next sibling. Built from tree-sitter's parse
    tree rather than blank-line guessing: typical hand-formatted HTML has
    no blank lines between tags, so the blank-line fallback alone finds
    nothing to localize against."""
    from tree_sitter_language_pack import get_parser

    source_bytes = source.encode("utf-8")
    tree = get_parser("html").parse(source_bytes)
    body = _find_element_by_tag(tree.root_node, "body", source_bytes)
    container = body if body is not None else tree.root_node
    children = [c for c in container.children if c.type in ("element", "script_element", "style_element")]
    if not children:
        return []

    blocks = []
    for idx, el in enumerate(children):
        start_line = el.start_point[0] + 1
        end_line = children[idx + 1].start_point[0] if idx + 1 < len(children) else container.end_point[0]
        blocks.append(
            SymbolInfo(
                name=_html_child_name(el, source_bytes),
                symbol_type="block",
                start_line=start_line,
                end_line=max(end_line, el.end_point[0] + 1),
                docstring_first_line=None,
                indent=0,
            )
        )
    return blocks

_MARKER_LOCATORS = {
    "markdown": (_HEADING_RE, 2),
    "toml": (_TOML_INI_SECTION_RE, 1),
    "ini": (_TOML_INI_SECTION_RE, 1),
    "yaml": (_YAML_TOP_KEY_RE, 1),
}


def _blocks_from_markers(lines: List[str], marker_re: "re.Pattern", name_group: int) -> List[SymbolInfo]:
    starts = [(i, m.group(name_group).strip()) for i, line in enumerate(lines, start=1) if (m := marker_re.match(line))]
    blocks = []
    for idx, (start_line, name) in enumerate(starts):
        end_line = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines)
        blocks.append(SymbolInfo(name=name, symbol_type="block", start_line=start_line, end_line=end_line, docstring_first_line=None, indent=0))
    return blocks


def _blocks_from_blank_lines(lines: List[str]) -> List[SymbolInfo]:
    """Fallback when the format has no native section marker: groups of
    consecutive non-blank lines, separated by blank lines. A block's
    "name" is its own first line (truncated)."""
    blocks: List[SymbolInfo] = []
    start: Optional[int] = None
    for i, line in enumerate(lines, start=1):
        if line.strip():
            if start is None:
                start = i
        elif start is not None:
            blocks.append(_make_blank_line_block(lines, start, i - 1))
            start = None
    if start is not None:
        blocks.append(_make_blank_line_block(lines, start, len(lines)))
    return blocks


def _make_blank_line_block(lines: List[str], start: int, end: int) -> SymbolInfo:
    label = lines[start - 1].strip()[:60]
    return SymbolInfo(
        name=label or f"block@{start}", symbol_type="block", start_line=start, end_line=end, docstring_first_line=None, indent=0
    )


def index_text_blocks(source: str, language: Optional[str]) -> List[SymbolInfo]:
    """Every block in `source`, covering the whole file with no gaps or
    overlaps -- markdown headings, TOML/INI sections, or YAML top-level
    keys when the format has one; tag-nesting-aware blocks for HTML;
    blank-line paragraphs otherwise (also the fallback if a file has none
    of its own format's markers)."""
    lines = source.splitlines()
    if not lines:
        return []
    if language == "html":
        try:
            blocks = _html_blocks(source)
        except Exception:
            blocks = []
        if blocks:
            return blocks
    marker = _MARKER_LOCATORS.get(language or "")
    if marker:
        blocks = _blocks_from_markers(lines, *marker)
        if blocks:
            return blocks
    return _blocks_from_blank_lines(lines)


def _name_match_ranking(blocks: List[SymbolInfo], user_request: str) -> List[tuple]:
    """Literal-name + word-overlap scoring, same technique
    locator.locate_candidates uses for functions (simplified: blocks have
    no enclosing-class concept). Returns [(score, block), ...] in
    descending order so the caller can detect ties or fuse with
    BM25/vector rankings."""
    request_lower = user_request.lower()
    request_words = _words(user_request)
    scored = []
    for block in blocks:
        name_words = _words(block.name)
        literal_hit = bool(block.name) and bool(re.search(rf"\b{re.escape(block.name.lower())}\b", request_lower))
        overlap = len(name_words & request_words)
        score = (5 if literal_hit else 0) + overlap
        if score > 0:
            scored.append((score, block))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def _body_match_ranking(blocks: List[SymbolInfo], source: str, user_request: str) -> List[tuple]:
    """Same technique locate_candidates_by_body uses for functions: a
    request word absent from a block's NAME can still be in its body --
    e.g. an HTML <script>/<style> block, whose name is just
    "script"/"style". Returns [(score, block), ...], same shape as
    _name_match_ranking."""
    target_words = {w for w in _words(user_request) - _GENERIC_LEADING_VERBS - _GENERIC_TYPE_WORDS if len(w) >= 3}
    if not target_words:
        return []
    lines = source.splitlines()
    scored = []
    for block in blocks:
        body = "\n".join(lines[block.start_line - 1 : block.end_line])
        hits = len(target_words & _words(body))
        if hits:
            scored.append((hits, block))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def locate_text_blocks(source: str, user_request: str, language: Optional[str]) -> List[SymbolInfo]:
    """Every block whose name OR body relates to the request -- not just
    the single best match locate_text_block commits to. Needed when an
    edit genuinely spans two blocks (e.g. a rename touching both a markup
    tag's id and its inline <script> reference), which locate_text_block
    would otherwise see as a tie and refuse.

    Returns [] when nothing scores; otherwise every scoring block, ranked
    highest first -- caller decides single- vs multi-block edit. A block
    that scores only incidentally costs nothing extra: the multi-block
    prompt tells the model to leave unrelated blocks byte-identical."""
    blocks = index_text_blocks(source, language)
    if len(blocks) < 2:
        return []

    name_scored = {id(b): s for s, b in _name_match_ranking(blocks, user_request)}
    body_scored = {id(b): s for s, b in _body_match_ranking(blocks, source, user_request)}
    matched = [b for b in blocks if id(b) in name_scored or id(b) in body_scored]
    matched.sort(key=lambda b: max(name_scored.get(id(b), 0), body_scored.get(id(b), 0)), reverse=True)
    return matched


def _to_repo_symbols(blocks: List[SymbolInfo], source: str) -> List[RepoSymbol]:
    """Same RepoSymbol adapter context_builder.py uses for function/class
    symbols, here for blocks -- BM25Retriever/VectorRetriever/fuse are
    duck-typed, so the same retrieval machinery applies to a text block
    just as well as a function."""
    lines = source.splitlines()
    return [
        RepoSymbol(
            file="",
            name=b.name,
            symbol_type="block",
            start_line=b.start_line,
            end_line=b.end_line,
            docstring_first_line=None,
            source="\n".join(lines[b.start_line - 1 : b.end_line]),
            language="text",
        )
        for b in blocks
    ]


def _safe_vector_rank(repo_blocks: List[RepoSymbol], user_request: str, top_k: int) -> list:
    """Vector retrieval is one signal among several -- a failure here
    degrades to "signal didn't fire" rather than failing the whole
    locate call, same as context_builder.py's _safe_vector_rank."""
    try:
        return VectorRetriever(repo_blocks).rank(user_request, top_k=top_k)
    except Exception:
        return []


def locate_text_block(
    source: str, user_request: str, language: Optional[str], use_hybrid: bool = False
) -> Optional[SymbolInfo]:
    """The single best-matching block for a request, or None when nothing
    scores confidently (including a tie). Returns at most one: a block
    edit commits to regenerating exactly the block returned, so an
    ambiguous match must fall through to a whole-file edit rather than
    guess.

    `use_hybrid=True` fuses the name/word-overlap ranking with BM25 and
    vector/semantic retrieval over every block in the file, same
    reasoning as context_builder._hybrid_rerank for functions. Note:
    vector retrieval is a real embeddings-API call, not free."""
    blocks = index_text_blocks(source, language)
    if len(blocks) < 2:
        return None

    scored = _name_match_ranking(blocks, user_request)
    if not use_hybrid:
        if not scored:
            return None
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None
        return scored[0][1]

    repo_blocks = _to_repo_symbols(blocks, source)
    by_key = {(rb.name, rb.start_line): b for rb, b in zip(repo_blocks, blocks)}
    name_repo_key = {(b.name, b.start_line) for _, b in scored}
    name_ranking = [(rb, 0.0) for rb in repo_blocks if (rb.name, rb.start_line) in name_repo_key]

    wide_k = max(len(blocks), 2)
    rankings = [name_ranking, BM25Retriever(repo_blocks).rank(user_request, top_k=wide_k)]
    vector_ranking = _safe_vector_rank(repo_blocks, user_request, wide_k)
    if vector_ranking:
        rankings.append(vector_ranking)

    fused = fuse(rankings, top_k=wide_k)
    if not fused:
        return None
    if len(fused) > 1 and fused[0][1] == fused[1][1]:
        return None
    winner_key = (fused[0][0].name, fused[0][0].start_line)
    return by_key.get(winner_key)


def block_context_window(source: str, block: SymbolInfo, n: int = 2) -> tuple:
    """A few lines immediately outside `block`'s own range, read-only
    context for generation -- NOT part of what gets regenerated or
    spliced back (block boundaries stay exact). Lets a boundary-sensitive
    edit ("match the tone above") see neighboring content without
    expanding what gets rewritten; standard RAG chunk-overlap technique,
    since a block has no name-line shorthand the way a function signature
    gives code-symbol context.

    Bounded by neighboring content only, never crosses the file's own
    line range."""
    lines = source.splitlines()
    before_start = max(block.start_line - 1 - n, 0)
    before = "\n".join(lines[before_start : block.start_line - 1])
    after_end = min(block.end_line + n, len(lines))
    after = "\n".join(lines[block.end_line : after_end])
    return before, after
