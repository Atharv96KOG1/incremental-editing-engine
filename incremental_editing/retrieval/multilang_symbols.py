"""Multi-language symbol + call extraction via Tree-sitter -- extends the
Python-`ast`-only extraction in `repo_index.py`/`dependency_graph.py` to any
language `tree-sitter-language-pack` supports, same tool that already made
`validation/syntax.py` multi-language.

Node type names differ per grammar (Java's method_declaration vs Go's
function_declaration vs JS's method_definition, Python's own function_
definition, ...), so instead of one hand-written query per language, a
symbol is recognized generically: any node whose type name contains
"function"/"method" is function-like, one containing "class"/"struct"/
"interface" is class-like. Verified empirically against Java, JavaScript,
and Go grammars. Deliberate tradeoff: broad coverage, no per-language
maintenance, at the cost of missing constructs this substring match
doesn't catch (Rust's impl_item, C++ operator overloads, ...).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tree_sitter_language_pack import detect_language_from_path, get_parser


@dataclass
class RawSymbol:
    name: str
    symbol_type: str  # "function" | "class"
    start_line: int
    end_line: int
    start_column: int = 0  # 0-indexed column of the symbol's own start line, for re-indentation


def detect_language(filename: str) -> Optional[str]:
    return detect_language_from_path(filename)


def _classify(node_type: str) -> Optional[str]:
    t = node_type.lower()
    # A call site like Java's method_invocation also contains "method" --
    # restrict to definition/declaration-shaped node types so a *call* to
    # helper() is never mistaken for a *definition* of it. "item" covers
    # Rust (function_item/struct_item -- neither contains "declaration"/
    # "definition"/"specifier" at all, verified via a direct grammar
    # dump). Bare "function"/"method"/"class" (no suffix at all) covers
    # Ruby (its function/class definitions really are just "method"/
    # "class" node kinds) -- safe to allow here because the actual parsed
    # *node* is additionally required (see extract_symbols's walk()) to
    # be named with at least one child before this is ever consulted,
    # which is what excludes the bare keyword literal token every one of
    # these grammars separately exposes under this exact same type name
    # (verified: that literal token is always unnamed with zero children,
    # never a real definition).
    is_decl_shaped = "declaration" in t or "definition" in t or "specifier" in t or "item" in t
    if not is_decl_shaped and t not in ("function", "method", "class"):
        return None
    if "function" in t or "method" in t:
        return "function"
    if "class" in t or "struct" in t or "interface" in t:
        return "class"
    return None


# Per-language, process-lifetime: a grammar's set of node kinds never
# changes mid-process, so this is worth caching the same way
# vector_retriever.py caches embeddings -- language_has_symbol_concept
# gets called on every STRUCTURED_EDIT request.
_symbol_concept_cache: dict = {}


def language_has_symbol_concept(language: str) -> bool:
    """True if this language's own tree-sitter grammar defines ANY node
    kind _classify() would recognize as function/class-shaped -- a real,
    verifiable structural fact about the FORMAT (python: yes, even for
    an empty file, since "add a new function" is still a valid INSERT;
    markdown/json/yaml/csv/html/css/ini/dotenv: no, full stop, regardless
    of what any request asks for) -- not a guess about one specific
    file's current content, and not a hardcoded per-language list
    (checked directly against the grammar's own node kinds instead, so
    it stays correct for every language tree-sitter-language-pack
    supports without maintaining one).

    Real waste this exists to let a caller avoid: STRUCTURED_EDIT's
    classification call (its whole premise is "REPLACE/INSERT/DELETE on
    a function/class, or escalate") can only ever answer "escalate" for
    a language with no such concept at all -- every single time -- so a
    real request to add one line to a 30-line README paid a full
    classification call (~1045 tokens for the system prompt alone)
    before ever reaching the generation call that could actually help."""
    if language not in _symbol_concept_cache:
        try:
            lang_obj = get_parser(language).language
            _symbol_concept_cache[language] = any(
                _classify(lang_obj.node_kind_for_id(i) or "") is not None for i in range(lang_obj.node_kind_count)
            )
        except Exception:
            _symbol_concept_cache[language] = True  # unknown -- don't assume incapable, fall through normally
    return _symbol_concept_cache[language]


def _find_name_node(node):
    """Fallback for a grammar that doesn't expose its identifier as a
    direct "name" field -- C/C++'s function_definition is the real case
    this closes: the identifier is nested inside its own function_declarator
    child, two levels down, with no "name" field on function_definition
    itself at all, so child_by_field_name("name") always returned None and
    the symbol was silently dropped, every time, for every C/C++ function.

    Pre-order DFS for the first descendant whose type ends in
    "identifier" (covers identifier/type_identifier/field_identifier/...
    across every grammar checked) -- general, not per-language, and safe
    because a definition's own name always appears in source order before
    its parameter list and body (verified for C/C++: the declarator's
    identifier is walked before its own parameter_list, and the whole
    declarator before the function's body)."""
    for child in node.children:
        if child.type.endswith("identifier"):
            return child
        found = _find_name_node(child)
        if found is not None:
            return found
    return None


def extract_symbols(source: str, language: str) -> List[RawSymbol]:
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    symbols: List[RawSymbol] = []

    def walk(node) -> None:
        # Unnamed, childless nodes are keyword/punctuation *tokens*, not
        # real definitions -- e.g. the literal "class"/"def" keyword
        # several grammars (Python, JS, Ruby, C++, ...) separately expose
        # as its own leaf node sharing the exact same bare type name a
        # real class/method definition container also uses. Excluding
        # those here is what makes it safe for _classify to accept those
        # bare names at all (see its own comment) without misfiring on
        # the keyword token itself.
        if node.is_named and node.child_count > 0:
            symbol_type = _classify(node.type)
            if symbol_type:
                name_node = node.child_by_field_name("name") or _find_name_node(node)
                if name_node is not None:
                    symbols.append(
                        RawSymbol(
                            name_node.text.decode("utf-8"),
                            symbol_type,
                            node.start_point[0] + 1,
                            node.end_point[0] + 1,
                            node.start_point[1],
                        )
                    )
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return symbols


def extract_called_names(source: str, language: str) -> List[str]:
    """Every callee name referenced in source, name-based only -- same
    no-cross-module-resolution approach `dependency_graph.py` already uses
    for Python, extended to any language via the same generic node-type
    match ("call"/"invocation" in the node type)."""
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    names: List[str] = []

    def walk(node) -> None:
        t = node.type.lower()
        if "call" in t or "invocation" in t:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                names.append(name_node.text.decode("utf-8"))
            else:
                func_node = node.child_by_field_name("function")
                if func_node is not None:
                    names.append(func_node.text.decode("utf-8").rsplit(".", 1)[-1])
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return names


def extract_import_lines(source: str, language: str) -> List[Tuple[int, int]]:
    """(start_line, end_line) 1-indexed inclusive ranges for every top-level
    import statement -- import statements are always direct children of
    the file's root node in every grammar checked (Java/JS/Go), so unlike
    symbols/calls this doesn't need a whole-tree walk: only root-level
    children are considered, which also keeps a language's nested
    per-import child nodes (e.g. Java's leaf "import" token, Go's
    import_spec entries inside one grouped import_declaration) from being
    picked up as separate, overlapping ranges."""
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    return [
        (child.start_point[0] + 1, child.end_point[0] + 1)
        for child in tree.root_node.children
        if "import" in child.type.lower()
    ]
