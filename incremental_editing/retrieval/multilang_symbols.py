"""Multi-language symbol + call extraction via Tree-sitter -- extends the
Python-`ast`-only extraction in `repo_index.py`/`dependency_graph.py` to any
language `tree-sitter-language-pack` supports.

Node type names differ per grammar, so a symbol is recognized generically:
any node type containing "function"/"method" is function-like, "class"/
"struct"/"interface" is class-like. Deliberate tradeoff: broad coverage, no
per-language maintenance, at the cost of missing some constructs (Rust's
impl_item, C++ operator overloads, ...).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tree_sitter_language_pack import detect_language_from_path, get_parser


@dataclass
class RawSymbol:
    name: str
    symbol_type: str
    start_line: int
    end_line: int
    start_column: int = 0


def detect_language(filename: str) -> Optional[str]:
    return detect_language_from_path(filename)


def _classify(node_type: str) -> Optional[str]:
    t = node_type.lower()
    is_decl_shaped = "declaration" in t or "definition" in t or "specifier" in t or "item" in t
    if not is_decl_shaped and t not in ("function", "method", "class"):
        return None
    if "function" in t or "method" in t:
        return "function"
    if "class" in t or "struct" in t or "interface" in t:
        return "class"
    return None


_symbol_concept_cache: dict = {}


def language_has_symbol_concept(language: str) -> bool:
    """True if this language's grammar defines any node kind _classify()
    recognizes as function/class-shaped -- checked against the grammar's
    own node kinds, not a hardcoded per-language list, so it's correct for
    any language tree-sitter-language-pack supports.

    Lets a caller (e.g. STRUCTURED_EDIT's classifier, which can only ever
    answer "escalate" for a language with no such concept) skip a wasted
    classification call for languages like markdown/json/yaml/css."""
    if language not in _symbol_concept_cache:
        try:
            lang_obj = get_parser(language).language
            _symbol_concept_cache[language] = any(
                _classify(lang_obj.node_kind_for_id(i) or "") is not None for i in range(lang_obj.node_kind_count)
            )
        except Exception:
            _symbol_concept_cache[language] = True
    return _symbol_concept_cache[language]


def _find_name_node(node):
    """Fallback for a grammar without a direct "name" field -- e.g. C/C++'s
    function_definition nests its identifier inside a function_declarator
    child, so child_by_field_name("name") always returns None.

    Pre-order DFS for the first descendant whose type ends in "identifier".
    Safe because a definition's name always appears in source order before
    its parameter list and body."""
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
    no-cross-module-resolution approach as `dependency_graph.py` for
    Python, via a generic node-type match ("call"/"invocation")."""
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
    """(start_line, end_line) 1-indexed ranges for top-level import
    statements. Only root-level children are considered -- imports are
    always direct root children in every grammar checked, which also
    avoids double-counting a language's nested per-import child nodes
    (e.g. Go's grouped import_declaration)."""
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    return [
        (child.start_point[0] + 1, child.end_point[0] + 1)
        for child in tree.root_node.children
        if "import" in child.type.lower()
    ]
