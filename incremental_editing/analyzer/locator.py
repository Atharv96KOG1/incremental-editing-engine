"""Local change localization: which symbol(s) a request likely targets --
no LLM call, heuristic matching against symbol names/docstrings.

Falls back to "no confident match" rather than guessing, so the context
builder falls back to the whole file instead of dropping needed context.
"""

import ast
import difflib
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..retrieval.dependency_graph import _called_names as _ast_called_names
from ..retrieval.multilang_symbols import extract_called_names, extract_symbols


class AmbiguousSymbolError(Exception):
    """Name defined more than once. Picking one match blind risks editing
    dead/shadowed code instead of the live definition -- callers must
    refuse, not guess."""


@dataclass
class SymbolInfo:
    name: str
    symbol_type: str
    start_line: int
    end_line: int
    docstring_first_line: Optional[str]
    indent: int
    parent_class: Optional[str] = None


def _index_python_symbols(tree: ast.AST) -> List[SymbolInfo]:
    """Recursive (not ast.walk's flat BFS) so each symbol carries its
    enclosing class -- needed to disambiguate same-named methods across
    different classes."""
    symbols: List[SymbolInfo] = []

    def walk(node: ast.AST, parent_class: Optional[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbol_type = "class" if isinstance(child, ast.ClassDef) else "function"
                start = child.decorator_list[0].lineno if child.decorator_list else child.lineno
                doc = ast.get_docstring(child)
                first_line = doc.strip().splitlines()[0] if doc else None
                symbols.append(
                    SymbolInfo(child.name, symbol_type, start, child.end_lineno, first_line, child.col_offset, parent_class)
                )
                walk(child, child.name if isinstance(child, ast.ClassDef) else parent_class)
            else:
                walk(child, parent_class)

    walk(tree, None)
    return symbols


def index_symbols(source: str, language: str = "python") -> List[SymbolInfo]:
    """Python uses the ast-based path (docstrings, decorator-aware start
    lines, class scope); any other Tree-sitter language goes through
    multilang_symbols.extract_symbols instead, with docstring_first_line/
    parent_class always None. Unrecognized/unparseable language returns []
    rather than mis-parsing as Python."""
    if language == "python":
        return _index_python_symbols(ast.parse(source))

    if language is None:
        return []
    try:
        raw_symbols = extract_symbols(source, language)
    except Exception:
        return []
    return [
        SymbolInfo(r.name, r.symbol_type, r.start_line, r.end_line, None, r.start_column) for r in raw_symbols
    ]


def looks_like_python(source: str) -> bool:
    """Stricter than a bare ast.parse() check: requires at least one real
    function/class/import, not just parse success. Catches real Python
    files with misleading extensions (detect_language() is extension-only)
    while not misdetecting plain CSV/tabular data as Python (each row also
    parses as a valid tuple expression)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
        for node in tree.body
    )


def _is_trivial_delegate(source: str, wrapper: "SymbolInfo", impl: "SymbolInfo") -> bool:
    """True when `wrapper` (module-level function) is just a one-line
    facade calling `impl` (same name, method on a class), e.g.
    `def divide(a, b): return Calculator.divide(a, b)`. Detected
    structurally (tiny body + call referencing impl's class+name),
    covering OOP-refactor shims kept for backward compat -- editing the
    name should hit the real implementation, not the shim."""
    if impl.parent_class is None or wrapper.parent_class is not None:
        return False
    lines = source.splitlines()
    body_lines = [l for l in lines[wrapper.start_line - 1 : wrapper.end_line] if l.strip()]
    if len(body_lines) < 2 or len(body_lines) > 3:
        return False
    body_text = "\n".join(body_lines[1:])
    return bool(re.search(rf"\b{re.escape(impl.parent_class)}\s*\.\s*{re.escape(impl.name)}\s*\(", body_text))


def find_symbol(
    symbols: List[SymbolInfo],
    symbol_type: str,
    symbol_name: str,
    prefer_line: Optional[int] = None,
    source: Optional[str] = None,
    allow_delegate_resolution: bool = True,
) -> Optional[SymbolInfo]:
    """Single symbol matching (symbol_type, symbol_name), or None. Raises
    AmbiguousSymbolError on multiple matches, unless `prefer_line` pins one
    occurrence already resolved earlier, or the two matches are a class
    method + its trivial module-level delegate (_is_trivial_delegate) --
    then the real method wins.

    `allow_delegate_resolution` should be False for DELETE: auto-resolving
    to the implementation there would leave the shim calling a now-missing
    method, so DELETE must ask instead of guessing."""
    matches = [s for s in symbols if s.symbol_type == symbol_type and s.name == symbol_name]
    if len(matches) > 1:
        if prefer_line is not None:
            preferred = [m for m in matches if m.start_line == prefer_line]
            if len(preferred) == 1:
                return preferred[0]
        if source is not None and allow_delegate_resolution and len(matches) == 2:
            a, b = matches
            for wrapper, impl in ((a, b), (b, a)):
                if _is_trivial_delegate(source, wrapper, impl):
                    return impl
        lines = ", ".join(str(m.start_line) for m in matches)
        raise AmbiguousSymbolError(
            f"{symbol_type} '{symbol_name}' is defined {len(matches)} times (lines {lines}) "
            "-- resolve the duplicate before editing"
        )
    return matches[0] if matches else None


_DEFINED_NAME_WRAPPER_CLASS = "__IEEWrapper__"


def defined_symbol_name(content: Optional[str], language: str = "python") -> Optional[str]:
    """Name of the single function/class `content` actually defines, or
    None if empty/unparseable/not exactly one. The model's declared
    symbol_name is advisory only -- this is ground truth, used to correct
    a mismatched INSERT name and to catch an INSERT that would create a
    duplicate definition of an existing name."""
    if not content:
        return None
    if language == "python":
        try:
            tree = ast.parse(textwrap.dedent(content))
        except SyntaxError:
            return None
        top_level = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        return top_level[0].name if len(top_level) == 1 else None
    if language is None:
        return None
    try:
        syms = extract_symbols(content, language)
    except Exception:
        return None
    if len(syms) == 1:
        return syms[0].name
    if not syms:
        try:
            wrapped = extract_symbols(f"class {_DEFINED_NAME_WRAPPER_CLASS} {{\n{content}\n}}", language)
        except Exception:
            return None
        real = [s for s in wrapped if s.name != _DEFINED_NAME_WRAPPER_CLASS]
        return real[0].name if len(real) == 1 else None
    return None


_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with",
    "is", "are", "be", "it", "this", "that", "as", "at", "by", "from",
    "so", "do", "does", "did", "has", "have", "had", "can", "will",
    "would", "should", "could", "its", "into", "than", "then", "there",
    "here", "you", "your", "we", "our", "i", "me", "my",
}

_GENERIC_LEADING_VERBS = {
    "add", "remove", "delete", "update", "create", "make", "build", "fix",
    "change", "modify", "edit", "insert", "append", "get", "set", "check",
    "verify", "test", "write", "read", "load", "save", "clear", "reset", "run",
}


def _words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower())) - _STOPWORDS


def _local_called_by_counts(
    source: str, symbols: List[SymbolInfo], language: str
) -> Dict[Tuple[str, int], int]:
    """How many other symbols in this file call each symbol -- a free (no
    LLM/embeddings) tiebreak signal checked before the paid semantic
    tiebreak below. Keyed by (name, start_line), not bare name: two
    methods sharing a name across different classes (e.g. Trig.cos vs
    Hyperbolic.cos) must be counted separately, or the tiebreak this
    exists for can never actually distinguish them. A caller inside a
    class crediting a same-named callee prefers that class's own
    occurrence (self.cos() means this class's cos) over a same-named
    method elsewhere; a module-level caller or one with no same-class
    match credits every occurrence, same as before.

    Class-container symbols are skipped as callers: index_symbols gives a
    class's own span every one of its methods' lines, so treating it as a
    caller would double-count each method's calls as a second, phantom
    caller distinct from the method itself."""
    lines = source.splitlines()
    by_name: Dict[str, List[SymbolInfo]] = {}
    for s in symbols:
        by_name.setdefault(s.name, []).append(s)
    known_names = set(by_name)
    called_by: Dict[Tuple[str, int], set] = {(s.name, s.start_line): set() for s in symbols}

    for sym in symbols:
        if sym.symbol_type == "class":
            continue
        snippet = "\n".join(lines[sym.start_line - 1 : sym.end_line])
        try:
            if language == "python":
                callees = _ast_called_names(ast.parse(textwrap.dedent(snippet)))
            else:
                callees = extract_called_names(snippet, language)
        except Exception:
            continue
        for callee in set(callees):
            if callee not in known_names or callee == sym.name:
                continue
            occurrences = by_name[callee]
            same_class = [o for o in occurrences if o.parent_class == sym.parent_class]
            for target in same_class or occurrences:
                called_by[(target.name, target.start_line)].add((sym.name, sym.start_line))

    return {key: len(callers) for key, callers in called_by.items()}


def _semantic_tiebreak(source: str, user_request: str, tied: List[SymbolInfo]) -> SymbolInfo:
    """Breaks a genuine score tie (e.g. same method name repeated across
    classes, near-identical bodies) via embedding similarity, with class
    name prefixed onto each body for signal. Lazy import: the only
    embeddings call in this module; any failure falls back to the first
    tied candidate."""
    try:
        from ..retrieval.vector_retriever import _embed
    except Exception:
        return tied[0]

    lines = source.splitlines()
    try:
        texts = [user_request] + [
            f"{s.parent_class + ' ' if s.parent_class else ''}{s.name}\n"
            + "\n".join(lines[s.start_line - 1 : s.end_line])
            for s in tied
        ]
        vectors = _embed(texts)
        similarities = vectors[1:] @ vectors[0]
        return tied[int(similarities.argmax())]
    except Exception:
        return tied[0]


def locate_candidates(
    source: str, user_request: str, max_candidates: int = 2, min_score: int = 1, language: str = "python"
) -> List[SymbolInfo]:
    """Top matching symbols for a request, or [] if nothing scores confidently."""
    symbols = index_symbols(source, language)
    request_lower = user_request.lower()
    request_words = _words(user_request)
    leading_word = request_words and request_lower.strip().split()[0] or None

    scored = []
    for sym in symbols:
        name_lower = sym.name.lower()
        literal_hit = bool(re.search(rf"\b{re.escape(name_lower)}\b", request_lower))
        name_overlap = len(_words(sym.name.replace("_", " ")) & request_words)
        doc_overlap = len(_words(sym.docstring_first_line) & request_words) if sym.docstring_first_line else 0
        class_hit = bool(
            sym.parent_class and re.search(rf"\b{re.escape(sym.parent_class.lower())}\b", request_lower)
        )

        is_leading_generic_verb = name_lower in _GENERIC_LEADING_VERBS and name_lower == leading_word
        corroborated = doc_overlap > 0 or name_overlap > 1 or class_hit
        if is_leading_generic_verb and literal_hit and not corroborated:
            continue

        if not literal_hit and not corroborated:
            continue

        score = 0
        if literal_hit:
            score += 5
        if class_hit:
            score += 5
        score += name_overlap
        score += doc_overlap
        if score >= min_score:
            scored.append((score, sym))

    by_name: Dict[str, List[Tuple[int, SymbolInfo]]] = {}
    for score, sym in scored:
        by_name.setdefault(sym.name, []).append((score, sym))

    local_called_by: Optional[Dict[Tuple[str, int], int]] = None
    collapsed: List[Tuple[int, SymbolInfo]] = []
    for group in by_name.values():
        if len(group) == 1:
            collapsed.append(group[0])
            continue
        top_score = max(s for s, _ in group)
        tied = [sym for s, sym in group if s == top_score]
        if len(tied) == 1:
            winner = tied[0]
        else:
            if local_called_by is None:
                local_called_by = _local_called_by_counts(source, symbols, language)
            by_callers = sorted({local_called_by.get((s.name, s.start_line), 0) for s in tied}, reverse=True)
            top_callers = by_callers[0]
            free_winners = (
                [s for s in tied if local_called_by.get((s.name, s.start_line), 0) == top_callers]
                if top_callers > 0
                else tied
            )
            winner = free_winners[0] if len(free_winners) == 1 else _semantic_tiebreak(source, user_request, tied)
        collapsed.append((top_score, winner))

    collapsed.sort(key=lambda pair: pair[0], reverse=True)
    return [sym for _, sym in collapsed[:max_candidates]]


def locate_candidates_by_body(
    source: str, user_request: str, language: str = "python", max_candidates: int = 3
) -> List[SymbolInfo]:
    """Fallback for when locate_candidates finds nothing: searches each
    symbol's BODY text for a request word, not just name/docstring --
    catches targets that only appear inside a function (e.g. a dict
    literal), which name-only matching misses entirely.

    Conservative: whole-word matches only (not find_delete_candidates'
    broad substring, since this feeds normal REPLACE/INSERT context, not
    an auto-applied delete), ranked by distinct-word hit count and capped.
    Uses `_words()` tokenization (splits on non-alphanumeric runs) rather
    than `\\bword\\b`, since `_` isn't a real boundary for regex but code
    identifiers (snake_case/SCREAMING_SNAKE_CASE) need it to be."""
    target_words = {w for w in _words(user_request) - _GENERIC_LEADING_VERBS - _GENERIC_TYPE_WORDS if len(w) >= 3}
    if not target_words:
        return []
    symbols = index_symbols(source, language)
    lines = source.splitlines()
    scored = []
    for sym in symbols:
        body = "\n".join(lines[sym.start_line - 1 : sym.end_line])
        hits = len(target_words & _words(body))
        if hits:
            scored.append((hits, sym))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    def _contains(outer: SymbolInfo, inner: SymbolInfo) -> bool:
        return (
            outer is not inner
            and outer.start_line <= inner.start_line
            and outer.end_line >= inner.end_line
            and (outer.start_line, outer.end_line) != (inner.start_line, inner.end_line)
        )

    filtered = [(hits, sym) for hits, sym in scored if not any(_contains(sym, other) for _, other in scored)]
    return [sym for _, sym in filtered[:max_candidates]]


_DELETE_VERBS = {"remove", "delete"}
_GENERIC_TYPE_WORDS = {"function", "method", "class", "def", "file"}


def is_delete_intent(user_request: str) -> bool:
    words = user_request.lower().strip().split()
    return bool(words) and words[0] in _DELETE_VERBS


def is_whole_file_delete_target(file: str, user_request: str) -> bool:
    """True when a delete/remove request means the file itself, matched
    against the file's real basename/stem (never a guessed keyword list).
    Only called after find_delete_candidates/find_multi_delete_targets
    found nothing inside the file, so a real symbol name always wins
    first.

    Deliberately does NOT drop _GENERIC_LEADING_VERBS/_GENERIC_TYPE_WORDS
    here (unlike find_delete_candidates) -- doing so would let "delete the
    add function" (a real symbol name that's also a generic verb) misfire
    as a whole-file delete. Every remaining word must match the filename
    or this returns False."""
    from pathlib import PurePosixPath

    words = user_request.lower().strip().split()
    if not words or words[0] not in _DELETE_VERBS:
        return False
    target_words = _words(user_request) - {words[0], "file"}
    if not target_words:
        return True
    base = PurePosixPath(file).name.lower()
    stem = PurePosixPath(file).stem.lower()
    return all(w in base or w in stem for w in target_words)




def target_file_for_conversion(file: str, target_extension: str) -> str:
    """SHA.go -> SHA.py: same directory and base name, extension swapped
    to the target the model itself supplied in its escalate response --
    so a conversion request produces a normal-looking sibling file
    instead of overwriting the original."""
    from pathlib import PurePosixPath

    return str(PurePosixPath(file).with_suffix(f".{target_extension.lstrip('.')}"))


def find_delete_candidates(source: str, user_request: str, language: str = "python") -> List[SymbolInfo]:
    """Every symbol whose name contains a real target word from a
    remove/delete request -- deliberately broad substring search (not
    locate_candidates' scored path), since for DELETE picking wrong
    silently costs more than asking. Returns [] if no target word
    survives (verb/generic-type-noun stripped) or nothing matches.

    Words under 3 chars excluded (would substring-match almost anything).
    A name defined multiple times surfaces as one candidate per
    occurrence (own start_line/end_line), so a confirmed pick can resolve
    the exact occurrence via find_symbol's prefer_line instead of hitting
    AmbiguousSymbolError again.

    Falls back to fuzzy (edit-distance) matching against real symbol
    names when substring search finds nothing -- catches typos ("remove
    substract") cheaply instead of escalating to a full LLM regeneration.
    Still only matches real indexed names, and still refuses via the
    picker if more than one fuzzy match is close enough."""
    target_words = {w for w in _words(user_request) - _GENERIC_LEADING_VERBS - _GENERIC_TYPE_WORDS if len(w) >= 3}
    if not target_words:
        return []
    symbols = index_symbols(source, language)
    exact = [sym for sym in symbols if any(word in sym.name.lower() for word in target_words)]
    if exact:
        return exact

    all_names = {sym.name.lower() for sym in symbols}
    fuzzy_matches: set = set()
    for word in target_words:
        fuzzy_matches.update(difflib.get_close_matches(word, all_names, n=3, cutoff=0.75))
    return [sym for sym in symbols if sym.name.lower() in fuzzy_matches]


_DELETE_LIST_SEPARATOR_RE = re.compile(r",|\band\b|&|;", re.IGNORECASE)


def find_multi_delete_targets(source: str, user_request: str, language: str = "python") -> Optional[List[SymbolInfo]]:
    """"delete tan, sec and cos" names three distinct targets --
    find_delete_candidates' broad substring search would instead match
    all similarly-named symbols as one ambiguous picker.

    Splits on comma/and/&/; into phrases; returns a result only when
    EVERY phrase resolves to exactly one EXACT, unambiguous symbol match
    -- strict all-or-nothing. Returns None (not []) if that bar isn't
    met, so the caller falls back to the normal single-target path."""
    words = user_request.strip().split()
    rest = " ".join(words[1:]) if words and words[0].lower() in _DELETE_VERBS else user_request
    phrases = [p.strip() for p in _DELETE_LIST_SEPARATOR_RE.split(rest) if p.strip()]
    if len(phrases) < 2:
        return None

    symbols = index_symbols(source, language)
    resolved: List[SymbolInfo] = []
    for phrase in phrases:
        phrase_words = _words(phrase) - _GENERIC_LEADING_VERBS - _GENERIC_TYPE_WORDS
        exact_matches = [s for s in symbols if s.name.lower() in phrase_words]
        distinct_names = {m.name for m in exact_matches}
        if len(distinct_names) != 1:
            return None
        name_matches = [m for m in exact_matches if m.name == next(iter(distinct_names))]
        if len(name_matches) != 1:
            return None
        resolved.append(name_matches[0])
    return resolved


_RENAME_RE = re.compile(
    r"^(?:rename|replace)\b"
    r"(?:\s+(?:the\s+)?name\s+of)?"
    r"(?:\s+(?:the\s+)?(?:function|method|class|variable))?"
    r"(?:\s+of)?"
    r"\s+[\"']?(?P<old>[A-Za-z_][A-Za-z0-9_]*)[\"']?"
    r"\s+(?:to|with|by)\s+"
    r"[\"']?(?P<new>[A-Za-z_][A-Za-z0-9_]*)[\"']?\s*[.!]?\s*$",
    re.IGNORECASE,
)


def find_rename_target(source: str, user_request: str, language: str = "python") -> Optional[Tuple[str, str]]:
    """(old_name, new_name) when the request is confidently a rename of
    one function/class -- zero LLM calls, since old_name must be a real,
    unambiguous (defined once) symbol definition, never a bare word
    match. A non-definition rename (e.g. a variable) falls through to the
    normal LLM-driven path instead."""
    match = _RENAME_RE.match(user_request.strip())
    if not match:
        return None
    old_name, new_name = match.group("old"), match.group("new")
    symbols = index_symbols(source, language)
    matches = [s for s in symbols if s.name == old_name]
    if len(matches) != 1:
        return None
    return old_name, new_name


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CASE_TRANSITION_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _match_case(replacement: str, like: str) -> str:
    if like.isupper():
        return replacement.upper()
    if like[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement[:1].lower() + replacement[1:]


def _rename_subword(identifier: str, old_name: str, new_name: str) -> Optional[str]:
    """If `old_name` matches one of `identifier`'s sub-words (or a
    concatenated run of them), split convention-aware (snake_case on '_',
    camelCase/PascalCase at case transitions), replaces just that span
    cased to match, leaves the rest untouched. Shortest match tried
    first. None if old_name isn't found.

    Multi-word spans matter: "OpenAI" inside "buildOpenAIClient" splits
    into "Open"+"AI" sub-words, so old_name="openai" only matches when
    both are tried together as one span."""
    if "_" in identifier:
        parts = identifier.split("_")
        for span_len in range(1, len(parts) + 1):
            for i in range(len(parts) - span_len + 1):
                span_parts = parts[i : i + span_len]
                if "".join(span_parts).lower() == old_name.lower():
                    replacement = _match_case(new_name, span_parts[0] or new_name)
                    return "_".join(parts[:i] + [replacement] + parts[i + span_len :])
        return None

    starts = [0] + [m.start() for m in _CASE_TRANSITION_RE.finditer(identifier)]
    boundaries = starts + [len(identifier)]
    n = len(boundaries) - 1
    for span_len in range(1, n + 1):
        for i in range(n - span_len + 1):
            start, end = boundaries[i], boundaries[i + span_len]
            word = identifier[start:end]
            if word.lower() == old_name.lower():
                return identifier[:start] + _match_case(new_name, word) + identifier[end:]
    return None


def rename_with_subword_fallback(source: str, old_name: str, new_name: str) -> str:
    """Renames every whole-word occurrence of old_name; if that finds
    nothing, falls back to renaming it as a sub-word inside a longer
    camelCase/PascalCase identifier (e.g. "OpenAI" inside
    "buildOpenAIClient") -- a plain \\bold_name\\b regex has no concept
    of a case transition as a word boundary, so it misses these
    otherwise.

    Convention-based, not per-language: camelCase/PascalCase/snake_case
    mix across every language this project supports, reusing
    bm25_retriever.py's existing camelCase-splitting convention."""
    whole_word_re = re.compile(rf"\b{re.escape(old_name)}\b")
    new_source = whole_word_re.sub(new_name, source)
    if new_source != source:
        return new_source

    def _replace_identifier(m: "re.Match") -> str:
        renamed = _rename_subword(m.group(0), old_name, new_name)
        return renamed if renamed is not None else m.group(0)

    return _IDENTIFIER_RE.sub(_replace_identifier, source)
