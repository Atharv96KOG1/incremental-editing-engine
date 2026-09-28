"""Cheap, local change localization: which symbol(s) in a file does a
request likely target -- no LLM call, pure heuristic matching against
symbol names and docstrings (doc sections 12/13).

Deliberately simple for phase 4: falls back to "no confident match" rather
than guessing, so the context builder can fall back to the whole file
instead of silently dropping context the model actually needs.
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
    """The file defines the same top-level name more than once. Python only
    keeps the last definition at runtime, so silently picking one match
    (as a plain first-hit search would) risks editing dead, shadowed code
    while the live definition stays untouched. Callers should refuse and
    surface this rather than guess."""


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
    """Recursive (not ast.walk's flat BFS) so each symbol can be tagged
    with its nearest enclosing class -- needed so two methods that
    legitimately share a bare name in different classes (completely
    normal Python, not a bug) can eventually be told apart, instead of
    only ever being addressable by a name that's ambiguous the moment a
    file has more than one class."""
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
    """`language` selects how source gets parsed: "python" uses the
    ast-based path below (docstrings, decorator-aware start lines, class
    scope -- more precise than the generic path, so kept as its own
    branch rather than folded into Tree-sitter too). Any other
    Tree-sitter language name goes through `multilang_symbols.
    extract_symbols` instead -- no docstring or class-scope tracking
    there (no single cross-language convention for either), so
    `docstring_first_line`/`parent_class` are always None for those. An
    unrecognized/unparseable language returns [] rather than guessing at
    Python syntax (the earlier bug this fixes: a .java file's `public
    class Foo {` isn't valid Python and used to raise SyntaxError here)."""
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
    """A real failure this catches: a file literally named "new.db",
    containing a genuine Python script (`def create_database(): ...
    cur.executescript(...)`) that builds a SQLite database -- naming the
    *setup script* after the database it creates, a real, observed
    naming choice. detect_language() is extension-only, so a ".db"
    extension always comes back as language=None regardless of content,
    losing all symbol extraction: REPLACE on a real function inside it
    then fails with "not found" (index_symbols(source, None) == []), and
    LOCALIZE is forced to send the whole file instead of the one
    relevant function. A real ast.parse() success is a much stronger
    signal than a mismatched filename extension.

    Deliberately stricter than a bare ast.parse() check: parsing alone
    isn't enough -- a CSV's rows ("id,name,salary") are *also*
    syntactically valid Python (each line parses as a harmless, unused
    tuple expression), which would misdetect every plain CSV as Python
    and silently break the whole-file fallback that already handles CSV
    correctly. Requiring at least one real function/class/import --
    something an actual Python file almost always has and mere tabular
    data never does -- is what tells the two apart."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
        for node in tree.body
    )


def _is_trivial_delegate(source: str, wrapper: "SymbolInfo", impl: "SymbolInfo") -> bool:
    """True when `wrapper` (a module-level function) is nothing but a
    one-line facade calling `impl` (the same name, but a method on some
    class) -- e.g. `def divide(a, b): return Calculator.divide(a, b)`.
    Detected structurally (tiny body, a call expression referencing
    impl's own class + the shared name) -- never by a specific name,
    file, or class. Real pattern this exists for: a broad "use OOP" /
    "wrap this in a class" refactor introduces a class while keeping
    every old flat function as a backward-compatible shim, so nothing
    already importing the flat API breaks. Editing "divide" from then on
    should mean the real implementation, not its forwarding shim."""
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
    """The single symbol matching (symbol_type, symbol_name), or None if
    there's no match. Raises AmbiguousSymbolError if the name is defined
    more than once (Python only keeps the last definition at module
    scope, and even where that's not the concern -- e.g. the same method
    name in two different classes -- a bare name alone can't say which
    one is meant) UNLESS:
    - `prefer_line` is given and matches exactly one of the candidates'
      start_line: the ambiguity was already resolved earlier
      (locate_candidates picked a specific occurrence before the LLM was
      ever called), so re-raising here would refuse an edit whose target
      was never actually in doubt; or
    - `source` is given, `allow_delegate_resolution` is True, and the two
      matches are a class method plus a trivial module-level delegate to
      it (_is_trivial_delegate) -- then the class method (the real
      implementation) is returned, since a one-line forwarding shim has
      no independent logic an edit could mean instead. Only ever applies
      to exactly 2 matches in exactly this shape; 3+ matches, or 2 that
      aren't a delegate pair, still raise -- this is a narrow, structural
      exception, not a general "just pick one" fallback.

    `allow_delegate_resolution` defaults True for REPLACE/INSERT-anchor
    resolution (editing the real implementation behind a shim is safe
    and almost always what's meant) but callers resolving a DELETE
    target should pass False: silently deleting just the implementation
    would leave its wrapper calling a method that no longer exists --
    a real regression the auto-resolution that's safe everywhere else
    would introduce here, so DELETE keeps asking instead of guessing."""
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
    """The name of the single function/class `content` actually defines,
    or None if it's empty, unparseable, or doesn't define exactly one.
    The model's declared symbol_name/target is advisory only -- this is
    the only source of truth for what a snippet of generated code really
    creates, used both to correct a mismatched *displayed* INSERT name
    (api/run_pipeline.py) and, more importantly, to catch an INSERT that
    would create a *second* definition of a name that already exists
    elsewhere in the file (validate_targets below) -- the exact class of
    real corruption this project has hit: a class-method version of a
    function inserted alongside an old top-level one that was never
    removed, both now silently coexisting until something tries to
    REPLACE/DELETE the name and hits an unresolvable AmbiguousSymbolError."""
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


def _local_called_by_counts(source: str, symbols: List[SymbolInfo], language: str) -> Dict[str, int]:
    """How many *other* symbols in this same file call each symbol, by
    name -- a free signal (zero LLM/embeddings calls) computed from the
    same call-extraction primitives dependency_graph.py/multilang_
    symbols.py already use repo-wide, just scoped to one file's own
    symbols. Used as a tiebreak that costs nothing before ever reaching
    for the paid semantic tiebreak below -- a symbol real code in this
    file actually calls is a stronger, free signal than an arbitrary
    pick, and most ties never need to pay for embeddings at all once
    this is checked first."""
    lines = source.splitlines()
    known_names = {s.name for s in symbols}
    called_by: Dict[str, set] = {name: set() for name in known_names}

    for sym in symbols:
        snippet = "\n".join(lines[sym.start_line - 1 : sym.end_line])
        try:
            if language == "python":
                callees = _ast_called_names(ast.parse(textwrap.dedent(snippet)))
            else:
                callees = extract_called_names(snippet, language)
        except Exception:
            continue
        for callee in set(callees):
            if callee in known_names and callee != sym.name:
                called_by[callee].add(sym.name)

    return {name: len(callers) for name, callers in called_by.items()}


def _semantic_tiebreak(source: str, user_request: str, tied: List[SymbolInfo]) -> SymbolInfo:
    """Breaks a genuine score tie (two or more symbols equally strong by
    every text heuristic -- the common shape in a large, class-heavy file:
    the same method name repeated across classes, e.g. Trig.tan and
    Hyperbolic.tan, with near-identical bodies and no literal name in the
    request to prefer one) using real semantic similarity instead of an
    arbitrary pick. The enclosing class name is prefixed onto each
    candidate's own body text specifically because near-duplicate method
    bodies otherwise carry almost no differentiating signal on their own
    -- the class each belongs to is usually what the request is actually
    about. Lazy import: this is the only place in this module that ever
    calls an embeddings API, and only when heuristics alone can't decide;
    any failure (offline, gateway down) falls back to the first tied
    candidate rather than failing the whole request over a tiebreak."""
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

    local_called_by: Optional[Dict[str, int]] = None
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
            by_callers = sorted({local_called_by.get(s.name, 0) for s in tied}, reverse=True)
            top_callers = by_callers[0]
            free_winners = [s for s in tied if local_called_by.get(s.name, 0) == top_callers] if top_callers > 0 else tied
            winner = free_winners[0] if len(free_winners) == 1 else _semantic_tiebreak(source, user_request, tied)
        collapsed.append((top_score, winner))

    collapsed.sort(key=lambda pair: pair[0], reverse=True)
    return [sym for _, sym in collapsed[:max_candidates]]


def locate_candidates_by_body(
    source: str, user_request: str, language: str = "python", max_candidates: int = 3
) -> List[SymbolInfo]:
    """Fallback localization for when locate_candidates finds nothing at
    all -- searches each symbol's own BODY text for a real content word
    from the request, instead of just its name/docstring.

    Real waste this closes: "remove the minio" meant deleting a
    `"minio": {...}` dict literal *inside* chatbot.py's `health()`
    function -- "minio" never appears in any symbol's name or docstring,
    only in that one function's body, so name-only matching found
    nothing and the request fell all the way through to a full-file
    regeneration (a ~$0.01, several-second round trip) just so the model
    could see where "minio" actually was. A file can be arbitrarily
    large while the part actually relevant to a request stays a handful
    of lines -- this project's whole premise is sending that handful,
    not the whole thing, whenever it's findable at all.

    Deliberately conservative: only whole-word matches (not
    find_delete_candidates' broad substring, since this feeds normal
    REPLACE/INSERT context selection, not an auto-applied delete target)
    against real, already-parsed content words (stopwords/generic verbs/
    type words dropped, same filtering as the delete path), ranked by
    how many distinct target words each body contains and capped to a
    small number of candidates -- a common word can't drag in half the
    file's functions the way it could an unranked, uncapped search.
    Tokenized with `_words()` (splits on any non-alphanumeric run), not
    a raw `\\bword\\b` regex -- code identifiers are routinely
    snake_case/SCREAMING_SNAKE_CASE ("MINIO_ENDPOINT"), where `_` isn't
    a real word boundary to regex's `\\b` but very much is one for how
    this project already tokenizes everywhere else."""
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
    """True when a delete/remove request means the file itself, not a
    symbol inside it -- matched the same structural way find_delete_candidates
    matches a code-level delete: against the file's own *real* name (its
    basename and extension-less stem), never a guessed keyword list.

    Only called after find_delete_candidates/find_multi_delete_targets
    already found nothing inside the file, so a real symbol name always
    wins over this. A bare "delete this file"/"delete the file" (nothing
    left but the instruction verb and the generic word "file") counts
    too -- with a single target file already selected there's nothing
    else it could mean. "delete new.db"/"remove calculator.py" also
    match, against the real path.

    Deliberately does NOT also drop _GENERIC_LEADING_VERBS/_GENERIC_TYPE_WORDS
    here the way find_delete_candidates does for symbol matching -- doing
    so let "delete the add function" (add is a real symbol name that also
    happens to be a generic imperative verb) leave zero target words and
    misfire as a whole-file delete, permanently destroying the wrong
    thing. Every remaining content word must survive and be found in the
    real filename for this to count -- if it isn't, this returns False
    and the request safely falls through to the normal (LLM-driven)
    path instead of guessing."""
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
    remove/delete request -- deliberately a broad substring search, not
    the scored/ranked path `locate_candidates` uses for every other edit,
    because for a DELETE the cost of silently picking wrong is much
    higher than the cost of asking. Returns [] if the request has no
    target word left after dropping the instruction verb and generic type
    nouns (e.g. bare "remove the function"), or if nothing matches.

    Words shorter than 3 characters are excluded from the target term --
    a bare "x" or "on" would substring-match almost any real name in a
    typical file (max, matrix, box, ...) and turn the picker into a
    meaningless wall of unrelated chips instead of a real disambiguation.

    A name defined more than once now surfaces as one candidate *per
    occurrence*, not collapsed to one -- each carries its own real
    start_line/end_line, which the picker already displays and which
    run_edit now threads through as confirm_symbol_line (find_symbol's
    prefer_line) so a confirmed delete can resolve the exact occurrence
    instead of hitting the same "defined N times" AmbiguousSymbolError a
    bare name confirmation can't get past. (Real failure this replaces:
    a file with a name genuinely defined 3 times collapsed to one picker
    chip; picking it and confirming still failed with REFERENCE_ERROR,
    since the bare name alone is inherently ambiguous no matter how many
    times it's confirmed.)

    Falls back to a fuzzy (edit-distance) match against real symbol
    names when the substring search finds nothing at all -- a typo'd
    target ("remove substract", "delete the squre function") is common
    enough in practice, and every real name in the file is already fully
    known right here, so correcting it structurally is strictly better
    than falling through to a full LLM call: real runs showed exactly
    this ("remove substract" against a file with a real `subtract`)
    escalating all the way to a 10+ second, ~$0.03 full-file
    regeneration just to find a name a spell-check-grade match already
    could, for free. Still matched against real, already-indexed names
    only -- never a guessed/hardcoded word list -- and still refuses
    (via the same multi-candidate picker) rather than silently guessing
    if more than one real name is a close enough match."""
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
    """"delete tan, sec and cos" names three distinct, deliberate targets
    -- not one ambiguous one. find_delete_candidates' broad substring
    search can't tell the two apart: it matched 10 symbols for that exact
    request (tan/tanh/atan_inverse/cos/cosin/acos_inverse/acos/sec/
    asec_inverse/sech), which would present as one confusing "which one
    did you mean?" picker instead of confirming three separate deletes.

    Splits the request on comma/and/&/; into phrases, and only returns a
    result when EVERY phrase resolves to exactly one EXACT (not
    substring) symbol name match that is itself unambiguous (not itself
    defined more than once) -- deliberately strict: this is an
    all-or-nothing fast path for genuinely unambiguous multi-target
    requests, not a general list parser. Returns None (not []) whenever
    that strict bar isn't met -- fewer than 2 phrases, any phrase with no
    exact match, more than one exact match, or a matched name that's
    itself duplicated -- so the caller falls back to the existing
    single-target substring/disambiguation path unchanged."""
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
    one function/class, zero LLM calls needed anywhere -- not even the
    normal classification call, since neither name requires seeing any
    body to be safely renamed: the request itself already says what to
    rename and what to call it, and only a real, unambiguous (defined
    exactly once) symbol name is accepted as old_name.

    Deliberately restricted to actual function/class *definitions*
    (index_symbols), not any word appearing anywhere in the source the
    way find_delete_candidates' broad substring search works for a
    destructive-but-reviewable delete -- a bare word match here would
    risk this fully-mechanical, no-human-review-until-after path
    renaming an unrelated same-named local variable or parameter in a
    completely different function purely from a regex guess. A genuine
    module-level variable rename (e.g. "replace col by column") still
    goes through the normal LLM-driven escalate path instead, where the
    model has actually seen the file before deciding old_name."""
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
    """If `old_name` matches one of `identifier`'s own sub-words, or a
    run of consecutive sub-words concatenated (case-insensitively) --
    split the same convention-aware way for snake_case (split on '_')
    and camelCase/PascalCase (split at case transitions, see
    _CASE_TRANSITION_RE) -- returns the identifier with just that span
    replaced, cased to match what it replaced, and every other sub-
    word/separator left exactly as it was. Shortest match tried first
    (a single sub-word before a multi-word span), so this never expands
    further than it needs to. None if old_name isn't found in this
    identifier at all, so the caller can leave an unrelated identifier
    untouched.

    Multi-word spans matter for a real case: "OpenAI" inside
    "buildOpenAIClient" splits into "Open"+"AI" sub-words (a trailing
    capitalized acronym is its own split point) -- old_name="openai"
    only ever matches when "Open"+"AI" are tried *together* as one
    span, not as two separate single-word attempts."""
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
    """Renames every whole-word occurrence of old_name (\\bold_name\\b,
    unchanged, exact behavior as before) -- and if that finds nothing
    at all, falls back to renaming old_name wherever it's a sub-word
    inside a longer camelCase/PascalCase identifier instead (e.g.
    "OpenAI" inside "apiKeyOpenAI" or "buildOpenAIClient").

    Real bug this closes: "use anthropic instead of openai" against a
    Go file using apiKeyOpenAI/buildOpenAIClient silently renamed
    nothing and reported a safe-but-useless no-op -- a plain
    \\bold_name\\b regex has no concept of a lowercase-to-uppercase
    transition as a word boundary, only underscores/non-alphanumerics,
    so it never found "OpenAI" packed inside those identifiers at all.

    Not a per-language rule: camelCase/PascalCase/snake_case are naming
    *conventions*, not languages -- Java, JavaScript, C#, Go, and Python
    all mix them routinely, so this generalizes across every language
    this project supports the same way, reusing the exact camelCase-
    splitting convention retrieval/bm25_retriever.py's own tokenizer
    already established rather than inventing a second one."""
    whole_word_re = re.compile(rf"\b{re.escape(old_name)}\b")
    new_source = whole_word_re.sub(new_name, source)
    if new_source != source:
        return new_source

    def _replace_identifier(m: "re.Match") -> str:
        renamed = _rename_subword(m.group(0), old_name, new_name)
        return renamed if renamed is not None else m.group(0)

    return _IDENTIFIER_RE.sub(_replace_identifier, source)
