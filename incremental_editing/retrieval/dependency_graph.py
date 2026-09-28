"""Native call graph -- calls/called_by across the whole repo (PHOENIX doc
section 1's "Dependency Index" / section 26's dependency graph), built with
Python's own `ast` module for Python symbols instead of Joern/CPG, and
Tree-sitter (`multilang_symbols.extract_called_names`) for every other
language `repo_index.py` indexes.

Name-based resolution only: a call `foo(...)` or `self.foo(...)` links to
every indexed symbol named `foo`, with no cross-module import resolution.
That's a real, stated limitation -- two unrelated `foo` functions in
different files (or languages) look identical here. Good enough to answer
"what else might this change affect" as a warning, not as a correctness
guarantee.
"""

import ast
import textwrap
from collections import defaultdict
from typing import Dict, List

from .multilang_symbols import extract_called_names
from .repo_index import RepoSymbol


def _called_names(tree: ast.AST) -> List[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


def _callees_for(sym: RepoSymbol) -> List[str]:
    if sym.language == "python":
        try:
            tree = ast.parse(textwrap.dedent(sym.source))
        except SyntaxError:
            return []
        return _called_names(tree)

    try:
        return extract_called_names(sym.source, sym.language)
    except Exception:
        return []


def build_call_graph(symbols: List[RepoSymbol]) -> Dict[str, dict]:
    """Returns {"calls": {"file::name": [callee names]}, "called_by":
    {callee name: ["file::name", ...]}}."""
    known_names = {s.name for s in symbols}
    calls: Dict[str, List[str]] = {}
    called_by: Dict[str, List[str]] = defaultdict(list)

    for sym in symbols:
        key = f"{sym.file}::{sym.name}"
        callees = sorted({n for n in _callees_for(sym) if n in known_names and n != sym.name})
        calls[key] = callees
        for callee in callees:
            called_by[callee].append(key)

    return {"calls": calls, "called_by": dict(called_by)}
