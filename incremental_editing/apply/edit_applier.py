"""Applies a validated Delta IR to source text via AST-located line spans.

Only the targeted symbol's lines are replaced/removed/inserted around — the
rest of the file passes through untouched, byte for byte.
"""

import textwrap
from typing import Dict, Optional

from ..analyzer.locator import AmbiguousSymbolError, index_symbols
from ..analyzer.locator import find_symbol as _find_symbol
from ..delta.schema import DeltaIR


class ApplyError(Exception):
    pass


def _reindent(content: str, indent: int) -> str:
    """Force `content` to the given indentation column, regardless of
    whatever indentation the model returned it with. A class method target
    needs its `def` back at, say, column 4 -- trusting the model to get
    that right produced silently-wrong output (a de-indented method that
    Python parses fine but treats as a different, un-nested definition)."""
    dedented = textwrap.dedent((content or "").rstrip("\n")) + "\n"
    if indent == 0:
        return dedented
    prefix = " " * indent
    return "\n".join(prefix + line if line.strip() else line for line in dedented.splitlines()) + "\n"


def apply_delta(
    source: str, delta: DeltaIR, language: str = "python", prefer_lines: Optional[Dict[str, int]] = None
) -> str:
    lines = source.splitlines(keepends=True)
    symbols = index_symbols(source, language)
    prefer_lines = prefer_lines or {}

    actions = []
    for op in delta.operations:
        t = op.target
        try:
            if op.operation in ("REPLACE", "DELETE"):
                allow_delegate = op.operation == "REPLACE"
                sym = _find_symbol(
                    symbols, t.symbol_type, t.symbol_name, prefer_lines.get(t.symbol_name), source, allow_delegate
                )
                if sym is None:
                    raise ApplyError(f"cannot apply {op.operation}: target '{t.symbol_name}' not found")
                actions.append((sym.start_line, "range", sym.start_line, sym.end_line, sym.indent, op))
            elif op.operation == "INSERT":
                if t.anchor:
                    anchor = _find_symbol(symbols, t.symbol_type, t.anchor, prefer_lines.get(t.anchor), source)
                    if anchor is None:
                        raise ApplyError(f"cannot apply INSERT: anchor '{t.anchor}' not found")
                    pos = anchor.end_line
                    indent = anchor.indent
                else:
                    pos = len(lines)
                    indent = 0
                actions.append((pos, "insert", pos, None, indent, op))
            else:
                raise ApplyError(f"unsupported operation '{op.operation}'")
        except AmbiguousSymbolError as e:
            raise ApplyError(str(e)) from e

    actions.sort(key=lambda a: a[0], reverse=True)

    for _, kind, a, b, indent, op in actions:
        if op.operation == "DELETE":
            end = b
            while end < len(lines) and lines[end].strip() == "":
                end += 1
            lines[a - 1 : end] = []
            continue
        content = _reindent(op.content, indent)
        if kind == "range":
            lines[a - 1 : b] = [content]
        elif kind == "insert":
            lines[a:a] = ["\n", content]

    return "".join(lines)
