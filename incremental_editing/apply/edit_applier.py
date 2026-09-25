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

    # Resolve every span against the *original* symbol index first, then apply
    # bottom-to-top so an earlier edit never shifts a later target's line numbers.
    actions = []
    for op in delta.operations:
        t = op.target
        try:
            if op.operation in ("REPLACE", "DELETE"):
                # DELETE never auto-resolves a delegate pair -- removing
                # just the implementation would leave its wrapper calling
                # a method that no longer exists.
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
                    pos = anchor.end_line  # insert right after anchor's last line
                    indent = anchor.indent  # new sibling symbol matches the anchor's nesting level
                else:
                    pos = len(lines)  # end of file
                    indent = 0
                actions.append((pos, "insert", pos, None, indent, op))
            else:
                raise ApplyError(f"unsupported operation '{op.operation}'")
        except AmbiguousSymbolError as e:
            raise ApplyError(str(e)) from e

    actions.sort(key=lambda a: a[0], reverse=True)

    for _, kind, a, b, indent, op in actions:
        if op.operation == "DELETE":
            # No replacement content -- remove the symbol's lines outright
            # instead of the old behavior of splicing in a single blank
            # line where it used to be. Also absorb the whole contiguous
            # run of blank lines immediately following it (there may be
            # one, two -- PEP8's convention between top-level defs -- or
            # none), so deleting a function doesn't leave extra blank
            # lines at the junction. What's left is exactly the gap that
            # existed *before* the deleted symbol, which in a consistently
            # styled file is the same convention -- so this self-adjusts
            # to whatever blank-line style the file already used, rather
            # than hardcoding a count. Only ever touches lines directly
            # adjacent to the deleted symbol -- never reflows blank lines
            # anywhere else in the file.
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
