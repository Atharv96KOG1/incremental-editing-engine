"""Delta validation: schema shape, then target existence against the real source."""

from typing import Dict, Optional

import jsonschema

from ..analyzer.locator import AmbiguousSymbolError, defined_symbol_name, find_symbol, index_symbols
from .schema import DELTA_JSON_SCHEMA, DeltaIR


class DeltaValidationError(Exception):
    pass


class UnseenReplaceTargetError(DeltaValidationError):
    """A REPLACE named a real symbol, but its body was never shown to the
    model (only its bare name). Distinct from DeltaValidationError so
    run_pipeline.py can force escalation to whole-file regeneration
    instead of retrying through the repair loop, which the model doesn't
    reliably self-correct from."""


def _check_decorators_preserved(source: str, sym, new_content: str, symbol_name: str) -> None:
    """A REPLACE must restate a symbol's complete body including
    decorators -- a model once dropped a Flask route's '@app.route(...)'
    decorator in a REPLACE, silently un-registering the route with no
    parse error to catch it. Checked against the original symbol's real
    source lines. `@`-prefixed lines also cover Java/Kotlin annotations
    and TypeScript decorators, not just Python."""
    original_lines = source.splitlines()[sym.start_line - 1 : sym.end_line]
    original_decorators = [line.strip() for line in original_lines if line.strip().startswith("@")]
    if not original_decorators:
        return
    new_lines = {line.strip() for line in (new_content or "").splitlines()}
    missing = [d for d in original_decorators if d not in new_lines]
    if missing:
        raise DeltaValidationError(
            f"REPLACE for '{symbol_name}' is missing its original decorator(s) ({', '.join(missing)}) -- "
            "a REPLACE must restate the complete symbol, decorators included, or removing one silently "
            "changes behavior (e.g. un-registering a route)"
        )


def validate_schema(delta_dict: dict) -> None:
    try:
        jsonschema.validate(delta_dict, DELTA_JSON_SCHEMA)
    except jsonschema.ValidationError as e:
        raise DeltaValidationError(f"schema validation failed: {e.message}") from e


def validate_targets(
    delta: DeltaIR,
    source: str,
    language: str = "python",
    prefer_lines: Optional[Dict[str, int]] = None,
    content_shown_for: Optional[set] = None,
) -> None:
    """Confirms every REPLACE/DELETE target and INSERT anchor exists in
    source exactly once; an ambiguous name is refused unless
    `prefer_lines` already resolved which occurrence is meant. Also
    refuses an INSERT whose content would create a duplicate definition
    of an existing name (caught a real case of two methods silently
    coexisting undetected). `content_shown_for`, when given, is the set
    of symbol names whose full body was actually shown to the model --
    REPLACE on a name outside it is refused, since the model would
    otherwise fabricate/drop content for a symbol it never saw. DELETE
    and INSERT are exempt (no prior content to compare against)."""
    symbols = index_symbols(source, language)
    prefer_lines = prefer_lines or {}
    for op in delta.operations:
        t = op.target
        try:
            if op.operation in ("REPLACE", "DELETE"):
                if op.operation == "REPLACE" and content_shown_for is not None and t.symbol_name not in content_shown_for:
                    raise UnseenReplaceTargetError(
                        f"cannot REPLACE {t.symbol_type} '{t.symbol_name}': its body was never shown to you "
                        "(only its bare name), so rewriting it risks silently discarding real content you "
                        "never saw -- escalate:{\"kind\":\"whole_file\"} instead to get its real content first"
                    )
                allow_delegate = op.operation == "REPLACE"
                sym = find_symbol(
                    symbols, t.symbol_type, t.symbol_name, prefer_lines.get(t.symbol_name), source, allow_delegate
                )
                if sym is None:
                    raise DeltaValidationError(
                        f"target {t.symbol_type} '{t.symbol_name}' not found for {op.operation}"
                    )
                if op.operation == "REPLACE":
                    _check_decorators_preserved(source, sym, op.content, t.symbol_name)
            elif op.operation == "INSERT":
                if t.anchor:
                    if find_symbol(symbols, t.symbol_type, t.anchor, prefer_lines.get(t.anchor), source) is None:
                        raise DeltaValidationError(f"anchor {t.symbol_type} '{t.anchor}' not found for INSERT")
                inserted_name = defined_symbol_name(op.content, language)
                if inserted_name and any(s.symbol_type == t.symbol_type and s.name == inserted_name for s in symbols):
                    raise DeltaValidationError(
                        f"cannot INSERT {t.symbol_type} '{inserted_name}': already exists in this file -- "
                        "use REPLACE to change it, or INSERT under a genuinely new name"
                    )
        except AmbiguousSymbolError as e:
            raise DeltaValidationError(str(e)) from e
