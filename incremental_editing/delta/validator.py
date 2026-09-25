"""Delta validation: schema shape, then target existence against the real source."""

from typing import Dict, Optional

import jsonschema

from ..analyzer.locator import AmbiguousSymbolError, defined_symbol_name, find_symbol, index_symbols
from .schema import DELTA_JSON_SCHEMA, DeltaIR


class DeltaValidationError(Exception):
    pass


class UnseenReplaceTargetError(DeltaValidationError):
    """A REPLACE named a real symbol, but its body was never shown to the
    model -- only its bare name, via the compact-context fallback.
    Distinct from a generic DeltaValidationError so run_pipeline.py can
    react to it specifically: force a direct escalation to whole-file
    regeneration (which sends the real content) instead of retrying the
    same limited context through the normal repair loop and hoping the
    model chooses to escalate on its own -- observed not to, reporting a
    safe but unhelpful no-op instead of actually satisfying the request."""


def _check_decorators_preserved(source: str, sym, new_content: str, symbol_name: str) -> None:
    """A REPLACE must restate a symbol's *complete* body, decorators
    included -- real regression this caught live: a REPLACE for a Flask
    route handler dropped its own '@app.route(...)' decorator, which the
    apply engine then spliced right over (find_symbol's start_line
    already includes decorator lines, by design, so the model was shown
    them), silently un-registering that route -- the file still parsed
    fine and nothing else failed, so nothing surfaced the loss until the
    endpoint itself was hit and 404'd.

    Checked structurally against the ORIGINAL symbol's own real source
    lines, not by re-running the app or guessing what a missing
    decorator does. `@`-prefixed lines generalize past Python (Java/
    Kotlin annotations and TypeScript decorators use the same syntax),
    so this isn't Python-only despite most real hits being Flask
    routes."""
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
    """Confirm every REPLACE/DELETE target and INSERT anchor actually exists in source,
    exactly once -- a name defined more than once is refused, not guessed at, UNLESS
    `prefer_lines` (name -> start_line) already resolved which occurrence is meant
    (locate_candidates does this via class-name mention or a semantic tiebreak when a
    bare name is duplicated, e.g. the same method repeated across classes).

    Also refuses an INSERT whose own content would create a *second*
    definition of a name that already exists elsewhere in the file --
    real corruption this caught retroactively: a class-method version of
    a function inserted (as part of a broader refactor) alongside an old
    top-level one that was never removed, both silently coexisting until
    something later tried to REPLACE/DELETE that name and hit an
    unresolvable "defined N times" error with no way to tell which one
    was meant. The model's own declared symbol_name is never trusted for
    this -- defined_symbol_name looks at what `content` actually
    defines, the same way api/run_pipeline.py corrects a mismatched
    *displayed* INSERT name.

    `content_shown_for` -- when given, the set of symbol names whose
    *full body* was actually included in the model's context (i.e.
    context_builder.py's candidate_symbols, not the bare compact name
    list every other symbol only gets a name for) -- refuses a REPLACE
    on any name outside it. Real data loss this caught: a request that
    matched no localized candidate fell back to the compact context
    (imports + bare names, no bodies), and the model still emitted a
    REPLACE for one of those bare names -- since REPLACE's content must
    be the *complete* new body, and the model had never seen the real
    one, it fabricated an entirely different, much smaller function,
    silently dropping real content (an extra table, half the original
    columns) it was never shown in the first place. DELETE and INSERT
    are exempt: DELETE needs no content, and INSERT's content is new by
    definition, not a rewrite of something unseen."""
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
                # DELETE never auto-resolves a delegate pair -- removing
                # just the implementation would leave its wrapper calling
                # a method that no longer exists.
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
