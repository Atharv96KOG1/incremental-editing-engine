"""Optional pre-classification of a request's escalate kind via
TypeSafe AI's Jev model (docs.typesafe.ai) -- a type-safe, structured-
decision model, not a general chat model: "unstructured state in, typed
probabilistic decisions out," with a calibrated confidence score on every
answer.

STRUCTURED_EDIT's own classification (structured_edit.py's escalate
field) already decides this correctly -- but only as a side effect of a
real generation call that also produces Delta IR operations, priced for
that whole job. Two of its seven escalate kinds ("question", "whole_file")
need nothing beyond the request's own wording to identify and dispatch
directly (_run_question / _run_whole_file_edit both already accept an
optional classification_gen and work with zero prior LLM cost) -- so a
confident Jev call ahead of context-building can skip straight to the
right handler and skip STRUCTURED_EDIT's own classification generation
entirely, for the same reason the mechanical fast paths above it in
run_pipeline.py already skip it for delete/rename.

This is deliberately NOT a hardcoded keyword/regex classifier: the
decision is still made by a model, on the actual request text, with a
real confidence score attached -- same "escalate is model-driven"
contract this project already holds structured_edit.py to, just backed
by a faster/cheaper model tuned for exactly this kind of decision instead
of a full generation call. A low-confidence or unavailable answer always
degrades to "don't shortcut" -- the existing pipeline runs completely
unchanged, exactly like a failed VectorRetriever/BM25 signal degrades in
context_builder.py and text_blocks.py.
"""

from typing import Optional

from ..config import get_settings

# Jev is advertised at 70-500ms per call -- this is a generous safety cap
# for a slow/cold-starting gateway, not a normal-case expectation (same
# reasoning vector_retriever.py's own timeout applies to embeddings
# calls): a live edit request is waiting on this, so a half-broken
# endpoint must fail fast into the existing pipeline, not hang it.
_TIMEOUT_SECONDS = 10.0

# Every escalate kind structured_edit.py's own prompt can produce, plus
# "structured_edit" for the ordinary REPLACE/INSERT/DELETE case (no
# escalation at all) -- kept in the model's own words so this classifier
# and the LLM's own escalate detection can never quietly drift into
# disagreeing on what a kind means.
_KIND_CRITERIA = {
    "structured_edit": "A targeted change to one or a few existing functions/classes -- adding, "
    "replacing, or removing something inside their own bodies. The default/most common case.",
    "whole_file": "Reaches outside any single function/class's own span -- comments, whitespace, "
    "formatting, or module-level statements not inside any one symbol.",
    "language_conversion": "Rewrite this file's code into a different programming language entirely.",
    "create_files": "Create new file(s) or a folder unrelated to this file's own existing content.",
    "delete_file": "Delete this whole file, not one symbol inside it.",
    "rename_identifier": "The whole change is swapping one exact name/value for another everywhere "
    "it appears (a function, class, variable, or literal), nothing else.",
    "question": "Asks about the code or requests information -- not an instruction to change anything.",
}

# Only these kinds are safe to dispatch on directly from this
# classification alone: both _run_question and _run_whole_file_edit need
# nothing beyond the request text and the file's current source, the
# same two inputs Jev itself saw. The other kinds need extra structured
# fields (a target language, a file list, old/new name pairs) a Choice
# answer doesn't produce -- those still go through STRUCTURED_EDIT's own
# escalate detection (or an existing mechanical fast path) unchanged.
DISPATCHABLE_KINDS = {"question", "whole_file"}


def _get_client():
    """None when Jev isn't configured (no API key) or the SDK isn't
    installed -- an opt-in dependency, not a required one. Every caller
    treats None the same as "no confident answer": fall through."""
    settings = get_settings()
    if not settings.typesafe_api_key:
        return None
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError:
        return None
    return TypeSafeClient(api_key=settings.typesafe_api_key, timeout=_TIMEOUT_SECONDS)


def classify_request_kind(request: str) -> Optional[str]:
    """The request's escalate kind per Jev's Choice classification, only
    when confidence clears settings.jev_confidence_threshold -- None
    otherwise (not configured, a transport/API failure, or genuinely
    unsure), which every caller must treat as "don't shortcut, run the
    existing pipeline unchanged."

    The threshold itself lives in Settings, not a local constant, and
    defaults higher than the 0.5 docs.typesafe.ai's own intent-routing
    example uses for customer-service ticket routing: there, a
    low-confidence miss just means a human re-reads the ticket. Here, a
    wrong shortcut either answers a real edit request as a question (the
    file is silently never touched) or throws away an edit as file-wide
    regeneration when it wasn't -- both real, silent misbehavior, not a
    recoverable "ask again" -- so the bar for trusting the shortcut is
    higher.

    Never raises: a broken gateway degrades to this signal simply not
    firing, same contract _safe_vector_rank already holds embeddings
    retrieval to."""
    client = _get_client()
    if client is None:
        return None

    try:
        from typesafe_sdk import Choice, TypeSafeError
    except ImportError:
        return None

    try:
        response = client.system_one(
            state=request,
            questions={
                "kind": Choice(
                    instructions="What kind of code-editing request is this, by its own wording alone",
                    criteria=_KIND_CRITERIA,
                )
            },
        )
    except TypeSafeError:
        return None
    except Exception:
        return None

    answer = response.answers["kind"]
    if answer.confidence < get_settings().jev_confidence_threshold:
        return None
    return answer.choice
