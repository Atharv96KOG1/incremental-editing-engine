"""Optional pre-classification of a request's escalate kind via TypeSafe
AI's Jev model (docs.typesafe.ai) -- a structured-decision model with a
calibrated confidence score, not a chat model.

STRUCTURED_EDIT's own classification already decides this, but only as a
side effect of a full generation call. Two escalate kinds ("question",
"whole_file") can be identified from the request's wording alone, so a
confident Jev call before context-building can dispatch directly and
skip STRUCTURED_EDIT's generation entirely.

Still model-driven, not a keyword/regex classifier -- same
"escalate is model-driven" contract structured_edit.py holds to, just
backed by a cheaper model tuned for this decision. A low-confidence or
unavailable answer always degrades to running the pipeline unchanged.
"""

from typing import Optional

from ..config import get_settings

_TIMEOUT_SECONDS = 10.0

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

DISPATCHABLE_KINDS = {"question", "whole_file"}


def _get_client():
    """None when Jev isn't configured or the SDK isn't installed; callers
    treat None as "no confident answer, fall through"."""
    settings = get_settings()
    if not settings.typesafe_api_key:
        return None
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError:
        return None
    return TypeSafeClient(api_key=settings.typesafe_api_key, timeout=_TIMEOUT_SECONDS)


def classify_request_kind(request: str) -> Optional[str]:
    """Returns the request's escalate kind per Jev's Choice
    classification, only when confidence clears
    `settings.jev_confidence_threshold`; None otherwise, which callers
    must treat as "don't shortcut". The threshold defaults higher than
    typesafe's own 0.5 example, since a wrong shortcut here causes real
    silent misbehavior (answers an edit as a question, or discards an
    edit as regeneration), not just a re-askable miss. Never raises -- a
    broken gateway just means this signal doesn't fire."""
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
