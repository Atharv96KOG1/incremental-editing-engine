"""STRUCTURED_EDIT strategy: ask the model for a Delta IR JSON object directly.

`generate_repair()` is the bounded self-repair path (PHOENIX architecture
doc, section 18): instead of retrying with the exact same prompt, it shows
the model its own previous (failed) Delta IR plus the classified failure
and the real error text, and asks for a corrected one.
"""

import json
import time

from ..config import get_settings
from ..delta.schema import DELTA_SCHEMA_VERSION
from ..optimization.prompt_compression import compress

# Readable source of truth -- edit THIS when the instructions need to
# change. SYSTEM_PROMPT below is the compressed form actually sent to the
# model; compress() itself refuses (raises) if compression would have
# altered any negation/conditional word's count, so this can't silently
# drift into meaning something different than what's written here.
#
# Deliberately NOT asking for schema_version, base_version, or target.file:
# all three are already known before this call is ever made (schema_version
# is a fixed constant; base_version and file_path are both parameters right
# here), and grep across the whole codebase confirms nothing ever reads the
# *parsed* delta's own copies of them back out afterward. _fill_known_fields
# below backfills all three onto the model's response; asking for them here
# was pure wasted output tokens (billed well above input's rate for this
# model) on every single call, for values the model was never even the
# source of truth for.
#
# Also deliberately NOT describing the operations/escalate JSON *shape* in
# here at all anymore (a previous version restated it in full, in JSON-
# literal form, costing ~150 tokens on every single call): the API call
# below now enforces STRICT_DELTA_RESPONSE_SCHEMA via response_format's
# strict json_schema mode (verified live against this project's own
# gateway), so the shape is a structural guarantee from the API itself,
# not something the model needs telling in English. Everything left here
# is the part no schema can express -- the semantic/decision rules for
# *when* to use which field, not what fields exist.
_SYSTEM_PROMPT_SOURCE = (
    "Code-editing engine. Context may be a partial file excerpt -- match the language its path's "
    "extension implies, never assume Python. Touch fewest symbols needed; content must be complete, "
    "syntactically valid, with no unrelated changes. A trailing comment lists every other real "
    "symbol name in this file -- no bodies, sometimes a signature/decorator for disambiguation only. "
    "symbol_name/anchor must be the bare name ('foo', strip from the first '(' on), taken only from "
    "that list or a symbol shown in full -- never invented. A shown class method (a constructor "
    "included) is itself a function-type target -- REPLACE it directly, never its enclosing class. "
    "A near-miss (e.g. a misspelling) of "
    "exactly one listed name still means that name -- correct it and act normally, not a reason to "
    "escalate. Unlisted and not shown, with no close match either, means it doesn't exist, full stop. "
    "If a name already exists, INSERT something else instead of redefining it. For a conditional "
    "request ('add X if X doesn't exist'), check the list yourself and act -- don't return empty "
    "operations just because you can't see a body.\n\n"
    "A symbol needing an import not already present: add it inside that symbol's own body, don't "
    "escalate for it.\n\n"
    "Leave operations empty if already satisfied, OR set escalate (never guess symbol-level "
    "operations for any of these) when REPLACE/INSERT/DELETE genuinely can't do it:\n"
    "whole_file -- ONLY content outside every symbol's own span (module docstring, bare "
    "module-level statement, standalone comment, or blank-line formatting between symbols) -- "
    "touching many symbols is still several REPLACE ops, never this: each already restates its "
    "complete body regardless of how many others change too.\n"
    "language_conversion -- a different language entirely; give target_language and target_extension "
    "(no dot).\n"
    "create_files -- new file(s) unrelated to this file's own content; one path per real "
    "responsibility, never merged -- bare 'dir/' only when nothing belongs inside it (naming a "
    "folder never alone means folder-only; described behavior still gets its own real file(s) "
    "there). also_link_current_file only if this file should then update to use them.\n"
    "delete_file -- deletes this whole file, not one symbol in it.\n"
    "rename_identifier -- the whole change is swapping exact text everywhere it occurs (an "
    "identifier, env-var name, or literal value); one renames entry per distinct swap -- a "
    "multi-part swap (e.g. a class name, an env var, and a default value together) is still this, "
    "several entries, not whole_file.\n"
    "question -- request (by its own wording alone, no file content needed) asks about the code "
    "rather than instructing a change."
)

SYSTEM_PROMPT = compress(_SYSTEM_PROMPT_SOURCE)

# Strict JSON-schema structured output (verified live: this project's own
# Bifrost gateway supports response_format's strict json_schema mode).
# Strict mode requires every declared property to be listed in "required"
# at every level (a field that's merely optional becomes required-but-
# nullable instead) and additionalProperties:false everywhere -- this is
# a separate, API-communication-shaped schema from delta.schema.DELTA_JSON_
# SCHEMA (the lenient, internal post-parse validation schema used
# elsewhere), not a replacement for it. _strip_strict_nulls below
# immediately normalizes a parsed response back to the lenient shape
# (explicit nulls removed) so every downstream consumer -- validate_schema,
# _fill_known_fields, Operation.from_dict, run_pipeline's own `.get
# ("escalate") or {}` -- sees exactly the same shape it always has, unaware
# this exists.
STRICT_DELTA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "escalate": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "whole_file",
                                "language_conversion",
                                "question",
                                "create_files",
                                "delete_file",
                                "rename_identifier",
                            ],
                        },
                        "target_language": {"type": ["string", "null"]},
                        "target_extension": {"type": ["string", "null"]},
                        "renames": {
                            "type": ["array", "null"],
                            "items": {
                                "type": "object",
                                "properties": {"old_name": {"type": "string"}, "new_name": {"type": "string"}},
                                "required": ["old_name", "new_name"],
                                "additionalProperties": False,
                            },
                        },
                        "files": {"type": ["array", "null"], "items": {"type": "string"}},
                        "also_link_current_file": {"type": ["boolean", "null"]},
                    },
                    "required": [
                        "kind",
                        "target_language",
                        "target_extension",
                        "renames",
                        "files",
                        "also_link_current_file",
                    ],
                    "additionalProperties": False,
                },
            ]
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": list(("REPLACE", "INSERT", "DELETE"))},
                    "target": {
                        "type": "object",
                        "properties": {
                            "symbol_type": {"type": "string", "enum": list(("function", "class"))},
                            "symbol_name": {"type": "string"},
                            "anchor": {"type": ["string", "null"]},
                        },
                        "required": ["symbol_type", "symbol_name", "anchor"],
                        "additionalProperties": False,
                    },
                    "content": {"type": ["string", "null"]},
                },
                "required": ["operation", "target", "content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["escalate", "operations"],
    "additionalProperties": False,
}


def _strip_strict_nulls(delta_dict: dict) -> dict:
    """Undoes strict mode's required-but-nullable convention right after
    parsing, so every consumer downstream of this call keeps seeing the
    exact same optional-key-may-be-absent shape it always has -- an
    explicit `"content": null` becomes simply no "content" key, matching
    what a DELETE operation always looked like before strict mode existed
    (and what delta.schema.DELTA_JSON_SCHEMA's own lenient validation
    still expects)."""
    if delta_dict.get("escalate") is None:
        delta_dict.pop("escalate", None)
    else:
        escalate = delta_dict["escalate"]
        for key in ("target_language", "target_extension", "renames", "files", "also_link_current_file"):
            if escalate.get(key) is None:
                escalate.pop(key, None)
    for op in delta_dict.get("operations") or []:
        if not isinstance(op, dict):
            continue
        if op.get("content") is None:
            op.pop("content", None)
        target = op.get("target")
        if isinstance(target, dict) and target.get("anchor") is None:
            target.pop("anchor", None)
    return delta_dict


def _fill_known_fields(delta_dict: dict, file_path: str, base_version: str) -> None:
    """Backfills schema_version, base_version, and each operation's
    target.file onto the model's (now slimmer) response -- values this
    call already had before it was ever made, never requested from the
    model in the first place. Mutates in place; called on every
    generate_delta/generate_repair result before anything downstream
    (schema validation, DeltaIR parsing) ever sees it."""
    delta_dict.setdefault("schema_version", DELTA_SCHEMA_VERSION)
    delta_dict.setdefault("base_version", base_version)
    for op in delta_dict.get("operations", []):
        if isinstance(op, dict):
            op.setdefault("target", {}).setdefault("file", file_path)


def build_messages(file_path: str, context_content: str, user_request: str, base_version: str) -> list:
    user_prompt = f"{file_path} (base {base_version}):\n{context_content}\n\nRequest: {user_request}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_messages(
    file_path: str,
    context_content: str,
    user_request: str,
    base_version: str,
    failed_delta: dict,
    failure_class: str,
    failure_detail: str,
) -> list:
    user_prompt = (
        f"{file_path} (base {base_version}):\n{context_content}\n\nRequest: {user_request}\n\n"
        f"Your previous attempt returned this Delta IR:\n{json.dumps(failed_delta)}\n\n"
        f"That attempt failed with [{failure_class}]: {failure_detail}\n\n"
        "Return a corrected Delta IR JSON that fixes exactly this failure. Keep the rest of your "
        "approach unless it's actually the cause -- don't rewrite parts that weren't the problem."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _call_llm(messages: list, model: str = None) -> dict:
    from openai import OpenAI  # lazy import: only needed when a live call is made

    settings = get_settings()
    model = model or settings.llm_model
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=0,
    )

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "delta", "schema": STRICT_DELTA_RESPONSE_SCHEMA, "strict": True},
        },
        temperature=0,
        max_tokens=settings.max_output_tokens,
    )
    latency_ms = int((time.time() - start) * 1000)

    raw_json = response.choices[0].message.content
    usage = response.usage

    # Real, measured behavior (verified against the live gateway): an
    # identical system+context prefix across two calls -- exactly what
    # every repair retry sends, since only the failure-specific suffix
    # changes -- gets served up to ~85% from cache on the second call.
    # Providers bill cached input tokens at a discount; without reading
    # this field, estimate_cost() charges every repair attempt as if the
    # whole prompt were freshly computed, overstating its real cost.
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0

    return {
        "raw_json": raw_json,
        "delta_dict": _strip_strict_nulls(json.loads(raw_json)),
        "input_tokens": usage.prompt_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": latency_ms,
        "model": model,
    }


def generate_delta(
    file_path: str, context_content: str, user_request: str, base_version: str, model: str = None
) -> dict:
    messages = build_messages(file_path, context_content, user_request, base_version)
    gen = _call_llm(messages, model)
    _fill_known_fields(gen["delta_dict"], file_path, base_version)
    return gen


def generate_repair(
    file_path: str,
    context_content: str,
    user_request: str,
    base_version: str,
    failed_delta: dict,
    failure_class: str,
    failure_detail: str,
    model: str = None,
) -> dict:
    messages = build_repair_messages(
        file_path, context_content, user_request, base_version, failed_delta, failure_class, failure_detail
    )
    gen = _call_llm(messages, model)
    _fill_known_fields(gen["delta_dict"], file_path, base_version)
    return gen
