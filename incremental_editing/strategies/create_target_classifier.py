"""Decides whether a request with no existing edit target actually wants
brand-new file(s)/folder(s) created, and if so, what real path(s) they
should have -- the one real gap between "auto-locate found nothing" and
refusing the request outright. Handles a multi-file/folder request (e.g.
"frontend and backend in separate folders") the same way run_pipeline.
_run_create_files already does for an escalated edit-mode request --
one path per real responsibility, never merged into a single file.

Real gap this closes: edit mode with no file selected, and hybrid
retrieval finding no existing candidate (a genuinely new/near-empty
project, or a request that just doesn't match anything that exists),
previously refused outright ("no file given and hybrid retrieval found
no candidate") even for a request that plainly describes creating a new
file -- forcing a manual switch to Create mode (and typing the exact
filename by hand) for something the request itself already said.

Deliberately a real, structured-output model call -- never a keyword/
regex guess at intent ("contains 'make a new file'..."). Same "escalate
is model-driven, mechanical code never guesses" contract every other
classification in this project already holds itself to. A single,
cheap call: no context beyond the request and a bare file listing, no
generation, nothing this project's own offline test suite can't mock
exactly like generate_delta's own classification call already is."""

import json
import time
from typing import Optional

from ..config import get_settings

_SYSTEM_PROMPT = (
    "No existing file in this project matched the request below. Decide whether the "
    "request actually describes creating brand-new file(s)/folder(s). If so, propose the "
    "real relative FILE path(s) needed -- one per real responsibility, never merged (e.g. "
    "\"frontend and backend in separate folders\" means real files under both frontend/ "
    "and backend/ -- frontend/index.html, backend/server.py, and so on -- never just the "
    "two folder names with nothing inside them). A bare path ending in \"/\" with no file "
    "in it is ONLY for a request that explicitly asks for an empty folder and nothing else "
    "-- almost never the right answer for a request that describes real functionality. "
    "Never propose one of the existing paths already listed. If the request is ambiguous, "
    "or really describes editing something that should already exist (just unmatched or "
    "misspelled), say no -- never guess paths unless the new-file intent is genuinely "
    "unambiguous."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "wants_new_file": {"type": "boolean"},
        "paths": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["wants_new_file", "paths"],
    "additionalProperties": False,
}


def classify_create_intent(request: str, project_listing: str, model: Optional[str] = None) -> dict:
    from openai import OpenAI

    settings = get_settings()
    model = model or settings.llm_model
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=0,
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Request: {request}\n\n"
                f"Existing files in this project (for context / name collisions):\n"
                f"{project_listing or '(empty project)'}"
            ),
        },
    ]
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "create_intent", "schema": _SCHEMA, "strict": True},
        },
        temperature=0,
        max_tokens=400,
    )
    latency_ms = int((time.time() - start) * 1000)
    raw = json.loads(response.choices[0].message.content)
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0

    return {
        "wants_new_file": raw["wants_new_file"],
        "paths": raw["paths"],
        "model": model,
        "input_tokens": usage.prompt_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": latency_ms,
    }
