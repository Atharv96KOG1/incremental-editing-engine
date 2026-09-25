"""FULL_REGENERATION strategy (doc section 10, Approach A).

Used three places: as the baseline to compare incremental editing
against; the only way to get code into a file that doesn't exist yet, to
bootstrap a brand-new file from a prompt before any incremental edit has
something to work against; and as the fallback for an existing file when
the request is file-wide (comments, whitespace, docstrings, formatting --
see analyzer/locator.is_whole_file_request) rather than a change to one
named symbol, since that reaches content STRUCTURED_EDIT's per-symbol
Delta IR can't touch.
"""

import re
import time
from typing import Optional

from ..config import get_settings

SYSTEM_PROMPT = (
    "Code generator. Match the language the target file's own path/extension implies -- never "
    "assume Python. Output ONLY complete runnable source for the requested file — no prose, no "
    "markdown fences, no explanation."
)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip() + "\n"


def build_messages(file_path: str, user_request: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{file_path}: {user_request}"},
    ]


# Given a complete existing file, transform it per the request and hand
# back the complete new file -- the *edit*-mode counterpart to the
# create-from-nothing prompt above, used for a file-wide structural/
# hygiene request (comments, whitespace, docstrings, formatting; see
# analyzer/locator.is_whole_file_request) that reaches content outside any
# single symbol's own span, which STRUCTURED_EDIT's per-symbol Delta IR
# fundamentally can't touch. Not hardcoded to Python -- EDIT already
# supports every language index_symbols/detect_language recognize.
_EDIT_SYSTEM_PROMPT = (
    "Code-transformation engine. Given a complete existing file and a request describing a "
    "file-wide change, output ONLY the complete new file content in the file's own language — "
    "no prose, no markdown fences, no explanation. Preserve all real logic and behavior exactly; "
    "change only what the request asks for. The result must still be syntactically valid and "
    "runnable in that language: a request like 'remove all whitespace' or 'remove all blank "
    "lines' means trailing/extra whitespace and unnecessary blank lines, never whitespace the "
    "language's own syntax requires (indentation, token separators) -- apply the reasonable "
    "reading, not the literal one, whenever the literal one would break parsing."
)


def build_edit_messages(file_path: str, original_source: str, user_request: str) -> list:
    return [
        {"role": "system", "content": _EDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{file_path}:\n{original_source}\n\nRequest: {user_request}"},
    ]


def _call_llm_for_full_file(messages: list, model: Optional[str]) -> dict:
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
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    latency_ms = int((time.time() - start) * 1000)

    raw = response.choices[0].message.content
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0

    return {
        "code": _strip_fences(raw),
        "input_tokens": usage.prompt_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": latency_ms,
        "model": model,
    }


def generate_full_file(file_path: str, user_request: str, model: str = None) -> dict:
    return _call_llm_for_full_file(build_messages(file_path, user_request), model)


def generate_full_file_edit(file_path: str, original_source: str, user_request: str, model: str = None) -> dict:
    return _call_llm_for_full_file(build_edit_messages(file_path, original_source, user_request), model)
