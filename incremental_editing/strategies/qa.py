"""Q&A strategy: answer a question *about* a file instead of editing it.

Reached only via generate_delta's escalate:{"kind":"question"} signal (see
structured_edit.py's prompt) -- the model itself recognizes, from the
request's own shape alone (no file content needed to tell "list all
algorithms" or "what does handle() do" apart from an edit instruction),
that there's nothing to change here. Given the whole file (scope choice:
this file only, not a repo-wide retrieval -- cheaper, and correct for
"explain/list/describe this file" which is the common case), answer the
actual question directly. No Delta IR, no apply/test/versioning/commit,
nothing written -- there's nothing to confirm either.
"""

import time
from typing import Optional

from ..config import get_settings

SYSTEM_PROMPT = (
    "Code Q&A assistant. Given a complete file and a question about it, answer clearly and "
    "concisely in plain text -- no code fences, no proposed edits, no prose about what you would "
    "change. Answer only what was asked, grounded in the actual file content shown."
)


def build_messages(file_path: str, source: str, question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{file_path}:\n{source}\n\nQuestion: {question}"},
    ]


def generate_answer(file_path: str, source: str, question: str, model: Optional[str] = None) -> dict:
    from openai import OpenAI

    settings = get_settings()
    model = model or settings.llm_model
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=0,
    )
    messages = build_messages(file_path, source, question)

    start = time.time()
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    latency_ms = int((time.time() - start) * 1000)

    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0

    return {
        "answer": response.choices[0].message.content.strip(),
        "input_tokens": usage.prompt_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": latency_ms,
        "model": model,
    }
