"""TEXT_BLOCK_EDIT strategy: regenerate exactly one located block (see
analyzer/text_blocks.py) of a file with no function/class concept at
all, instead of the whole file. Same generation mechanics
full_regeneration.py already uses (reused directly, not duplicated),
just given one section instead of the entire file as both input and
expected output shape.
"""

from typing import Optional

from .full_regeneration import _call_llm_for_full_file

_BLOCK_SYSTEM_PROMPT = (
    "Text-block editor. Given ONE block (a section) from a larger file, by name and its current "
    "content, plus a request, output ONLY the complete new content for THAT block -- the file's own "
    "format/language, nothing before or after it, no prose, no markdown fences, no explanation. "
    "Preserve the block's own heading/section marker line if it had one. Change only what the "
    "request asks for; leave everything else in the block exactly as it was. Surrounding context "
    "(if shown) is the neighboring sections for reference only -- never reproduce it in your output."
)


def build_block_messages(
    file_path: str,
    block_name: str,
    block_content: str,
    user_request: str,
    context_before: str = "",
    context_after: str = "",
) -> list:
    parts = [f'{file_path}, block "{block_name}":']
    if context_before:
        parts.append(f"--- context immediately before this block (do not include in output) ---\n{context_before}")
    parts.append(f"--- block content, regenerate only this ---\n{block_content}")
    if context_after:
        parts.append(f"--- context immediately after this block (do not include in output) ---\n{context_after}")
    parts.append(f"Request: {user_request}")
    return [
        {"role": "system", "content": _BLOCK_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def generate_block_replacement(
    file_path: str,
    block_name: str,
    block_content: str,
    user_request: str,
    model: Optional[str] = None,
    context_before: str = "",
    context_after: str = "",
) -> dict:
    return _call_llm_for_full_file(
        build_block_messages(file_path, block_name, block_content, user_request, context_before, context_after), model
    )
