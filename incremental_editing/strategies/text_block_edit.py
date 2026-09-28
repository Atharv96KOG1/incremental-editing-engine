"""TEXT_BLOCK_EDIT strategy: regenerate exactly one located block (see
analyzer/text_blocks.py) of a file with no function/class concept at
all, instead of the whole file. Same generation mechanics
full_regeneration.py already uses (reused directly, not duplicated),
just given one section instead of the entire file as both input and
expected output shape.
"""

import re
from typing import List, Optional, Tuple

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


_MULTI_BLOCK_START_RE = re.compile(r"^===BLOCK: (.*?)===\s*$", re.MULTILINE)
_MULTI_BLOCK_END = "===END BLOCK==="

_MULTI_BLOCK_SYSTEM_PROMPT = (
    "Text-block editor. Given SEVERAL blocks (sections) from the same file, plus ONE request that "
    "may need changes across more than one of them, output the complete new content for EVERY block "
    "shown -- including any block left completely unchanged, repeated verbatim -- using exactly this "
    "format once per block, in the same order shown, nothing else outside it (no prose, no markdown "
    "fences, no explanation):\n\n"
    "===BLOCK: <exact name as shown>===\n"
    "<complete new content for that block>\n"
    "===END BLOCK===\n\n"
    "Change only what the request actually asks for in each block; a block the request doesn't touch "
    "must come back byte-identical to how it was shown."
)


def build_multi_block_messages(file_path: str, blocks: List[Tuple[str, str]], user_request: str) -> list:
    parts = [f"{file_path}, {len(blocks)} blocks:"]
    for name, content in blocks:
        parts.append(f"===BLOCK: {name}===\n{content}\n===END BLOCK===")
    parts.append(f"Request: {user_request}")
    return [
        {"role": "system", "content": _MULTI_BLOCK_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_multi_block_response(raw: str, expected_names: List[str]) -> dict:
    """Maps each expected block name to its new content, in the order
    `expected_names` gives (not the order the model happened to return
    them in, though it's told to preserve it) -- a name the response
    never mentions at all means the model dropped it; the caller must
    treat that as a real failure (UnseenReplaceTargetError's own "never
    silently keep the old content when the model didn't actually
    address it" contract), not silently reuse the original."""
    found = {}
    starts = list(_MULTI_BLOCK_START_RE.finditer(raw))
    for i, m in enumerate(starts):
        name = m.group(1).strip()
        body_start = m.end()
        body_end_marker = raw.find(_MULTI_BLOCK_END, body_start)
        next_start = starts[i + 1].start() if i + 1 < len(starts) else len(raw)
        body_end = body_end_marker if 0 <= body_end_marker < next_start else next_start
        found[name] = raw[body_start:body_end].strip("\n")
    return {name: found.get(name) for name in expected_names}


def generate_multi_block_replacement(
    file_path: str, blocks: List[Tuple[str, str]], user_request: str, model: Optional[str] = None
) -> dict:
    gen = _call_llm_for_full_file(build_multi_block_messages(file_path, blocks, user_request), model)
    gen["blocks"] = parse_multi_block_response(gen["code"], [name for name, _ in blocks])
    return gen
