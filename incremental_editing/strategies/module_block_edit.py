"""MODULE_BLOCK_EDIT strategy: regenerate exactly one located module-level
region (see analyzer/module_blocks.py) instead of the whole file, for
content that sits outside every function/class's own span in a language
that otherwise has a real symbol concept.

Same generation mechanics strategies/text_block_edit.py already uses
(reused directly, not duplicated) -- just given one module-level region
instead of a whole-file-format section as both input and expected output
shape.
"""

from typing import Optional

from .full_regeneration import _call_llm_for_full_file

_MODULE_BLOCK_SYSTEM_PROMPT = (
    "Module-level code editor. Given ONE region of a larger file -- content outside every function/class's "
    "own span (a constant, an import block, a bare statement, a __main__ guard) -- plus a request, output "
    "ONLY the complete new content for THAT region, in the file's own language -- nothing before or after "
    "it, no prose, no markdown fences, no explanation. Change only what the request asks; leave everything "
    "else in the region exactly as it was. Surrounding context (if shown) is neighboring code for reference "
    "only -- never reproduce it in your output."
)


def build_module_block_messages(
    file_path: str,
    block_name: str,
    block_content: str,
    user_request: str,
    context_before: str = "",
    context_after: str = "",
) -> list:
    parts = [f'{file_path}, region "{block_name}":']
    if context_before:
        parts.append(f"--- context immediately before this region (do not include in output) ---\n{context_before}")
    parts.append(f"--- region content, regenerate only this ---\n{block_content}")
    if context_after:
        parts.append(f"--- context immediately after this region (do not include in output) ---\n{context_after}")
    parts.append(f"Request: {user_request}")
    return [
        {"role": "system", "content": _MODULE_BLOCK_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def generate_module_block_replacement(
    file_path: str,
    block_name: str,
    block_content: str,
    user_request: str,
    model: Optional[str] = None,
    context_before: str = "",
    context_after: str = "",
) -> dict:
    return _call_llm_for_full_file(
        build_module_block_messages(file_path, block_name, block_content, user_request, context_before, context_after),
        model,
    )
