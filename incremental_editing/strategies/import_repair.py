"""IMPORT_REPAIR strategy: fix ONLY a broken import block, instead of
regenerating the whole file, when a mechanical rename (see
run_pipeline._run_rename_identifier) leaves an inconsistent reference --
a class/function/module name renamed everywhere it's *used*, but not
inside the import statement that names its own module path (real case:
"replace openai with xai" renamed OpenAI-the-class everywhere, but
`from openai import OpenAI` mechanically became `from xai import OpenAI`
-- syntactically valid, a real ModuleNotFoundError the instant it runs).

Before this existed, ANY import breakage here escalated straight to
`_run_whole_file_edit` -- a full file regeneration, discarding the
mechanical rename's own (otherwise entirely correct) output and asking
the model to reconstruct the *entire* file from scratch to fix a handful
of broken lines. Same reasoning TEXT_BLOCK_EDIT already applies to a
markdown/YAML section: locate the one contiguous region that's actually
wrong, send only that as input and expected output, splice it back by
line range. Falls back to the existing whole-file path automatically
(run_pipeline's own responsibility, not this module's) when this doesn't
resolve the import error.
"""

from typing import Optional

from .full_regeneration import _call_llm_for_full_file

_IMPORT_REPAIR_SYSTEM_PROMPT = (
    "Python import-line repair. Given ONLY a file's import block and the exact error importing it produces, "
    "output ONLY the corrected import block -- one valid import statement per logical import, nothing else, "
    "no prose, no markdown fences, no explanation. Fix exactly what the error names (a wrong module path, a "
    "renamed package that doesn't exist, a name that moved) and nothing else -- preserve every import that "
    "isn't implicated by the error exactly as it was, in the same order."
)


def build_import_repair_messages(file_path: str, import_block: str, error: str) -> list:
    return [
        {"role": "system", "content": _IMPORT_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{file_path}, import block:\n{import_block}\n\nImport error:\n{error}",
        },
    ]


def generate_import_fix(file_path: str, import_block: str, error: str, model: Optional[str] = None) -> dict:
    return _call_llm_for_full_file(build_import_repair_messages(file_path, import_block, error), model)
