"""IMPORT_REPAIR strategy: fix only a broken import block instead of
regenerating the whole file, for when a mechanical rename (see
run_pipeline._run_rename_identifier) renames a symbol everywhere it's used
but misses the import statement naming its own module path. Falls back to
whole-file edit (run_pipeline's responsibility) if this doesn't resolve it.
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
