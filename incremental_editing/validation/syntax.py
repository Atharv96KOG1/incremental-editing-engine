"""Cheapest validation gate: does the generated code even parse.

Python files use Python's own ast.parse() (zero extra dependency,
already proven, faster). Every other language goes through Tree-sitter --
one parser library with grammars for 100+ languages -- matching the
"Python AST initially, Tree-sitter later for multi-language" plan the
original architecture doc set out from day one. The language is detected
from the file's own extension; an extension Tree-sitter doesn't recognize
is left unvalidated (skipped, not failed) rather than guessed at.

Tree-sitter's parser is error-tolerant -- it always returns *some* tree,
even for invalid input -- so "did this parse" isn't an exception to catch,
it's "does the resulting tree contain an ERROR node or a MISSING token".
"""

import ast
import subprocess
import sys


class SyntaxCheckError(Exception):
    pass


def _check_python_syntax(source: str, filename: str) -> None:
    try:
        ast.parse(source, filename=filename)
    except SyntaxError as e:
        raise SyntaxCheckError(str(e)) from e


def _find_errors(node, limit: int = 5) -> list:
    errors = []

    def _walk(n) -> None:
        if len(errors) >= limit:
            return
        if n.type == "ERROR":
            errors.append(f"line {n.start_point[0] + 1}: unexpected syntax near '{n.text.decode('utf-8', 'replace')[:40]}'")
        elif n.is_missing:
            errors.append(f"line {n.start_point[0] + 1}: missing expected '{n.type}'")
        for child in n.children:
            _walk(child)

    _walk(node)
    return errors


def _check_tree_sitter_syntax(source: str, filename: str, language: str) -> None:
    from tree_sitter_language_pack import get_parser  # lazy: only needed for non-Python files

    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))

    errors = _find_errors(tree.root_node)
    if errors:
        raise SyntaxCheckError(f"{filename} ({language}): " + "; ".join(errors))


def check_syntax(source: str, filename: str = "<generated>") -> None:
    if filename.endswith(".py"):
        _check_python_syntax(source, filename)
        return

    from tree_sitter_language_pack import detect_language_from_path  # lazy: only needed for non-Python files

    language = detect_language_from_path(filename)
    if language is None:
        return  # unrecognized extension -- nothing we can validate, don't guess
    _check_tree_sitter_syntax(source, filename, language)


def check_python_imports(relpath: str, cwd: str, timeout: float = 15.0) -> None:
    """Genuinely imports the file (in a subprocess, cwd set to a real
    copy of the project so sibling imports resolve) and raises
    SyntaxCheckError if that fails -- ast.parse only proves the file is
    grammatically valid Python, not that every name it references
    actually exists where it's imported from.

    Real bug this catches: a mechanical rename (see
    run_pipeline._run_rename_identifier) correctly renamed a *class
    reference* (ChatOpenAI -> ChatAnthropic) but not the *import
    statement's own module path* it also appeared in, producing
    `from langchain_openai import ChatAnthropic` -- syntactically
    perfect Python, and a real ImportError the instant anything actually
    runs it. check_syntax alone reported this as passing; a project
    with no test suite covering this file's imports (common for a
    prototype, and true of the real case this was caught against) means
    pytest never imports it either, so nothing else in this pipeline
    would have caught it. Only ever runs a plain `import` -- real
    side-effecting work belongs behind `if __name__ == "__main__":`,
    which import itself never executes, same as pytest collection."""
    if not relpath.endswith(".py"):
        return
    module_name = relpath[:-3].replace("/", ".").replace("\\", ".")
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise SyntaxCheckError(f"importing '{relpath}' timed out after {timeout}s") from e
    if result.returncode != 0:
        raise SyntaxCheckError(f"'{relpath}' does not import cleanly: {result.stderr.strip()[-2000:]}")
