"""Cheapest validation gate: does the generated code even parse. Python
uses `ast.parse()`; other languages go through Tree-sitter, detected from
the file's extension and skipped (not failed) when unrecognized.

Tree-sitter's parser is error-tolerant and always returns some tree, so
"did this parse" means checking for an ERROR node or MISSING token, not
catching an exception.
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
    from tree_sitter_language_pack import get_parser

    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))

    errors = _find_errors(tree.root_node)
    if errors:
        raise SyntaxCheckError(f"{filename} ({language}): " + "; ".join(errors))


def check_syntax(source: str, filename: str = "<generated>") -> None:
    if filename.endswith(".py"):
        _check_python_syntax(source, filename)
        return

    from tree_sitter_language_pack import detect_language_from_path

    language = detect_language_from_path(filename)
    if language is None:
        return
    _check_tree_sitter_syntax(source, filename, language)


def check_python_imports(relpath: str, cwd: str, timeout: float = 15.0) -> None:
    """Actually imports the file in a subprocess (cwd set to a real copy
    of the project) and raises SyntaxCheckError on failure -- ast.parse
    only proves the file is grammatically valid, not that every
    referenced name actually exists. Caught a real case: a mechanical
    rename updated a class reference but not its import statement's
    module path, producing a syntactically valid but unimportable file
    that no test suite would catch. Only ever runs a plain import; real
    side-effecting work belongs behind `if __name__ == "__main__":`."""
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
