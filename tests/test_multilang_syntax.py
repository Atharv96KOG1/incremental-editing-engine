"""Offline tests for multi-language syntax validation. Real Tree-sitter
parses (local, no network) for every language except Python, which keeps
using Python's own ast.parse() -- this is what actually fixed the reported
failure: a real .java create request was rejected because check_syntax()
used to run ast.parse() on every file regardless of language, and Java
code is never valid Python syntax.
"""

import pytest

from incremental_editing.validation.syntax import SyntaxCheckError, check_syntax

VALID_JAVA = """\
public class Foo {
    void bar() {
        int x = 1;
    }
}
"""

INVALID_JAVA = """\
public class Foo {
    void bar() {
        int x = 1
    }
}
"""

VALID_JS = "function add(a, b) {\n    return a + b;\n}\n"
INVALID_JS = "function add(a, b) {\n    return a + b\n"  # unclosed brace


def test_python_files_still_use_ast_and_reject_bad_syntax():
    with pytest.raises(SyntaxCheckError):
        check_syntax("def broken(:\n    pass\n", filename="calculator.py")
    check_syntax("def ok(a, b):\n    return a + b\n", filename="calculator.py")  # should not raise


def test_java_file_uses_tree_sitter_not_python_ast():
    """The exact reported bug: real, valid Java used to fail because
    ast.parse() rejects all Java as invalid Python."""
    check_syntax(VALID_JAVA, filename="management.java")  # should not raise


def test_java_file_still_catches_a_real_syntax_error():
    with pytest.raises(SyntaxCheckError, match="management.java"):
        check_syntax(INVALID_JAVA, filename="management.java")


def test_javascript_file_validates_via_tree_sitter():
    check_syntax(VALID_JS, filename="app.js")  # should not raise
    with pytest.raises(SyntaxCheckError):
        check_syntax(INVALID_JS, filename="app.js")


def test_unrecognized_extension_is_skipped_not_failed():
    """No language detected -- nothing we can validate, so this must not
    raise just because the extension is unknown."""
    check_syntax("this is not code in any language {{{", filename="notes.wxyz123")
