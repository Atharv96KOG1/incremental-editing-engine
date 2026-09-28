"""Offline engine tests -- no LLM call, no MinIO/docker required.

Covers the deterministic half of the pipeline: schema validation, target
resolution, patch application, and local storage/versioning. The
LLM-generation half is exercised manually via run_pipeline against a real
OPENAI_API_KEY.
"""

import json
import os
import re
import tempfile
from pathlib import Path

import pytest

from incremental_editing.analyzer.locator import AmbiguousSymbolError
from incremental_editing.api.run_pipeline import (
    _classify_failure,
    _discover_relevant_test_target,
    _effective_test_target,
    _first_defined_name,
    _TestFailure,
    run_edit,
)
from incremental_editing.apply.edit_applier import ApplyError, apply_delta
from incremental_editing.delta.schema import DeltaIR
from incremental_editing.delta.validator import (
    DeltaValidationError,
    UnseenReplaceTargetError,
    validate_schema,
    validate_targets,
)
from incremental_editing.storage.minio_client import LocalStorage
from incremental_editing.validation.syntax import SyntaxCheckError
from incremental_editing.versioning.version_manager import VersionManager

# Frozen literal, not read from sample_project/calculator.py: that file is a
# live demo project mutated by manual `iee edit` runs, so a test fixture
# reading it off disk would drift and break as the demo evolves.
CALCULATOR_SOURCE = """\
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b
"""

CALCULATOR_TEST_SOURCE = """\
from calculator import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 2) == 3


def test_multiply():
    assert multiply(4, 3) == 12


def test_divide():
    assert divide(10, 2) == 5


def test_divide_by_zero():
    assert divide(4, 0) == 0
"""

DIVIDE_FIX_DELTA = {
    "schema_version": "1.0",
    "base_version": "v0",
    "operations": [
        {
            "operation": "REPLACE",
            "target": {"file": "calculator.py", "symbol_type": "function", "symbol_name": "divide"},
            "content": "def divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\n",
        }
    ],
}


def test_schema_validation_accepts_valid_delta():
    validate_schema(DIVIDE_FIX_DELTA)  # should not raise


def test_schema_validation_rejects_bad_operation():
    bad = {**DIVIDE_FIX_DELTA, "operations": [{**DIVIDE_FIX_DELTA["operations"][0], "operation": "OVERWRITE"}]}
    with pytest.raises(DeltaValidationError):
        validate_schema(bad)


def test_schema_validation_accepts_empty_operations():
    """A model can correctly decide there's nothing to do -- e.g. asked to
    remove a function that's already gone. That must be a valid, empty
    delta (the pipeline treats it as a no-op success), not a schema error."""
    no_op_delta = {**DIVIDE_FIX_DELTA, "operations": []}
    validate_schema(no_op_delta)  # should not raise
    delta = DeltaIR.from_dict(no_op_delta)
    assert delta.operations == []


def test_validate_targets_rejects_missing_symbol():
    delta = DeltaIR.from_dict(DIVIDE_FIX_DELTA)
    delta.operations[0].target.symbol_name = "does_not_exist"
    with pytest.raises(DeltaValidationError):
        validate_targets(delta, CALCULATOR_SOURCE)


def test_validate_targets_refuses_an_insert_that_would_duplicate_an_existing_name():
    """Real corruption this retroactively catches: a class-method version
    of a function got INSERTed (as part of a broader refactor) alongside
    an old top-level definition that was never removed -- both silently
    coexisted in the file until something later tried to REPLACE/DELETE
    that name and hit an unresolvable "defined 2 times" error, with no
    way to tell which occurrence was meant. An INSERT whose own content
    defines a name that already exists elsewhere must be refused up
    front, regardless of what the model's target.symbol_name claims."""
    source = "def divide(a, b):\n    return a / b\n"
    delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "INSERT",
                    "target": {"file": "calculator.py", "symbol_type": "function", "symbol_name": "divide"},
                    "content": "def divide(a, b):\n    return a / b if b else 0\n",
                }
            ],
        }
    )
    with pytest.raises(DeltaValidationError, match="already exists"):
        validate_targets(delta, source)


def test_validate_targets_allows_an_insert_of_a_genuinely_new_name():
    """The check above must not become over-broad and refuse ordinary,
    correct INSERTs of a name that doesn't exist yet."""
    source = "def divide(a, b):\n    return a / b\n"
    delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "INSERT",
                    "target": {"file": "calculator.py", "symbol_type": "function", "symbol_name": "multiply", "anchor": "divide"},
                    "content": "def multiply(a, b):\n    return a * b\n",
                }
            ],
        }
    )
    validate_targets(delta, source)  # must not raise


def test_validate_targets_refuses_replace_of_a_symbol_whose_body_was_never_shown():
    """Real data loss this catches: a request matched no localized
    candidate, so build_context fell back to the compact context
    (imports + bare names, no bodies). The model still emitted a REPLACE
    for one of those bare names -- since REPLACE's content must be the
    *complete* new body and the model never saw the real one, it
    fabricated a smaller function, silently dropping real content (an
    extra table, half the original columns) it was never shown. A
    REPLACE naming a real symbol outside content_shown_for must be
    refused, raising the specific UnseenReplaceTargetError subtype so
    run_pipeline.py can react by escalating to whole-file regeneration."""
    delta = DeltaIR.from_dict(DIVIDE_FIX_DELTA)
    with pytest.raises(UnseenReplaceTargetError, match="never shown"):
        validate_targets(delta, CALCULATOR_SOURCE, content_shown_for=set())


def test_validate_targets_allows_replace_of_a_symbol_whose_body_was_shown():
    """The guard must not become over-broad and refuse an ordinary
    REPLACE of a symbol that genuinely was localized and shown."""
    delta = DeltaIR.from_dict(DIVIDE_FIX_DELTA)
    shown = {delta.operations[0].target.symbol_name}
    validate_targets(delta, CALCULATOR_SOURCE, content_shown_for=shown)  # must not raise


def test_validate_targets_exempts_delete_and_insert_from_the_shown_guard():
    """DELETE needs no content (nothing to rewrite blind) and INSERT's
    content is new by definition, not a rewrite of something unseen --
    content_shown_for must only ever gate REPLACE."""
    delete_delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "DELETE",
                    "target": {"file": "calculator.py", "symbol_type": "function", "symbol_name": "add"},
                }
            ],
        }
    )
    validate_targets(delete_delta, CALCULATOR_SOURCE, content_shown_for=set())  # must not raise


def test_validate_targets_refuses_a_replace_that_drops_the_original_decorator():
    """Real regression this catches live: a REPLACE for a Flask route
    handler dropped its own '@app.route(...)' decorator -- the apply
    engine spliced the decorator-less content right over the original
    span (which includes the decorator line, by design), silently
    un-registering that route. The file still parsed and every other
    test kept passing; nothing surfaced the loss until the endpoint
    itself 404'd. A REPLACE must restate every decorator the original
    had."""
    source = (
        "@app.route('/health', methods=['GET'])\n"
        "def health():\n"
        "    return {'status': 'ok', 'minio': 'x'}\n"
    )
    delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "REPLACE",
                    "target": {"file": "app.py", "symbol_type": "function", "symbol_name": "health"},
                    "content": "def health():\n    return {'status': 'ok'}\n",
                }
            ],
        }
    )
    with pytest.raises(DeltaValidationError, match="missing its original decorator"):
        validate_targets(delta, source)


def test_validate_targets_allows_a_replace_that_keeps_the_original_decorator():
    source = (
        "@app.route('/health', methods=['GET'])\n"
        "def health():\n"
        "    return {'status': 'ok', 'minio': 'x'}\n"
    )
    delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "REPLACE",
                    "target": {"file": "app.py", "symbol_type": "function", "symbol_name": "health"},
                    "content": "@app.route('/health', methods=['GET'])\ndef health():\n    return {'status': 'ok'}\n",
                }
            ],
        }
    )
    validate_targets(delta, source)  # must not raise


def test_apply_delta_replaces_only_target_function():
    delta = DeltaIR.from_dict(DIVIDE_FIX_DELTA)
    new_source = apply_delta(CALCULATOR_SOURCE, delta)

    assert "if b == 0:" in new_source
    assert "def add(a, b):\n    return a + b" in new_source  # untouched
    ns = {}
    exec(compile(new_source, "calculator.py", "exec"), ns)
    assert ns["divide"](4, 0) == 0
    assert ns["divide"](10, 2) == 5


def test_apply_delta_raises_on_unknown_target():
    delta = DeltaIR.from_dict(DIVIDE_FIX_DELTA)
    delta.operations[0].target.symbol_name = "ghost_function"
    with pytest.raises(ApplyError):
        apply_delta(CALCULATOR_SOURCE, delta)


def test_apply_delta_reindents_class_method_regardless_of_model_output():
    """Regression test: a model asked to REPLACE a class method returned
    the new body de-indented to column 0. Splicing that in verbatim parses
    fine (ast.parse doesn't care) but silently turns the method into a
    module-level function, un-nesting every method defined after it. The
    apply engine must force the target's original indentation, not trust
    whatever the model sent."""
    class_source = (
        "class Greeter:\n"
        "    def __init__(self):\n"
        "        self.name = 'x'\n"
        "\n"
        "    def greet(self):\n"
        "        return 'hi'\n"
        "\n"
        "    def farewell(self):\n"
        "        return 'bye'\n"
    )
    delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "REPLACE",
                    "target": {"file": "greeter.py", "symbol_type": "function", "symbol_name": "greet"},
                    # De-indented to column 0, as an actual model response did.
                    "content": "def greet(self):\n    return 'hello'\n",
                }
            ],
        }
    )
    new_source = apply_delta(class_source, delta)

    ns = {}
    exec(compile(new_source, "greeter.py", "exec"), ns)
    instance = ns["Greeter"]()
    assert instance.greet() == "hello"
    assert instance.farewell() == "bye"  # must still be a method of the class, not orphaned


def test_apply_delta_refuses_ambiguous_duplicate_target():
    """A name defined twice must be refused, not silently resolved to the
    first (possibly dead/shadowed) match -- Python keeps only the last."""
    duplicate_divide_source = CALCULATOR_SOURCE + "\n\ndef divide(a, b):\n    return 0\n"
    delta = DeltaIR.from_dict(DIVIDE_FIX_DELTA)
    with pytest.raises(ApplyError, match="defined 2 times"):
        apply_delta(duplicate_divide_source, delta)
    with pytest.raises(DeltaValidationError, match="defined 2 times"):
        validate_targets(delta, duplicate_divide_source)


def test_delta_ir_strips_call_signature_a_model_echoed_as_symbol_name():
    """A real generation was observed copying a whole "name(params)" or
    "name(params) @decorator" entry straight out of context_builder.py's
    "other symbols defined in this file" line -- meant only to disambiguate
    -- as symbol_name/anchor verbatim, instead of the bare identifier. That
    failed REPLACE lookup with "target ... not found", and every repair
    attempt repeated the identical mistake since nothing told the model
    what was wrong. DeltaIR.from_dict now normalizes both fields so this
    class of mistake can't fail a run at all, independent of the prompt
    wording that also warns against it."""
    delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "REPLACE",
                    "target": {"file": "x.py", "symbol_type": "function", "symbol_name": "load_api_key()"},
                    "content": "def load_api_key():\n    return None\n",
                },
                {
                    "operation": "INSERT",
                    "target": {
                        "file": "x.py",
                        "symbol_type": "function",
                        "symbol_name": "new_fn",
                        "anchor": "build_client() @staticmethod",
                    },
                    "content": "def new_fn():\n    pass\n",
                },
                {
                    "operation": "REPLACE",
                    "target": {"file": "x.py", "symbol_type": "function", "symbol_name": "add"},
                    "content": "def add(a, b):\n    return a + b\n",
                },
            ],
        }
    )
    assert delta.operations[0].target.symbol_name == "load_api_key"
    assert delta.operations[1].target.anchor == "build_client"
    assert delta.operations[2].target.symbol_name == "add"  # already bare -- untouched


def test_apply_delta_delete_leaves_no_stray_blank_lines():
    """DELETE used to splice in a single blank line where the symbol used
    to be (content defaults to None, and the old code always spliced in
    `_reindent(None, indent)` regardless of operation) -- on top of the
    file's own existing blank-line separator, that left two consecutive
    blank lines at the junction. A delete should look like the function
    was cleanly excised, not blanked out in place."""
    src = "def foo():\n    return 1\n\ndef bar():\n    return 2\n\ndef baz():\n    return 3\n"
    delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [{"operation": "DELETE", "target": {"file": "x.py", "symbol_type": "function", "symbol_name": "bar"}}],
        }
    )
    new_source = apply_delta(src, delta)
    assert new_source == "def foo():\n    return 1\n\ndef baz():\n    return 3\n"
    assert "\n\n\n" not in new_source


def test_apply_delta_delete_matches_existing_blank_line_convention():
    """A file consistently using two blank lines between top-level defs
    (PEP8's convention) should still have two blank lines at the junction
    after a delete, not three -- the fix absorbs the deleted symbol's own
    trailing blank-line run entirely and leaves the leading gap (whatever
    it was) as the separator, so it self-adjusts to the file's own style
    instead of hardcoding a blank-line count."""
    src = "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n\n\ndef baz():\n    return 3\n"
    delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [{"operation": "DELETE", "target": {"file": "x.py", "symbol_type": "function", "symbol_name": "bar"}}],
        }
    )
    new_source = apply_delta(src, delta)
    assert new_source == "def foo():\n    return 1\n\n\ndef baz():\n    return 3\n"


def test_apply_delta_insert_after_anchor():
    insert_delta = DeltaIR.from_dict(
        {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "INSERT",
                    "target": {
                        "file": "calculator.py",
                        "symbol_type": "function",
                        "symbol_name": "power",
                        "anchor": "multiply",
                    },
                    "content": "def power(a, b):\n    return a ** b\n",
                }
            ],
        }
    )
    new_source = apply_delta(CALCULATOR_SOURCE, insert_delta)
    ns = {}
    exec(compile(new_source, "calculator.py", "exec"), ns)
    assert ns["power"](2, 3) == 8
    assert ns["multiply"](4, 3) == 12  # anchor untouched


def test_version_manager_creates_incrementing_versions(tmp_path):
    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    vm = VersionManager(storage, project_id="calc-test")

    assert vm.get_head() is None

    v1 = vm.create_version(
        files={"calculator.py": CALCULATOR_SOURCE},
        change_request="initial version",
        strategy="STRUCTURED_EDIT",
        delta_id="delta-000",
        validation_status="passed",
    )
    assert v1 == "v1"
    assert vm.get_head() == "v1"

    v2 = vm.create_version(
        files={"calculator.py": CALCULATOR_SOURCE},
        change_request="divide by zero fix",
        strategy="STRUCTURED_EDIT",
        delta_id="delta-001",
        validation_status="passed",
    )
    assert v2 == "v2"
    stored = storage.get_text(f"projects/calc-test/checkpoints/v2/files/calculator.py")
    assert stored == CALCULATOR_SOURCE

    version_meta = storage.get_json("projects/calc-test/versions/v2.json")
    assert version_meta["parent_version"] == "v1"


def test_full_pipeline_offline_against_sample_project(tmp_path):
    """Simulates the pipeline's apply+validate+test steps using a canned delta
    instead of a live LLM call -- proves the loop is wired correctly end to end."""
    project_copy = tmp_path / "sample_project"
    project_copy.mkdir()
    (project_copy / "calculator.py").write_text(CALCULATOR_SOURCE)
    (project_copy / "test_calculator.py").write_text(CALCULATOR_TEST_SOURCE)

    original = (project_copy / "calculator.py").read_text()
    delta = DeltaIR.from_dict(DIVIDE_FIX_DELTA)
    validate_targets(delta, original)
    patched = apply_delta(original, delta)
    (project_copy / "calculator.py").write_text(patched)

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", ".", "-q"], cwd=str(project_copy), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_classify_failure_maps_each_exception_to_its_phoenix_class():
    """Bounded self-repair (PHOENIX doc section 18) needs a stable failure
    class per exception so the repair prompt names the right problem."""
    assert _classify_failure(SyntaxCheckError("bad syntax")) == "SYNTAX_ERROR"
    assert _classify_failure(ApplyError("cannot apply")) == "PATCH_CONFLICT"
    assert _classify_failure(_TestFailure("1 failed")) == "TEST_FAILURE"
    assert _classify_failure(DeltaValidationError("schema validation failed: x")) == "SCHEMA_ERROR"
    assert _classify_failure(DeltaValidationError("target 'x' not found for REPLACE")) == "REFERENCE_ERROR"
    assert _classify_failure(AmbiguousSymbolError("function 'x' is defined 2 times")) == "REFERENCE_ERROR"
    assert _classify_failure(ValueError("something else entirely")) == "UNKNOWN"


def test_first_defined_name_reads_the_real_name_from_content():
    """Regression test: a real run's INSERT declared target.symbol_name as
    'asin_inverse' while its content actually defined 'acsc' -- apply_delta
    never checks symbol_name for INSERT (only anchor), so the code applied
    fine, but every display showed the wrong name. The pipeline must read
    the *real* name back out of content instead of trusting the label."""
    content = "def acsc(x):\n    import math\n    return math.asin(1 / x)"
    assert _first_defined_name(content) == "acsc"
    assert _first_defined_name(None) is None
    assert _first_defined_name("not valid python (((") is None
    assert _first_defined_name("x = 1\ny = 2") is None  # no def/class at all


def test_run_edit_skips_new_version_when_delta_produces_no_actual_change(tmp_path, monkeypatch):
    """A REPLACE that regenerates a symbol byte-identical to what was
    already there is a real, observed model behavior -- it declares
    operations, so it isn't the empty-operations no_op case, but applying
    it changes nothing. Versioning it anyway would grow the version chain
    for zero actual file change. Must be treated as a no-op: no new
    version, no version-manager write, no head bump."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "calculator.py").write_text(CALCULATOR_SOURCE)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    identical_delta = {
        "schema_version": "1.0",
        "base_version": "v0",
        "operations": [
            {
                "operation": "REPLACE",
                "target": {"file": "calculator.py", "symbol_type": "function", "symbol_name": "add"},
                "content": "def add(a, b):\n    return a + b\n",
            }
        ],
    }
    canned_gen = {
        "raw_json": "{}",
        "delta_dict": identical_delta,
        "input_tokens": 10,
        "output_tokens": 10,
        "total_tokens": 20,
        "latency_ms": 1,
        "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)

    metadata = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="no-op change to add",  # names "add" so it actually localizes -- content_shown_for
        # would otherwise refuse this REPLACE as targeting a symbol whose body was never shown
        test_target="nonexistent",  # no tests to collect -- isolates this from pytest's own result
        project_id="noop-test",
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["result"]["no_op"] is True
    assert "new_version" not in metadata
    assert VersionManager(storage, "noop-test").get_head() is None
    assert (project_dir / "calculator.py").read_text() == CALCULATOR_SOURCE


def test_run_edit_fails_a_delete_that_breaks_the_test_suite_instead_of_reverting(tmp_path, monkeypatch):
    """Real failure: "remove add operation" correctly generated a DELETE
    for `add`, which broke the project's own test_add() (an ImportError,
    not a logic bug in the edit). The repair loop "fixed" that failure
    the only way it structurally could -- by regenerating `add` right
    back -- and the run still reported plain "success" (change_ratio 0,
    reason buried in a `note` field), silently reverting the user's
    actual request. A delete that conflicts with an existing test must
    fail honestly instead, with no repair round-trip spent on it (there's
    nothing to correct -- only code elsewhere that still depends on what
    was asked to be removed)."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "calc.py").write_text("def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n")
    (project_dir / "test_calc.py").write_text(
        "from calc import add, subtract\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_subtract():\n    assert subtract(5, 2) == 3\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    delete_add = {
        "schema_version": "1.0",
        "base_version": "v0",
        "operations": [{"operation": "DELETE", "target": {"file": "calc.py", "symbol_type": "function", "symbol_name": "add"}}],
    }
    canned_gen = {
        "raw_json": "{}", "delta_dict": delete_add,
        "input_tokens": 10, "output_tokens": 10, "total_tokens": 20, "latency_ms": 1, "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)

    def _fail_if_called(**kwargs):
        raise AssertionError("a delete conflicting with an existing test must not spend a repair round-trip")

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_repair", _fail_if_called)

    metadata = run_edit(
        project_dir=project_dir,
        file="calc.py",
        request="remove add operation",
        test_target=".",
        project_id="delete-conflict-test",
    )

    assert metadata["result"]["status"] == "failed"
    assert metadata["result"]["retry_count"] == 0  # failed on the first attempt, no repair spent
    assert "add" in metadata["error"] and "test" in metadata["error"].lower()
    assert "def add" in (project_dir / "calc.py").read_text()  # never written -- still there, unchanged
    assert VersionManager(storage, "delete-conflict-test").get_head() is None


def test_run_edit_fails_fast_on_an_unrelated_test_collection_error_instead_of_burning_repair_attempts(
    tmp_path, monkeypatch
):
    """Real waste this closes: test_target scoped wider than the actual
    project (a real, observed case: an unrelated sample_project/
    test_calculator.py with its own pre-existing ModuleNotFoundError)
    means every single edit's TEST step hits the identical pytest
    COLLECTION error -- interrupted before a single test even runs, and
    utterly unrelated to whatever the delta just changed. The repair
    loop used to spend its full budget (2 extra LLM calls, real cost)
    retrying the exact same unfixable failure every time. A real pytest
    subprocess run here (not mocked) reproduces the actual collection
    error shape."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    # A genuinely broken, unrelated test file -- mirrors the real
    # sample_project/test_calculator.py case exactly (imports a module
    # that was never actually placed on pytest's collection path).
    (project_dir / "test_unrelated.py").write_text("from nonexistent_module import whatever\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    canned_gen = {
        "raw_json": "{}",
        "delta_dict": {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "REPLACE",
                    "target": {"file": "app.py", "symbol_type": "function", "symbol_name": "add"},
                    "content": "def add(a, b):\n    return a + b + 0\n",
                }
            ],
        },
        "input_tokens": 10, "output_tokens": 10, "total_tokens": 20, "latency_ms": 1, "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)

    def _fail_if_called(**kwargs):
        raise AssertionError("an unrelated collection error must not spend any repair round-trip")

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_repair", _fail_if_called)

    metadata = run_edit(
        project_dir=project_dir,
        file="app.py",
        request="make add also accept a default of 0",
        test_target=".",
        project_id="unrelated-collection-error-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "failed"
    assert metadata["result"]["retry_count"] == 0  # failed immediately, no repair spent
    assert metadata["result"]["failure_class"] == "UNRELATED_TEST_COLLECTION_ERROR"
    assert "test_unrelated.py" in metadata["error"]
    assert "unrelated to this edit of 'app.py'" in metadata["error"]


def test_discover_relevant_test_target_finds_a_co_located_test_file(tmp_path):
    (tmp_path / "cores").mkdir()
    (tmp_path / "cores" / "newchatbot.py").write_text("x = 1\n")
    (tmp_path / "cores" / "test_newchatbot.py").write_text("def test_x(): pass\n")
    assert _discover_relevant_test_target(tmp_path, "cores/newchatbot.py") == "cores/test_newchatbot.py"


def test_discover_relevant_test_target_finds_a_sibling_tests_directory(tmp_path):
    (tmp_path / "cores").mkdir()
    (tmp_path / "cores" / "newchatbot.py").write_text("x = 1\n")
    (tmp_path / "cores" / "tests").mkdir()
    (tmp_path / "cores" / "tests" / "test_newchatbot.py").write_text("def test_x(): pass\n")
    assert _discover_relevant_test_target(tmp_path, "cores/newchatbot.py") == "cores/tests/test_newchatbot.py"


def test_discover_relevant_test_target_returns_none_when_nothing_matches(tmp_path):
    (tmp_path / "cores").mkdir()
    (tmp_path / "cores" / "newchatbot.py").write_text("x = 1\n")
    assert _discover_relevant_test_target(tmp_path, "cores/newchatbot.py") is None


def test_effective_test_target_narrows_the_broad_default_to_the_files_own_directory(tmp_path):
    """Real bug this closes: project_dir pointed at a large, cluttered
    directory (other subprojects, scratch folders) made every edit's
    own validation pytest '.' the whole thing -- slow, and reporting
    failures entirely unrelated to the actual change. Narrows down to
    the edited file's own containing directory instead, real and
    already-existing, never invented."""
    assert _effective_test_target(".", tmp_path, "cores/newchatbot.py") == "cores"


def test_effective_test_target_prefers_a_discovered_test_file_over_the_directory(tmp_path):
    (tmp_path / "cores").mkdir()
    (tmp_path / "cores" / "test_newchatbot.py").write_text("def test_x(): pass\n")
    assert _effective_test_target(".", tmp_path, "cores/newchatbot.py") == "cores/test_newchatbot.py"


def test_effective_test_target_never_touches_an_explicit_non_default_value(tmp_path):
    assert _effective_test_target("some/explicit/path", tmp_path, "cores/newchatbot.py") == "some/explicit/path"


def test_effective_test_target_leaves_a_root_level_file_unchanged(tmp_path):
    """No narrower real target exists for a file that already lives at
    the project root -- correctly stays "." rather than narrowing to a
    no-op that hides the same broad scope under a different label."""
    assert _effective_test_target(".", tmp_path, "app.py") == "."


def test_run_edit_narrows_test_target_away_from_an_unrelated_sibling_directory(tmp_path, monkeypatch):
    """End-to-end version of the two unit tests above: a real, genuinely
    broken/unrelated test suite sits in a SEPARATE top-level directory
    from the file actually being edited -- the old behavior (test_target
    "." run against the whole project_dir) would have hit it; the
    narrowed target (just the edited file's own directory) never does."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "cores").mkdir()
    (project_dir / "cores" / "app.py").write_text("def add(a, b):\n    return a + b\n")
    # Unrelated, broken, and in a DIFFERENT directory than the edited file --
    # the old, unnarrowed "." target would have collected this and failed.
    (project_dir / "unrelated_dir").mkdir()
    (project_dir / "unrelated_dir" / "test_broken.py").write_text("from nonexistent_module import whatever\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    canned_gen = {
        "raw_json": "{}",
        "delta_dict": {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "REPLACE",
                    "target": {"file": "cores/app.py", "symbol_type": "function", "symbol_name": "add"},
                    "content": "def add(a, b):\n    return a + b + 0\n",
                }
            ],
        },
        "input_tokens": 10, "output_tokens": 10, "total_tokens": 20, "latency_ms": 1, "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)

    metadata = run_edit(
        project_dir=project_dir,
        file="cores/app.py",
        request="make add also accept a default of 0",
        test_target=".",
        project_id="narrow-test-target-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"


def test_run_edit_asks_before_deleting_an_ambiguous_target(tmp_path, monkeypatch):
    """Real Bifrost log: "remove atan function" against a file full of
    similarly-named trig functions. A deletion is destructive and
    permanent, so when the target word matches more than one real symbol,
    the pipeline must ask instead of guessing -- and asking must cost
    nothing, so the LLM is never called for either the initial ambiguity
    check or the confirmed follow-up (a DELETE's shape is fully
    determined once the target is known)."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    source = (
        "def atan2(y, x):\n    import math\n    return math.atan2(y, x)\n\n\n"
        "def atan_inverse(x):\n    import math\n    return math.atan(x)\n"
    )
    (project_dir / "calculator.py").write_text(source)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    def _fail_if_called(**kwargs):
        raise AssertionError("generate_delta must not be called for a needs_selection or confirmed-delete run")

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", _fail_if_called)

    ambiguous = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="remove the atan function",
        project_id="delete-test",
    )
    assert ambiguous["result"]["status"] == "needs_selection"
    names = {c["name"] for c in ambiguous["candidates"]}
    assert names == {"atan2", "atan_inverse"}

    confirmed = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="remove the atan function",
        project_id="delete-test",
        confirm_symbol="atan2",
        confirm_symbol_type="function",
    )
    assert confirmed["result"]["status"] == "success"
    assert confirmed["generation"]["total_tokens"] == 0
    assert "def atan2" not in (project_dir / "calculator.py").read_text()
    assert "def atan_inverse" in (project_dir / "calculator.py").read_text()


def test_run_edit_confirmed_delete_resolves_a_duplicate_name_via_line(tmp_path, monkeypatch):
    """Real failure: a name defined 3 times (accumulated real edit-tool
    cruft in a live file) collapsed to one needs_selection chip; picking
    it and confirming still failed with "defined N times", since the
    bare name alone is inherently ambiguous no matter how many times it's
    reconfirmed -- confirm_symbol on its own can never resolve a
    duplicate. Each occurrence must be its own candidate (with its own
    real start_line), and confirm_symbol_line must let the confirmed
    delete resolve the *specific* occurrence, deleting only that one."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    source = (
        "def add(a, b):\n    return a + b\n\n\n"
        "def dup(x):\n    return 1\n\n\n"
        "def dup(x):\n    return 2\n\n\n"
        "def dup(x):\n    return 3\n"
    )
    (project_dir / "calculator.py").write_text(source)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    def _fail_if_called(**kwargs):
        raise AssertionError("a confirmed delete must never call the LLM")

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", _fail_if_called)

    ambiguous = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="remove the dup function",
        project_id="dup-test",
    )
    assert ambiguous["result"]["status"] == "needs_selection"
    dup_candidates = [c for c in ambiguous["candidates"] if c["name"] == "dup"]
    assert len(dup_candidates) == 3  # every occurrence, not collapsed to one
    assert {c["start_line"] for c in dup_candidates} == {5, 9, 13}

    # Confirm the *middle* occurrence specifically.
    middle = next(c for c in dup_candidates if c["start_line"] == 9)
    confirmed = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="remove the dup function",
        project_id="dup-test",
        confirm_symbol="dup",
        confirm_symbol_type="function",
        confirm_symbol_line=middle["start_line"],
    )
    assert confirmed["result"]["status"] == "success"
    new_source = (project_dir / "calculator.py").read_text()
    assert new_source.count("def dup") == 2  # exactly one deleted
    assert "return 1" in new_source and "return 3" in new_source  # the other two survive
    assert "return 2" not in new_source  # the confirmed (middle) one is gone


def test_run_edit_confirmed_delete_fails_cleanly_when_target_no_longer_exists(tmp_path, monkeypatch):
    """A confirmed delete's target can go stale between the disambiguation
    prompt and the follow-up request (the file changed in between, or a
    caller passes a bad symbol name directly) -- this used to crash with
    a raw KeyError ('context' was never set on the confirm_symbol path's
    ctx dict, but the repair-retry branch unconditionally read it) that
    would have surfaced as an ugly, unhelpful error in the dashboard.
    Retrying via the LLM makes no sense here either -- a confirmed
    delete's shape is already fully determined, there's nothing for a
    repair prompt to productively change. Must fail immediately with a
    clear message, no LLM call, no crash."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "calculator.py").write_text(CALCULATOR_SOURCE)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    def _fail_if_called(**kwargs):
        raise AssertionError("a confirmed delete must never call the LLM, including on repair")

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", _fail_if_called)
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_repair", _fail_if_called)

    result = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="remove the nonexistent function",
        project_id="stale-test",
        confirm_symbol="does_not_exist",
        confirm_symbol_type="function",
    )
    assert result["result"]["status"] == "failed"
    assert result["result"]["retry_count"] == 0
    assert "does_not_exist" in result["error"]
    assert VersionManager(storage, "stale-test").get_head() is None


def test_run_edit_refreshes_the_per_file_metadata_json_on_a_real_commit(tmp_path, monkeypatch):
    """Every real commit (not a preview, not a rejected/awaiting run) must
    leave <project_dir>/iee_metadata/<file>.json reflecting the file's
    *current* symbols -- the deleted symbol must not linger in it."""
    from incremental_editing.analyzer.metadata_builder import per_file_metadata_path

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "calculator.py").write_text(CALCULATOR_SOURCE)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    metadata = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="remove the subtract function",
        test_target=".",
        project_id="metadata-refresh-test",
        require_confirmation=False,
    )
    assert metadata["result"]["status"] == "success"

    meta_path = per_file_metadata_path(str(project_dir), "calculator.py")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        doc = json.load(f)
    names = {s["name"] for s in doc["symbols"]}
    assert "subtract" not in names
    assert "add" in names  # untouched symbols still present


def test_run_edit_whole_file_delete_removes_the_metadata_json_too(tmp_path, monkeypatch):
    from incremental_editing.analyzer.metadata_builder import per_file_metadata_path, write_file_metadata

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "scratch.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    write_file_metadata(str(project_dir), "scratch.py", "VALUE = 1\n")
    meta_path = per_file_metadata_path(str(project_dir), "scratch.py")
    assert os.path.exists(meta_path)

    metadata = run_edit(
        project_dir=project_dir,
        file="scratch.py",
        request="delete this file",
        test_target=".",
        project_id="metadata-delete-test",
        require_confirmation=False,
    )
    assert metadata["result"]["status"] == "success"
    assert not os.path.exists(meta_path)


def test_run_edit_deletes_the_whole_file_with_zero_llm_calls(tmp_path, monkeypatch):
    """"delete this file" (or naming the file itself, e.g. "delete
    scratch.py") means the file, not a symbol inside it -- and once
    that's true there's nothing left for a model to decide, so this must
    never call the LLM at all, the same zero-token guarantee an
    unambiguous code-level delete already gets."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "scratch.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    def _fail_if_called(**kwargs):
        raise AssertionError("must not call the LLM for a whole-file delete")

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", _fail_if_called)

    metadata = run_edit(
        project_dir=project_dir,
        file="scratch.py",
        request="delete this file",
        test_target=".",
        project_id="file-delete-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["strategy"] == "FILE_DELETE"
    assert metadata["generation"]["total_tokens"] == 0
    assert not (project_dir / "scratch.py").exists()
    assert VersionManager(storage, "file-delete-test").get_head() == "v1"


def test_run_edit_escalates_to_delete_file_for_a_phrasing_the_fast_path_misses(tmp_path, monkeypatch):
    """Real gap this closes: "please delete this file" doesn't match
    is_whole_file_delete_target's narrow, zero-LLM verb-first fast path
    (the leading word isn't a bare "delete"/"remove"), so it used to fall
    through to the normal generate_delta call with no way to represent
    "delete the whole file" at all -- the model could only return empty
    operations, silently leaving the file untouched. Must now escalate
    via {"kind":"delete_file"} and actually delete it, folding the
    escalate call's own token cost in rather than dropping it."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "scratch.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen({"kind": "delete_file"}),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="scratch.py",
        request="please delete this file",
        test_target=".",
        project_id="delete-file-escalate-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["strategy"] == "FILE_DELETE"
    assert metadata["generation"]["total_tokens"] == 15  # the escalate call's own cost, folded in
    assert not (project_dir / "scratch.py").exists()
    assert VersionManager(storage, "delete-file-escalate-test").get_head() == "v1"


def test_run_edit_uses_text_block_edit_for_a_locatable_section(tmp_path, monkeypatch):
    """Real waste this closes: "add retrieval types in the notes part"
    against a 30-line markdown README paid ~1045 tokens for
    STRUCTURED_EDIT's classification call's system prompt alone (markdown
    has no function/class concept, so that call could only ever answer
    "escalate"), then regenerated the entire file to add 3 words to one
    section. Must skip the classification call AND only send/regenerate
    the one located block ("### Notes"), not the whole file."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "readme.md").write_text(
        "# Project\n\n## Features\n\n- feature one\n\n### Notes\n\n- existing note\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    def _fail_if_called(**kwargs):
        raise AssertionError("must not call generate_delta for a language with no symbol concept")

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", _fail_if_called)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file_edit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not regenerate the whole file when a block locates")),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_block_replacement",
        lambda file_path, block_name, block_content, user_request, **kwargs: {
            "code": block_content + "\n- retrieval types: dense, sparse, hybrid",
            "input_tokens": 30, "output_tokens": 15, "total_tokens": 45, "cached_tokens": 0,
            "latency_ms": 1, "model": "test-model",
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="readme.md",
        request="add retrieval types in the notes part",
        test_target=".",
        project_id="text-block-edit-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["strategy"] == "TEXT_BLOCK_EDIT"
    assert metadata["block"] == "Notes"
    assert metadata["generation"]["total_tokens"] == 45  # only the one small block-scoped call
    new_source = (project_dir / "readme.md").read_text()
    assert "retrieval types" in new_source
    assert "## Features" in new_source and "- feature one" in new_source  # untouched


def test_run_edit_text_block_edit_passes_neighboring_blocks_as_context(tmp_path, monkeypatch):
    """The block-scoped generation call must see the section immediately
    before and after the one it's regenerating -- read-only context, not
    part of what it should output -- so a boundary-sensitive request has
    something real to match against, same chunk-overlap reasoning
    context_builder.py's own "other symbols" name line already applies to
    code symbols."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "readme.md").write_text(
        "# Project\n\n## Features\n\n- feature one\n\n### Notes\n\n- existing note\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not call generate_delta")),
    )

    seen = {}

    def _fake_generate(file_path, block_name, block_content, user_request, context_before="", context_after=""):
        seen["context_before"] = context_before
        seen["context_after"] = context_after
        return {
            "code": block_content + "\n- retrieval types: dense, sparse, hybrid",
            "input_tokens": 30, "output_tokens": 15, "total_tokens": 45, "cached_tokens": 0,
            "latency_ms": 1, "model": "test-model",
        }

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_block_replacement", _fake_generate)

    run_edit(
        project_dir=project_dir,
        file="readme.md",
        request="add retrieval types in the notes part",
        test_target=".",
        project_id="text-block-context-test",
        require_confirmation=False,
    )

    assert "feature one" in seen["context_before"]
    assert seen["context_after"] == ""  # Notes is the last block -- nothing after it


def test_run_edit_refuses_whole_file_when_no_block_locates(tmp_path, monkeypatch):
    """Policy: whole-file regeneration is disabled for edits to an
    EXISTING file, full stop -- a request that doesn't confidently match
    any one section (or a file with no real block structure at all)
    must be refused, never silently regenerate the whole file."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    original = "Just one plain paragraph, no headings at all.\n"
    (project_dir / "readme.md").write_text(original)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file_edit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must never generate a whole-file rewrite for an edit")),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="readme.md",
        request="add one more sentence",
        test_target=".",
        project_id="no-block-refusal-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "failed"
    assert metadata["result"]["failure_class"] == "WHOLE_FILE_BLOCKED"
    assert metadata["generation"]["total_tokens"] == 0  # no generation call was made at all
    assert (project_dir / "readme.md").read_text() == original  # untouched


def test_run_edit_jev_classification_dispatches_question_with_zero_structured_edit_calls(tmp_path, monkeypatch):
    """A confident Jev "question" classification must skip STRUCTURED_EDIT's
    own classification generation entirely and go straight to
    _run_question -- classify_request_kind mocked so this stays offline
    and network-free."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("def add(a, b):\n    return a + b\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr("incremental_editing.api.run_pipeline.classify_request_kind", lambda request: "question")
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not call generate_delta when Jev is confident")),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_answer",
        lambda file_path, source, question: {
            "answer": "add(a, b) returns a + b.",
            "input_tokens": 10, "output_tokens": 8, "total_tokens": 18, "cached_tokens": 0,
            "latency_ms": 1, "model": "test-model",
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="main.py",
        request="what does add do",
        test_target=".",
        project_id="jev-question-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "answered"
    assert metadata["strategy"] == "QUESTION_ANSWERING"
    assert metadata["generation"]["total_tokens"] == 18  # only the answer call, no classification call


def test_run_edit_jev_classification_dispatches_whole_file_and_it_refuses(tmp_path, monkeypatch):
    """A confident Jev "whole_file" classification still skips
    STRUCTURED_EDIT's own classification generation entirely (Jev's own
    call is the only cost paid) -- but whole-file regeneration for an
    edit is refused by policy, so no generation call follows it."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    original = "def add(a, b):\n    return a + b\n"
    (project_dir / "main.py").write_text(original)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr("incremental_editing.api.run_pipeline.classify_request_kind", lambda request: "whole_file")
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not call generate_delta when Jev is confident")),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file_edit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must never generate a whole-file rewrite for an edit")),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="main.py",
        request="clean up formatting across the whole file",
        test_target=".",
        project_id="jev-whole-file-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "failed"
    assert metadata["result"]["failure_class"] == "WHOLE_FILE_BLOCKED"
    assert metadata["generation"]["total_tokens"] == 0
    assert (project_dir / "main.py").read_text() == original


def test_run_edit_ignores_non_dispatchable_jev_kinds(tmp_path, monkeypatch):
    """A Jev classification of a kind that isn't safely dispatchable
    (needs extra structured fields Choice alone can't produce) must fall
    straight through to the existing pipeline unchanged, not raise or
    misdispatch."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.classify_request_kind", lambda request: "rename_identifier"
    )

    called = {}

    def _fake_generate_delta(**kwargs):
        called["hit"] = True
        return {
            "raw_json": '{"operations":[]}',
            "delta_dict": {"schema_version": "1.0", "base_version": kwargs["base_version"], "operations": []},
            "input_tokens": 50, "output_tokens": 5, "total_tokens": 55, "cached_tokens": 0,
            "latency_ms": 1, "model": "test-model",
        }

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", _fake_generate_delta)

    run_edit(
        project_dir=project_dir,
        file="main.py",
        request="do nothing in particular",
        test_target=".",
        project_id="jev-non-dispatchable-test",
        require_confirmation=False,
    )

    assert called.get("hit") is True  # fell through to the normal classification call


def test_run_edit_fast_path_renames_a_function_with_zero_llm_calls(tmp_path, monkeypatch):
    """Real waste this closes: "replace name of function of
    oauthAuthorizationCodeFlow to oauthAuthorizationFlow" localized (and
    sent as input) both the target function's *and* its caller's full
    bodies just to decide something the request's own wording already
    settled -- neither body was ever needed to know what to rename or
    what to call it. Must skip localization and the LLM call entirely,
    the same zero-token guarantee an unambiguous delete already gets."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "auth.py").write_text(
        "def oauthAuthorizationCodeFlow():\n"
        "    return 'token'\n\n\n"
        "def main():\n"
        "    return oauthAuthorizationCodeFlow()\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    def _fail_if_called(**kwargs):
        raise AssertionError("must not call the LLM for an unambiguous function rename")

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", _fail_if_called)

    metadata = run_edit(
        project_dir=project_dir,
        file="auth.py",
        request="replace name of function of oauthAuthorizationCodeFlow to oauthAuthorizationFlow",
        test_target=".",
        project_id="fast-rename-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["strategy"] == "MECHANICAL_RENAME"
    assert metadata["generation"]["total_tokens"] == 0
    new_source = (project_dir / "auth.py").read_text()
    assert "oauthAuthorizationCodeFlow" not in new_source
    assert new_source.count("oauthAuthorizationFlow") == 2  # definition + call site


def test_run_edit_use_joern_off_by_default_never_touches_joern(tmp_path, monkeypatch):
    """use_joern defaults False -- Joern must never even be checked for
    availability unless explicitly requested, same offline-safe
    guarantee use_hybrid_retrieval already holds itself to."""
    from incremental_editing.retrieval import joern_graph

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "auth.py").write_text(
        "def oauthAuthorizationCodeFlow():\n    return 'token'\n\n\ndef main():\n    return oauthAuthorizationCodeFlow()\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        joern_graph, "is_available", lambda: (_ for _ in ()).throw(AssertionError("must not check Joern"))
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="auth.py",
        request="replace name of function of oauthAuthorizationCodeFlow to oauthAuthorizationFlow",
        test_target=".",
        project_id="joern-off-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert "cross_file_impact_warning" not in metadata


def test_run_edit_use_joern_warns_about_a_real_external_caller(tmp_path, monkeypatch):
    """The one blind spot this closes: rename_with_subword_fallback only
    ever rewrites the file it's given -- a caller in a DIFFERENT file is
    invisible to it. Joern's call graph mocked (never a real ~12-45s+ JVM
    call) to return exactly that shape."""
    from incremental_editing.retrieval import joern_graph, repo_index

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "auth.py").write_text(
        "def oauthAuthorizationCodeFlow():\n    return 'token'\n\n\ndef main():\n    return oauthAuthorizationCodeFlow()\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(joern_graph, "is_available", lambda: True)
    monkeypatch.setattr(repo_index, "build_repo_index", lambda project_dir: [])
    monkeypatch.setattr(
        joern_graph,
        "build_call_graph_via_joern",
        lambda project_dir, symbols: {
            "calls": {},
            "called_by": {"oauthAuthorizationCodeFlow": ["auth.py::main", "other_service.py::login_handler"]},
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="auth.py",
        request="replace name of function of oauthAuthorizationCodeFlow to oauthAuthorizationFlow",
        test_target=".",
        project_id="joern-warn-test",
        require_confirmation=False,
        use_joern=True,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["cross_file_impact_warning"] == ["other_service.py::login_handler"]  # not auth.py's own caller


def test_run_edit_use_joern_no_warning_when_every_caller_is_in_the_same_file(tmp_path, monkeypatch):
    from incremental_editing.retrieval import joern_graph, repo_index

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "auth.py").write_text(
        "def oauthAuthorizationCodeFlow():\n    return 'token'\n\n\ndef main():\n    return oauthAuthorizationCodeFlow()\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(joern_graph, "is_available", lambda: True)
    monkeypatch.setattr(repo_index, "build_repo_index", lambda project_dir: [])
    monkeypatch.setattr(
        joern_graph,
        "build_call_graph_via_joern",
        lambda project_dir, symbols: {"calls": {}, "called_by": {"oauthAuthorizationCodeFlow": ["auth.py::main"]}},
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="auth.py",
        request="replace name of function of oauthAuthorizationCodeFlow to oauthAuthorizationFlow",
        test_target=".",
        project_id="joern-no-warn-test",
        require_confirmation=False,
        use_joern=True,
    )

    assert metadata["result"]["status"] == "success"
    assert "cross_file_impact_warning" not in metadata


def test_run_edit_use_joern_gracefully_skips_when_joern_not_installed(tmp_path, monkeypatch):
    from incremental_editing.retrieval import joern_graph

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "auth.py").write_text(
        "def oauthAuthorizationCodeFlow():\n    return 'token'\n\n\ndef main():\n    return oauthAuthorizationCodeFlow()\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(joern_graph, "is_available", lambda: False)

    metadata = run_edit(
        project_dir=project_dir,
        file="auth.py",
        request="replace name of function of oauthAuthorizationCodeFlow to oauthAuthorizationFlow",
        test_target=".",
        project_id="joern-not-installed-test",
        require_confirmation=False,
        use_joern=True,
    )

    assert metadata["result"]["status"] == "success"
    assert "cross_file_impact_warning" not in metadata


def test_run_edit_escalates_to_rename_identifier_and_renames_mechanically(tmp_path, monkeypatch):
    """Real waste this closes: "replace col by column" against
    sessions_col/messages_col -- both a bare module-level assignment
    (REPLACE has no target for a top-level statement at all) and used
    inside ensure_indexes()'s body -- used to escalate all the way to a
    full-file LLM regeneration for a change with a known-correct,
    deterministic answer. Must rename every occurrence of both names
    mechanically instead, at zero LLM cost beyond the classification
    call that recognized this."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "new.py").write_text(
        "sessions_col = {}\n"
        "messages_col = {}\n\n\n"
        "def ensure_indexes():\n"
        "    sessions_col['x'] = 1\n"
        "    messages_col['y'] = 2\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen(
            {
                "kind": "rename_identifier",
                "renames": [
                    {"old_name": "sessions_col", "new_name": "sessions_column"},
                    {"old_name": "messages_col", "new_name": "messages_column"},
                ],
            }
        ),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="new.py",
        request="replace col by column",
        test_target=".",
        project_id="rename-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["strategy"] == "MECHANICAL_RENAME"
    assert metadata["generation"]["total_tokens"] == 15  # only the escalate call's own cost, folded in
    new_source = (project_dir / "new.py").read_text()
    assert not re.search(r"\bsessions_col\b", new_source)  # not a substring check -- "sessions_column" contains it
    assert not re.search(r"\bmessages_col\b", new_source)
    assert new_source.count("sessions_column") == 2  # declaration + usage
    assert new_source.count("messages_column") == 2


def test_run_edit_rename_identifier_never_touches_a_name_that_merely_contains_it(tmp_path, monkeypatch):
    """Word-boundary matched, not a substring replace -- renaming "col"
    must never mangle "column" or "collection" if either already exists
    as its own real name."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "new.py").write_text("col = 1\ncollection = 2\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen(
            {"kind": "rename_identifier", "renames": [{"old_name": "col", "new_name": "column"}]}
        ),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="new.py",
        request="rename col to column",
        test_target=".",
        project_id="rename-boundary-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    new_source = (project_dir / "new.py").read_text()
    assert new_source == "column = 1\ncollection = 2\n"  # collection left untouched


def test_run_edit_rename_identifier_is_a_no_op_when_the_name_doesnt_exist(tmp_path, monkeypatch):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "new.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen(
            {"kind": "rename_identifier", "renames": [{"old_name": "does_not_exist", "new_name": "whatever"}]}
        ),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="new.py",
        request="rename does_not_exist to whatever",
        test_target=".",
        project_id="rename-noop-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["result"]["no_op"] is True
    assert (project_dir / "new.py").read_text() == "VALUE = 1\n"


def test_run_edit_rename_identifier_refuses_rather_than_whole_file_when_import_stays_broken(tmp_path, monkeypatch):
    """Real bug this catches: a mechanical rename that renamed a class
    reference (ChatOpenAI -> ChatAnthropic) but not the *import
    statement's own module path* it also appeared in produced
    `from langchain_openai import ChatAnthropic` -- syntactically
    perfect Python, ast.parse reported it passing, and a real
    ImportError the instant anything actually ran it. Policy: whole-file
    regeneration is disabled for edits, so when the narrow import-only
    fix also can't resolve it, the run is refused (never silently report
    success on broken code, and never fall back to a full rewrite)."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    original = "import json\n\nVALUE = json.dumps({'a': 1})\n"
    (project_dir / "app.py").write_text(original)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen(
            # An incomplete rename: only renames the bare word, leaving
            # a reference to a module that doesn't exist -- the exact
            # shape of the real bug (renamed the class, not the import).
            {"kind": "rename_identifier", "renames": [{"old_name": "json", "new_name": "nonexistent_module_xyz"}]}
        ),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file_edit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must never generate a whole-file rewrite for an edit")),
    )
    # The narrow import-only fix attempt (tried first, see
    # _locate_import_span) is mocked to NOT actually fix it -- proves
    # the run is refused rather than falling back to a full rewrite.
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_import_fix",
        lambda file_path, import_block, error, model=None: {
            "code": "import nonexistent_module_xyz", "input_tokens": 20, "output_tokens": 5,
            "total_tokens": 25, "cached_tokens": 0, "latency_ms": 1, "model": "test-model",
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="app.py",
        request="rename json import",
        test_target=".",
        project_id="rename-refusal-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "failed"
    assert metadata["result"]["failure_class"] == "WHOLE_FILE_BLOCKED"
    assert (project_dir / "app.py").read_text() == original  # untouched, not silently left broken


def test_run_edit_rename_identifier_fixes_broken_import_narrowly_without_whole_file_regen(tmp_path, monkeypatch):
    """Real waste this closes: previously ANY broken import after a
    mechanical rename escalated straight to a full-file rewrite -- every
    line of the file paid as both input and output to fix what's often
    just one broken import line. Must try a narrow, import-block-only
    fix first and use it when it actually resolves the import, without
    ever calling generate_full_file_edit at all.

    Mirrors the real motivating bug precisely: renaming a class
    (ChatOpenAI -> ChatAnthropic) consistently updates BOTH the import
    and every body call site to "ChatAnthropic" -- the inconsistency is
    that the import's own MODULE PATH still points at a real, importable
    module (real_provider.py) that simply doesn't define a symbol by
    that name (a real ImportError, not ModuleNotFoundError). The correct
    narrow fix is repointing the import at the module that actually
    defines it (other_provider.py) -- the body's own "ChatAnthropic(...)"
    call site needs no change at all, since it was already correct."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "real_provider.py").write_text("class ChatOpenAI:\n    pass\n")
    (project_dir / "other_provider.py").write_text("class ChatAnthropic:\n    pass\n")
    (project_dir / "app.py").write_text(
        "from real_provider import ChatOpenAI\n\ndef build():\n    return ChatOpenAI()\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen(
            {"kind": "rename_identifier", "renames": [{"old_name": "ChatOpenAI", "new_name": "ChatAnthropic"}]}
        ),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file_edit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not regenerate the whole file")),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_import_fix",
        lambda file_path, import_block, error, model=None: {
            # The real, correct fix: repoint the import at the module
            # that actually defines ChatAnthropic -- the imported name
            # itself was already right.
            "code": "from other_provider import ChatAnthropic", "input_tokens": 20, "output_tokens": 5,
            "total_tokens": 25, "cached_tokens": 0, "latency_ms": 1, "model": "test-model",
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="app.py",
        request="rename ChatOpenAI class to ChatAnthropic",
        test_target=".",
        project_id="rename-narrow-import-fix-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["strategy"] == "MECHANICAL_RENAME"  # never escalated to FULL_REGENERATION
    new_source = (project_dir / "app.py").read_text()
    assert "from other_provider import ChatAnthropic" in new_source
    assert "return ChatAnthropic()" in new_source  # body untouched by the import-only fix
    assert metadata["generation"]["total_tokens"] == 40  # escalate call (15) + narrow import fix (25), no whole file


def test_run_edit_refuses_a_whole_file_delete_that_breaks_the_test_suite(tmp_path, monkeypatch):
    """Same real-conflict check code-level DELETE already gets: other
    code still importing the file must block the delete outright rather
    than silently removing something still depended on."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "helper.py").write_text("def value():\n    return 1\n")
    (project_dir / "test_helper.py").write_text(
        "from helper import value\n\ndef test_value():\n    assert value() == 1\n"
    )

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    metadata = run_edit(
        project_dir=project_dir,
        file="helper.py",
        request="delete helper.py",
        test_target=".",
        project_id="file-delete-conflict-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "failed"
    assert metadata["result"]["failure_class"] == "TEST_FAILURE"
    assert (project_dir / "helper.py").exists()  # refused before writing/deleting anything
    assert VersionManager(storage, "file-delete-conflict-test").get_head() is None


def test_run_edit_whole_file_delete_awaits_confirmation_then_resolves(tmp_path, monkeypatch):
    """The web UI's human-in-the-loop gate must apply to a file delete
    exactly like every other change: nothing removed from disk until the
    human explicitly accepts via pending_confirmations.resolve."""
    from incremental_editing.api import pending_confirmations

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "scratch.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr("incremental_editing.api.pending_confirmations.get_storage", lambda: storage)

    metadata = run_edit(
        project_dir=project_dir,
        file="scratch.py",
        request="delete scratch.py",
        test_target=".",
        project_id="file-delete-confirm-test",
        require_confirmation=True,
    )

    assert metadata["result"]["status"] == "awaiting_confirmation"
    assert (project_dir / "scratch.py").exists()  # nothing written yet

    result = pending_confirmations.resolve(metadata["run_id"], accept=True)
    assert result["result"]["status"] == "success"
    assert not (project_dir / "scratch.py").exists()
    assert VersionManager(storage, "file-delete-confirm-test").get_head() == "v1"


def test_run_edit_resolves_duplicate_method_name_via_class_mention(tmp_path, monkeypatch):
    """End-to-end (real build_context/locate_candidates, only the LLM call
    itself mocked): a large, class-heavy file with the same method name
    duplicated across classes (Trig.tan and Hyperbolic.tan) is exactly the
    "10-15 similarly-named things in a 5000-line file" scenario -- a
    REPLACE naming the bare method plus its enclosing class must resolve
    to that specific occurrence, not re-trigger AmbiguousSymbolError at
    validate/apply time the way a bare-name-only lookup would."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    source = (
        "class Trig:\n"
        "    def tan(self, x):\n"
        "        import math\n"
        "        return math.tan(x)\n"
        "\n\n"
        "class Hyperbolic:\n"
        "    def tan(self, x):\n"
        "        import math\n"
        "        return math.tanh(x)\n"
    )
    (project_dir / "calculator.py").write_text(source)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    canned_delta = {
        "schema_version": "1.0",
        "base_version": "v0",
        "operations": [
            {
                "operation": "REPLACE",
                "target": {"file": "calculator.py", "symbol_type": "function", "symbol_name": "tan"},
                "content": (
                    "    def tan(self, x):\n"
                    "        if not isinstance(x, (int, float)):\n"
                    "            raise TypeError('Argument must be a number')\n"
                    "        import math\n"
                    "        return math.tanh(x)\n"
                ),
            }
        ],
    }
    canned_gen = {
        "raw_json": "{}",
        "delta_dict": canned_delta,
        "input_tokens": 10,
        "output_tokens": 10,
        "total_tokens": 20,
        "latency_ms": 1,
        "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)

    metadata = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="fix Hyperbolic's tan method to validate input",
        test_target=".",  # no test_*.py file in project_dir -- cleanly "no tests collected", not a failure
        project_id="dup-name-test",
    )

    assert metadata["result"]["status"] == "success"
    new_source = (project_dir / "calculator.py").read_text()
    trig_section, hyperbolic_section = new_source.split("class Hyperbolic")
    assert "isinstance" not in trig_section  # Trig's tan is untouched
    assert "isinstance" in hyperbolic_section  # Hyperbolic's tan got the new, validating body


def test_run_edit_sweeps_stale_references_after_a_replace_renames_a_symbol(tmp_path, monkeypatch):
    """Real bug this closes: a compound request ("change the temperature
    to 0.5, also rename ask_all to ask_everyone") produced a REPLACE that
    renamed only the def line -- `self.ask_all(...)` elsewhere in the
    same file, outside the REPLACE's own span, still called a name that
    no longer existed. Syntactically valid, semantically broken, and
    committed as a plain success because nothing checked whether the old
    name was still referenced anywhere else. A REPLACE only ever touches
    its own target's span, so any such reference is guaranteed to still
    read the old name -- must be swept to the new one mechanically, not
    left stale."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    source = (
        "class Bot:\n"
        "    def ask_model(self, x):\n"
        "        return self._call(x, temperature=0.7)\n\n"
        "    def ask_all(self, x):\n"
        "        return self.ask_model(x)\n\n"
        "    def chat(self, x):\n"
        "        return self.ask_all(x)\n"
    )
    (project_dir / "bot.py").write_text(source)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    canned_delta = {
        "schema_version": "1.0",
        "base_version": "v0",
        "operations": [
            {
                "operation": "REPLACE",
                "target": {"file": "bot.py", "symbol_type": "function", "symbol_name": "ask_all"},
                "content": ("    def ask_everyone(self, x):\n" "        return self.ask_model(x)\n"),
            }
        ],
    }
    canned_gen = {
        "raw_json": "{}",
        "delta_dict": canned_delta,
        "input_tokens": 10,
        "output_tokens": 10,
        "total_tokens": 20,
        "latency_ms": 1,
        "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)

    metadata = run_edit(
        project_dir=project_dir,
        file="bot.py",
        request="also rename ask_all to ask_everyone",
        test_target=".",
        project_id="rename-sweep-test",
    )

    assert metadata["result"]["status"] == "success"
    new_source = (project_dir / "bot.py").read_text()
    assert "ask_all" not in new_source  # no stale reference left anywhere
    assert "def ask_everyone" in new_source
    assert "self.ask_everyone(x)" in new_source  # chat()'s call site was swept too


def test_localize_step_message_does_not_claim_whole_file_when_it_wasnt_sent(tmp_path, monkeypatch):
    """Real report: a run logged "context: 8/51 lines (symbols=none, using
    whole file)" -- the numbers themselves prove only 8 of 51 lines were
    sent (build_context's compact imports+name-index fallback), but the
    message text claimed the whole file anyway. The wording was written
    before that compact fallback existed and never updated -- must say
    "using whole file" only when context_lines actually equals
    total_lines, not unconditionally whenever no symbol matched."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    source = "\n\n".join(f"def helper_{i}(x):\n    return x + {i}" for i in range(12)) + "\n"
    (project_dir / "calculator.py").write_text(source)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    canned_gen = {
        "raw_json": "{}",
        "delta_dict": {"schema_version": "1.0", "base_version": "v0", "operations": []},
        "input_tokens": 10,
        "output_tokens": 10,
        "total_tokens": 20,
        "latency_ms": 1,
        "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)

    steps = []
    run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="completely unrelated request naming nothing real",
        project_id="label-test",
        on_step=lambda tag, msg: steps.append((tag, msg)),
    )

    localize_msg = next(msg for tag, msg in steps if tag == "LOCALIZE")
    assert "using whole file" not in localize_msg, localize_msg
    assert "imports + name index" in localize_msg
    assert not any(tag == "RETRIEVE" for tag, _ in steps)  # hybrid off by default -- nothing to announce


def test_run_edit_announces_hybrid_retrieval_when_requested(tmp_path, monkeypatch):
    """Real gap this closes: build_context's BM25+vector fusion (see
    use_hybrid_retrieval) is a pure function with no on_step of its own
    -- it ran silently, so a human watching the step log had no visible
    confirmation it fired at all, even when it was genuinely on. Must
    announce it explicitly, the same wording `iee find`'s own hybrid
    retrieval step already uses."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "calculator.py").write_text("def helper(x):\n    return x + 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    canned_gen = {
        "raw_json": "{}",
        "delta_dict": {"schema_version": "1.0", "base_version": "v0", "operations": []},
        "input_tokens": 10, "output_tokens": 10, "total_tokens": 20, "latency_ms": 1, "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)
    # VectorRetriever mocked so this stays offline/network-free -- only
    # checking that the step fires, not what it finds.
    monkeypatch.setattr(
        "incremental_editing.context.context_builder.VectorRetriever",
        lambda symbols: type("V", (), {"rank": lambda self, q, top_k: []})(),
    )

    steps = []
    run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="completely unrelated request naming nothing real",
        project_id="hybrid-announce-test",
        use_hybrid_retrieval=True,
        on_step=lambda tag, msg: steps.append((tag, msg)),
    )

    retrieve_msg = next(msg for tag, msg in steps if tag == "RETRIEVE")
    assert "BM25" in retrieve_msg and "vector" in retrieve_msg


def test_run_edit_refuses_rather_than_whole_file_when_replace_targets_an_unshown_symbol(tmp_path, monkeypatch):
    """Real case: a request against a file named "new.db" (real Python,
    misdetected as data) localized to nothing, so the model only saw the
    compact fallback (imports + bare names, no bodies) -- yet still
    emitted a REPLACE for one of those bare names. validate_targets
    refuses that as UnseenReplaceTargetError (the content_shown_for
    guard). Policy: whole-file regeneration is disabled for edits, so
    this now refuses outright rather than escalating to a full rewrite
    -- never silently reconstruct content the model never actually saw."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "calculator.py").write_text(CALCULATOR_SOURCE)

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)

    # "completely unrelated request naming nothing real" localizes to no
    # candidate symbols (see test_localize_step_message_... above), so
    # content_shown_for is empty -- yet the canned delta still REPLACEs
    # a real symbol ("add") it was never shown.
    canned_gen = {
        "raw_json": "{}",
        "delta_dict": {
            "schema_version": "1.0",
            "base_version": "v0",
            "operations": [
                {
                    "operation": "REPLACE",
                    "target": {"file": "calculator.py", "symbol_type": "function", "symbol_name": "add"},
                    "content": "def add(a, b):\n    return a + b  # unseen rewrite\n",
                }
            ],
        },
        "input_tokens": 10,
        "output_tokens": 10,
        "total_tokens": 20,
        "latency_ms": 1,
        "model": "test-model",
    }
    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_delta", lambda **kwargs: canned_gen)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file_edit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must never generate a whole-file rewrite for an edit")),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="calculator.py",
        request="completely unrelated request naming nothing real",
        test_target=".",
        project_id="unseen-replace-refusal-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "failed"
    assert metadata["result"]["failure_class"] == "WHOLE_FILE_BLOCKED"
    assert (project_dir / "calculator.py").read_text() == CALCULATOR_SOURCE
    # the rejected structured-edit attempt's own tokens are still folded in
    # (honest accounting), even though no whole-file generation follows it
    assert metadata["generation"]["total_tokens"] == 20


def _canned_escalate_gen(escalate: dict) -> dict:
    return {
        "raw_json": "{}",
        "delta_dict": {"schema_version": "1.0", "base_version": "v0", "operations": [], "escalate": escalate},
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "latency_ms": 1,
        "model": "test-model",
    }


def test_run_edit_escalates_to_create_files_for_new_unrelated_files(tmp_path, monkeypatch):
    """Real failure: "make a new code file of frontend and backend" while
    editing chatbot.py got forced through the whole_file escalate instead
    -- the only "this isn't a symbol-level edit" escape hatch available at
    the time -- silently regenerating chatbot.py's own content (bumped its
    version, created nothing new) since a request for brand-new,
    unrelated files doesn't fit "transform this file" at all. Must create
    the named files and leave the currently open file completely alone."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "chatbot.py").write_text("def reply(msg):\n    return msg\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen({"kind": "create_files", "files": ["frontend/index.html", "backend/server.py"]}),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file",
        lambda file_path, user_request: {
            "code": f"# {file_path}\n", "input_tokens": 5, "output_tokens": 5,
            "total_tokens": 10, "cached_tokens": 0, "latency_ms": 1, "model": "test-model",
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="chatbot.py",
        request="make a new code file of frontend and backend",
        test_target=".",
        project_id="create-files-test",
        require_confirmation=False,  # CLI-style auto-commit, same as every other test here
    )

    assert metadata["result"]["status"] == "success"
    assert set(metadata["files"]) == {"frontend/index.html", "backend/server.py"}
    assert (project_dir / "frontend" / "index.html").exists()
    assert (project_dir / "backend" / "server.py").exists()
    # chatbot.py itself was never touched -- not created/edited, still its original content
    assert (project_dir / "chatbot.py").read_text() == "def reply(msg):\n    return msg\n"


def test_run_edit_create_files_refuses_to_overwrite_an_existing_file(tmp_path, monkeypatch):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "chatbot.py").write_text("def reply(msg):\n    return msg\n")
    (project_dir / "backend.py").write_text("# already here\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen({"kind": "create_files", "files": ["backend.py"]}),
    )

    with pytest.raises(FileExistsError):
        run_edit(
            project_dir=project_dir,
            file="chatbot.py",
            request="make a backend file",
            test_target=".",
            project_id="create-files-overwrite-test",
        )
    assert (project_dir / "backend.py").read_text() == "# already here\n"  # untouched


def test_run_edit_create_files_materializes_a_real_binary_artifact(tmp_path, monkeypatch):
    """Real failure this fixes: "make new excel file and add the data of
    Agentic AI" produced a file literally named "Agentic_AI.xlsx" whose
    actual bytes were a Python script (using openpyxl) that would
    create the real file *if executed* -- never executed, so opening it
    failed. The escalate path must run the generated script and save
    its real output instead of the script text. Uses .db/sqlite3
    (standard library) instead of .xlsx/openpyxl to keep this test
    dependency-free -- same code path either way."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen({"kind": "create_files", "files": ["store.db"]}),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file",
        lambda file_path, user_request: {
            "code": (
                "import sqlite3\n"
                "conn = sqlite3.connect('store.db')\n"
                "conn.execute('CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)')\n"
                "conn.commit()\n"
                "conn.close()\n"
            ),
            "input_tokens": 20, "output_tokens": 40, "total_tokens": 60, "cached_tokens": 0,
            "latency_ms": 5, "model": "test-model",
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="main.py",
        request="make a new sqlite database file called store.db with a products table",
        test_target=".",
        project_id="binary-artifact-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    real_bytes = (project_dir / "store.db").read_bytes()
    assert real_bytes.startswith(b"SQLite format 3\x00")  # a real SQLite file, not a Python script


def test_run_edit_create_files_makes_a_bare_folder_with_no_llm_call(tmp_path, monkeypatch):
    """Real failure this fixes: "make a folder called frontend" had no
    way to express "just a directory" in the create_files schema, so the
    model invented a file named "frontend" (no extension) containing a
    Python script that creates a folder *when run* -- syntactically
    valid, technically "a file was created," completely wrong. A path
    ending in "/" must be mkdir'd directly, with no LLM call spent
    generating content for something that has none."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen({"kind": "create_files", "files": ["frontend/"]}),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not call the LLM for a bare folder")),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="main.py",
        request="make a folder called frontend",
        test_target=".",
        project_id="folder-create-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["folders"] == ["frontend"]
    assert metadata["files"] == []
    # Only the classification call's own cost (folded in) -- nothing
    # beyond it, since generate_full_file must never be called for a
    # bare folder (the monkeypatch above would raise if it were).
    assert metadata["generation"]["total_tokens"] == 15
    assert (project_dir / "frontend").is_dir()
    assert not (project_dir / "frontend").is_file()


def test_run_edit_create_files_mixes_a_folder_and_a_real_file_in_one_request(tmp_path, monkeypatch):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen({"kind": "create_files", "files": ["frontend/", "backend/server.py"]}),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file",
        lambda file_path, user_request: {
            "code": f"# {file_path}\n", "input_tokens": 5, "output_tokens": 5,
            "total_tokens": 10, "cached_tokens": 0, "latency_ms": 1, "model": "test-model",
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="main.py",
        request="make a frontend folder and a backend/server.py file",
        test_target=".",
        project_id="folder-and-file-test",
        require_confirmation=False,
    )

    assert metadata["result"]["status"] == "success"
    assert metadata["folders"] == ["frontend"]
    assert metadata["files"] == ["backend/server.py"]
    assert (project_dir / "frontend").is_dir()
    assert (project_dir / "backend" / "server.py").read_text() == "# backend/server.py\n"


def test_run_edit_create_files_folder_awaits_confirmation_then_resolves(tmp_path, monkeypatch):
    from incremental_editing.api import pending_confirmations

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr("incremental_editing.api.pending_confirmations.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen({"kind": "create_files", "files": ["frontend/"]}),
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="main.py",
        request="make a folder called frontend",
        test_target=".",
        project_id="folder-confirm-test",
        require_confirmation=True,
    )
    assert metadata["result"]["status"] == "awaiting_confirmation"
    assert metadata["file"] is None  # nothing a confirm-bar tab could preview
    assert not (project_dir / "frontend").exists()  # nothing written yet

    result = pending_confirmations.resolve(metadata["run_id"], accept=True)
    assert result["result"]["status"] == "success"
    assert (project_dir / "frontend").is_dir()


def test_run_edit_create_files_link_step_sends_signatures_not_full_content(tmp_path, monkeypatch):
    """Real waste this closes: linking a substantial new Python file
    (not a plain KEY=VALUE .env) used to re-send that file's *complete*
    text just to compose one import/usage edit -- dominating input
    tokens for a real multi-file "chatbot + storage + sessions" request.
    The link step's prompt must carry a compact signature instead of
    the function body."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("VALUE = 1\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen(
            {"kind": "create_files", "files": ["backend/app.py"], "also_link_current_file": True}
        ),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file",
        lambda file_path, user_request: {
            "code": (
                "def handle_chat(message, history):\n"
                "    # a long real body that must never be re-sent in full for linking\n"
                "    return {'reply': message}\n"
            ),
            "input_tokens": 20, "output_tokens": 40, "total_tokens": 60, "cached_tokens": 0,
            "latency_ms": 1, "model": "test-model",
        },
    )
    captured = {}

    def _fake_link(file_path, original_source, user_request):
        captured["user_request"] = user_request
        return {"code": "VALUE = 1\n", "input_tokens": 5, "output_tokens": 5, "total_tokens": 10,
                "cached_tokens": 0, "latency_ms": 1, "model": "test-model"}

    monkeypatch.setattr("incremental_editing.api.run_pipeline.generate_full_file_edit", _fake_link)

    run_edit(
        project_dir=project_dir,
        file="main.py",
        request="make a chatbot backend",
        test_target=".",
        project_id="link-summary-test",
        require_confirmation=False,
    )

    assert "handle_chat(message, history)" in captured["user_request"]
    assert "a long real body that must never be re-sent in full for linking" not in captured["user_request"]


def test_run_edit_create_files_also_links_current_file_atomically(tmp_path, monkeypatch):
    """"Make a .env file that links to this chatbot" -- the new file and
    the resulting edit to the currently open file must land as ONE
    accept/reject, not two separate surprises: a human reviewing "created
    .env" would also want to see the matching chatbot.py change in the
    same review."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "chatbot.py").write_text("API_KEY = 'hardcoded'\n")

    storage = LocalStorage(root_dir=str(tmp_path / "minio_local_data"))
    monkeypatch.setattr("incremental_editing.api.run_pipeline.get_storage", lambda: storage)
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_delta",
        lambda **kwargs: _canned_escalate_gen(
            {"kind": "create_files", "files": [".env"], "also_link_current_file": True}
        ),
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file",
        lambda file_path, user_request: {
            "code": "API_KEY=secret\n", "input_tokens": 5, "output_tokens": 5,
            "total_tokens": 10, "cached_tokens": 0, "latency_ms": 1, "model": "test-model",
        },
    )
    monkeypatch.setattr(
        "incremental_editing.api.run_pipeline.generate_full_file_edit",
        lambda file_path, original_source, user_request: {
            "code": "import os\nAPI_KEY = os.getenv('API_KEY')\n", "input_tokens": 8, "output_tokens": 8,
            "total_tokens": 16, "cached_tokens": 0, "latency_ms": 1, "model": "test-model",
        },
    )

    metadata = run_edit(
        project_dir=project_dir,
        file="chatbot.py",
        request="make a .env file that links to this chatbot",
        test_target=".",
        project_id="create-files-link-test",
        require_confirmation=True,
    )

    assert metadata["result"]["status"] == "awaiting_confirmation"
    assert set(metadata["files"]) == {".env", "chatbot.py"}
    # nothing written yet -- both files still their pre-run state
    assert not (project_dir / ".env").exists()
    assert (project_dir / "chatbot.py").read_text() == "API_KEY = 'hardcoded'\n"
