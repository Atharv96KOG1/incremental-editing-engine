"""Offline tests for the slimmer Delta IR request shape -- the model is no
longer asked for schema_version, base_version, or each operation's
target.file, since all three are already known before the call is ever
made and nothing downstream reads the model's own copies back out
(verified: pure wasted output tokens, removed for real, measured savings).
"""

import pytest

from incremental_editing.delta.schema import DELTA_SCHEMA_VERSION, DeltaIR
from incremental_editing.delta.validator import validate_schema
from incremental_editing.strategies.structured_edit import SYSTEM_PROMPT, _fill_known_fields


def test_system_prompt_no_longer_requests_the_backfilled_fields():
    """The instruction text itself must not ask for what's being
    backfilled -- if it still asked, the model would still spend output
    tokens producing them regardless of what the schema now accepts."""
    assert '"schema_version"' not in SYSTEM_PROMPT
    assert '"base_version"' not in SYSTEM_PROMPT
    assert '"file":<str>' not in SYSTEM_PROMPT
    # the actual editing rules must still be intact
    assert "never invent" in SYSTEM_PROMPT.lower()
    assert "redefin" in SYSTEM_PROMPT.lower()  # rule intact regardless of exact phrasing


def test_fill_known_fields_backfills_a_minimal_model_response():
    """Exactly the shape verified against the real model: an operation
    with only operation/target.symbol_type/target.symbol_name/content --
    no schema_version, base_version, or target.file at all."""
    delta_dict = {
        "operations": [
            {
                "operation": "INSERT",
                "target": {"symbol_type": "function", "symbol_name": "cube_root", "anchor": "cube"},
                "content": "def cube_root(a):\n    return a ** (1/3)\n",
            }
        ]
    }
    _fill_known_fields(delta_dict, file_path="calculator.py", base_version="v57")

    assert delta_dict["schema_version"] == DELTA_SCHEMA_VERSION
    assert delta_dict["base_version"] == "v57"
    assert delta_dict["operations"][0]["target"]["file"] == "calculator.py"

    # and the result must parse and validate cleanly through the real pipeline
    validate_schema(delta_dict)
    delta = DeltaIR.from_dict(delta_dict)
    assert delta.operations[0].target.file == "calculator.py"
    assert delta.base_version == "v57"


def test_fill_known_fields_never_overwrites_a_value_already_present():
    """setdefault, not assignment -- if a caller (or a future model
    version) ever does include these fields, its own values win rather
    than being silently replaced."""
    delta_dict = {
        "schema_version": "9.9",
        "base_version": "v999",
        "operations": [{"operation": "DELETE", "target": {"symbol_type": "function", "symbol_name": "x", "file": "other.py"}}],
    }
    _fill_known_fields(delta_dict, file_path="calculator.py", base_version="v57")
    assert delta_dict["schema_version"] == "9.9"
    assert delta_dict["base_version"] == "v999"
    assert delta_dict["operations"][0]["target"]["file"] == "other.py"


def test_validate_schema_accepts_a_delta_missing_the_backfilled_fields():
    """The JSON schema itself must not require what the model is no
    longer asked to produce -- required=[] on those three would defeat
    the whole point of the prompt change if validation still rejected a
    genuinely minimal response before _fill_known_fields ever ran."""
    minimal = {
        "operations": [
            {"operation": "REPLACE", "target": {"symbol_type": "function", "symbol_name": "f"}, "content": "def f(): pass"}
        ]
    }
    validate_schema(minimal)  # must not raise


def test_deltair_from_dict_still_requires_file_after_backfill():
    """Backfill happens before parsing, not instead of it -- from_dict
    itself is unchanged and still expects target.file to be present by
    the time it runs. A delta that skipped backfill (a real bug, not
    normal usage) should fail loudly here, not silently default."""
    with pytest.raises(KeyError):
        DeltaIR.from_dict(
            {
                "schema_version": "1.0",
                "base_version": "v0",
                "operations": [{"operation": "DELETE", "target": {"symbol_type": "function", "symbol_name": "f"}}],
            }
        )
