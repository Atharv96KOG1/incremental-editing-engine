"""Delta IR: normalized internal edit representation.

Any generation format (unified diff, structured edit, tool output) should
converge on this representation before the apply engine touches a file.
"""

from dataclasses import dataclass
from typing import Optional, List

DELTA_SCHEMA_VERSION = "1.0"

VALID_OPERATIONS = ("INSERT", "DELETE", "REPLACE")
VALID_SYMBOL_TYPES = ("function", "class")

DELTA_JSON_SCHEMA = {
    "type": "object",
    "required": ["operations"],
    # schema_version and base_version are optional here on purpose: the
    # model is no longer asked to output either (see structured_edit.py's
    # _fill_known_fields) -- both are already known before the call
    # (schema_version is a fixed constant; base_version is a parameter
    # generate_delta/generate_repair already have) and grep confirms zero
    # code anywhere reads the *parsed* delta's copies of them back out.
    # Asking a model to restate values we already have and never
    # re-check was pure wasted output tokens on every single call.
    "properties": {
        "schema_version": {"type": "string"},
        "base_version": {"type": "string"},
        # Set instead of (alongside empty) operations when the model itself
        # recognizes the request can't be expressed as REPLACE/INSERT/DELETE
        # on named symbols at all -- see structured_edit.py's prompt for
        # exactly when. Read by run_pipeline.py before any of the fields
        # below are relied on; deliberately not language-list-validated
        # here (the model supplies target_language/target_extension
        # itself -- no fixed catalog to keep in sync with "every language").
        "escalate": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {"enum": ["whole_file", "language_conversion", "question", "create_files", "delete_file", "rename_identifier"]},
                "target_language": {"type": "string"},
                "target_extension": {"type": "string"},
                # rename_identifier only: rename every occurrence of each
                # listed old_name to its new_name in this file -- a
                # mechanical, zero-LLM-cost word-boundary text
                # substitution per pair, not another model call. Exists
                # because renaming something that isn't cleanly one
                # function/class's own name (a module-level variable, or
                # a name used across several symbols) has no REPLACE
                # target at all in this schema -- escalating all the way
                # to whole_file for what's really a one-shot mechanical
                # rename wastes a full regeneration on a change with a
                # known-correct, deterministic answer. A list, not a
                # single pair: "replace col by column" against
                # sessions_col/messages_col means renaming BOTH
                # consistently, one request, one review -- not a
                # separate escalate round-trip per identifier.
                "renames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["old_name", "new_name"],
                        "properties": {"old_name": {"type": "string"}, "new_name": {"type": "string"}},
                    },
                },
                # create_files only: relative paths (project_dir-relative,
                # e.g. "frontend/index.html") to create -- not edits to the
                # currently open file at all. The model picks how many and
                # what to name them; no fixed catalog or count here. A path
                # ending in "/" (e.g. "frontend/") is a bare folder with no
                # content -- run_pipeline._run_create_files mkdir's it
                # directly rather than generating a placeholder file for it.
                "files": {"type": "array", "items": {"type": "string"}},
                # create_files only: true when the currently open file
                # should ALSO be updated to use what's being created (e.g.
                # "make a .env file that links to this chatbot" -- create
                # .env first, then load it here). Omitted/false means the
                # new file(s) stand alone.
                "also_link_current_file": {"type": "boolean"},
            },
        },
        "operations": {
            "type": "array",
            # empty is valid: the model may correctly determine the request is
            # already satisfied (or its target doesn't exist to act on) --
            # that's a no-op success, not a malformed delta.
            "items": {
                "type": "object",
                "required": ["operation", "target"],
                "properties": {
                    "operation": {"enum": list(VALID_OPERATIONS)},
                    "target": {
                        "type": "object",
                        # target.file is optional for the same reason: the
                        # caller always already knows which file it's
                        # editing (that's how it read `original_source` in
                        # the first place) -- filled in the same way as
                        # schema_version/base_version above, never actually
                        # read back out of the parsed delta by anything.
                        "required": ["symbol_type", "symbol_name"],
                        "properties": {
                            "file": {"type": "string"},
                            "symbol_type": {"enum": list(VALID_SYMBOL_TYPES)},
                            "symbol_name": {"type": "string"},
                            "anchor": {"type": "string"},
                        },
                    },
                    "content": {"type": "string"},
                },
            },
        },
    },
}


def _bare_name(name: Optional[str]) -> Optional[str]:
    """Strips a call-signature suffix a model sometimes echoes verbatim
    from the "other symbols" context line (context_builder.py renders a
    zero-arg function as e.g. "load_api_key()", and a real generation has
    been observed copying that whole string as symbol_name instead of
    just "load_api_key", failing REPLACE/DELETE lookup and then repeating
    the identical mistake across every repair attempt since nothing told
    it what was actually wrong). No identifier in any language this
    project targets can contain "(", so truncating at the first one is
    always safe and a no-op for an already-bare name."""
    if not name:
        return name
    return name.split("(", 1)[0].strip()


@dataclass
class Target:
    file: str
    symbol_type: str
    symbol_name: str
    anchor: Optional[str] = None  # INSERT only: symbol to insert after; None = end of file

    def to_dict(self) -> dict:
        d = {"file": self.file, "symbol_type": self.symbol_type, "symbol_name": self.symbol_name}
        if self.anchor:
            d["anchor"] = self.anchor
        return d


@dataclass
class Operation:
    operation: str
    target: Target
    content: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "Operation":
        t = d["target"]
        target = Target(
            file=t["file"],
            symbol_type=t["symbol_type"],
            symbol_name=_bare_name(t["symbol_name"]),
            anchor=_bare_name(t.get("anchor")),
        )
        return Operation(operation=d["operation"], target=target, content=d.get("content"))

    def to_dict(self) -> dict:
        return {"operation": self.operation, "target": self.target.to_dict(), "content": self.content}


@dataclass
class DeltaIR:
    schema_version: str
    base_version: str
    operations: List[Operation]

    @staticmethod
    def from_dict(d: dict) -> "DeltaIR":
        ops = [Operation.from_dict(o) for o in d["operations"]]
        return DeltaIR(schema_version=d["schema_version"], base_version=d["base_version"], operations=ops)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "base_version": self.base_version,
            "operations": [op.to_dict() for op in self.operations],
        }
