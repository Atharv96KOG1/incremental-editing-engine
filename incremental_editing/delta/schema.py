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
    "properties": {
        "schema_version": {"type": "string"},
        "base_version": {"type": "string"},
        "escalate": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {"enum": ["whole_file", "language_conversion", "question", "create_files", "delete_file", "rename_identifier"]},
                "target_language": {"type": "string"},
                "target_extension": {"type": "string"},
                "renames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["old_name", "new_name"],
                        "properties": {"old_name": {"type": "string"}, "new_name": {"type": "string"}},
                    },
                },
                "files": {"type": "array", "items": {"type": "string"}},
                "also_link_current_file": {"type": "boolean"},
            },
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["operation", "target"],
                "properties": {
                    "operation": {"enum": list(VALID_OPERATIONS)},
                    "target": {
                        "type": "object",
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
    anchor: Optional[str] = None

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
