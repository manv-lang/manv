from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HIRStatement:
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class HIRFunction:
    name: str
    params: list[dict[str, Any]]
    return_type: str | None
    body: list[HIRStatement]


@dataclass
class HIRModule:
    version: str
    source: str
    functions: list[HIRFunction]
    top_level: list[HIRStatement]
    stubs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "functions": [
                {
                    "name": fn.name,
                    "params": fn.params,
                    "return_type": fn.return_type,
                    "body": [{"kind": st.kind, "attrs": st.attrs} for st in fn.body],
                }
                for fn in self.functions
            ],
            "top_level": [{"kind": st.kind, "attrs": st.attrs} for st in self.top_level],
            "stubs": self.stubs,
        }
