"""Type annotations as a structure instead of a string.

The parser keeps an annotation as the text the programmer wrote (`array[int]`, `map[str, int]`,
`int?`, `fn(int) -> str`, `Vec*`). This module turns that text into a small tree so the semantic
pass can validate names and compare element types, without each check re-parsing strings.

Grammar (spaces are ignored; `const ` prefixes are dropped):

    type     := base { "[]" | "?" | "*" }
    base     := "fn" "(" [type {"," type}] ")" "->" type
              | NAME [ "[" type {"," type} "]" ]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TypeNode:
    kind: str  # "name", "optional", "pointer", "fn"
    name: str = ""
    args: tuple["TypeNode", ...] = ()

    def __str__(self) -> str:
        if self.kind == "optional":
            return f"{self.args[0]}?"
        if self.kind == "pointer":
            return f"{self.args[0]}*"
        if self.kind == "fn":
            return f"fn({','.join(str(a) for a in self.args[:-1])})->{self.args[-1]}"
        if self.args:
            return f"{self.name}[{','.join(str(a) for a in self.args)}]"
        return self.name


def parse_type(text: str) -> TypeNode:
    """Parse an annotation. Raises ValueError when it is malformed."""
    compact = text.replace("const ", "").replace(" ", "")
    if not compact:
        raise ValueError("empty type")
    node, rest = _parse(compact, 0)
    if rest != len(compact):
        raise ValueError(f"unexpected '{compact[rest:]}' in type '{text}'")
    return node


def _parse(s: str, i: int) -> tuple[TypeNode, int]:
    if s.startswith("fn(", i):
        i += 3
        params: list[TypeNode] = []
        if s[i : i + 1] != ")":
            while True:
                param, i = _parse(s, i)
                params.append(param)
                if s[i : i + 1] == ",":
                    i += 1
                    continue
                break
        if s[i : i + 3] != ")->":
            raise ValueError("a function type needs '->' and a result type")
        result, i = _parse(s, i + 3)
        node = TypeNode("fn", args=(*params, result))
    else:
        j = i
        while j < len(s) and (s[j].isalnum() or s[j] in "_."):
            j += 1
        if j == i:
            raise ValueError(f"expected a type name at '{s[i:]}'")
        name = s[i:j]
        i = j
        args: list[TypeNode] = []
        if s[i : i + 1] == "[" and s[i : i + 2] != "[]":
            i += 1
            while True:
                arg, i = _parse(s, i)
                args.append(arg)
                if s[i : i + 1] == ",":
                    i += 1
                    continue
                break
            if s[i : i + 1] != "]":
                raise ValueError("missing ']' in type")
            i += 1
        node = TypeNode("name", name, tuple(args))
    while i < len(s):
        if s.startswith("[]", i):
            node, i = TypeNode("name", "array", (node,)), i + 2
        elif s[i] == "?":
            node, i = TypeNode("optional", args=(node,)), i + 1
        elif s[i] == "*":
            node, i = TypeNode("pointer", args=(node,)), i + 1
        else:
            break
    return node, i


def names_in(node: TypeNode) -> list[str]:
    """Every type name mentioned in the tree (`map[str, Vec]` gives map, str, Vec)."""
    if node.kind == "name":
        found = [node.name]
    else:
        found = []
    for arg in node.args:
        found.extend(names_in(arg))
    return found
