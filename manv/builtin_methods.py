"""Methods of the built-in string, array and map types: `s.upper()`, `xs.append(1)`, `m.keys()`.

Why this exists:
- With no standard library, the builtins are the library (TODO.md Phase 4). Methods keep the
  call syntax uniform with user-defined types instead of a growing pile of global helper names.
- Kept apart from the interpreter so the whole surface is readable in one place and can be
  re-implemented in the C engine against the same table.

Semantics decided here (Decisions 8 and 9 in TODO.md, still open to change):
- Strings are sequences of Unicode code points and are immutable.
- Arrays and maps are reference values: assignment and calls share them. Mutating methods
  change the receiver and return `none`; `copy()` makes an independent shallow copy.
- Maps keep insertion order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .interpreter import Interpreter


class BuiltinMethods:
    def __init__(self, interpreter: "Interpreter"):
        self._interp = interpreter

    # name -> (min args, max args, implementation(recv, args, span))
    def _table(self, base: Any) -> tuple[str, dict[str, tuple[int, int, Callable[..., Any]]]] | None:
        if isinstance(base, str):
            return "str", self._STR
        if isinstance(base, list):
            return "array", self._ARRAY
        if isinstance(base, dict):
            return "map", self._MAP
        return None

    def lookup(self, base: Any, attr: str, span: Any) -> Callable[..., Any] | None:
        """Return a callable bound to `base` for the method `attr`, or None if there is none."""
        found = self._table(base)
        if found is None or attr not in found[1]:
            return None
        kind, table = found
        low, high, impl = table[attr]

        def bound(*args: Any) -> Any:
            if not low <= len(args) <= high:
                expected = str(low) if low == high else f"{low} to {high}"
                self._fail("TypeError", f"{kind}.{attr}() takes {expected} argument(s), got {len(args)}", span)
            return impl(self, base, args, span)

        return bound

    def _fail(self, error: str, message: str, span: Any) -> None:
        self._interp._raise_runtime(error, message, span)

    def _require_str(self, value: Any, what: str, span: Any) -> str:
        if not isinstance(value, str):
            self._fail("TypeError", f"{what} must be a string", span)
        return value

    def _require_int(self, value: Any, what: str, span: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self._fail("TypeError", f"{what} must be an int", span)
        return value

    def _index_of(self, items: list[Any], target: Any, span: Any) -> int:
        for i, item in enumerate(items):
            if self._interp._equals(item, target, span):
                return i
        return -1

    # ------------------------------------------------------------------------- strings
    @staticmethod
    def _s_split(self: "BuiltinMethods", s: str, args: tuple, span: Any) -> list[str]:
        if not args:
            return s.split()
        sep = self._require_str(args[0], "separator", span)
        if sep == "":
            self._fail("ValueError", "empty separator", span)
        return s.split(sep)

    @staticmethod
    def _s_join(self: "BuiltinMethods", s: str, args: tuple, span: Any) -> str:
        items = args[0]
        if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
            self._fail("TypeError", "join() needs an array of strings", span)
        return s.join(items)

    @staticmethod
    def _s_replace(self: "BuiltinMethods", s: str, args: tuple, span: Any) -> str:
        old = self._require_str(args[0], "old", span)
        new = self._require_str(args[1], "new", span)
        if old == "":
            self._fail("ValueError", "cannot replace an empty string", span)
        return s.replace(old, new)

    @staticmethod
    def _s_text(method: str) -> Callable[..., Any]:
        def impl(self: "BuiltinMethods", s: str, args: tuple, span: Any) -> Any:
            if method in {"find", "startswith", "endswith", "contains"}:
                arg = self._require_str(args[0], "argument", span)
                return {
                    "find": lambda: s.find(arg),
                    "startswith": lambda: s.startswith(arg),
                    "endswith": lambda: s.endswith(arg),
                    "contains": lambda: arg in s,
                }[method]()
            if method in {"strip", "lstrip", "rstrip"} and args:
                chars = self._require_str(args[0], "characters", span)
                return getattr(s, method)(chars)
            return getattr(s, method)()

        return impl

    # ------------------------------------------------------------------------- arrays
    @staticmethod
    def _a_append(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> None:
        xs.append(args[0])

    @staticmethod
    def _a_pop(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> Any:
        if not xs:
            self._fail("IndexError", "pop from an empty array", span)
        index = self._require_int(args[0], "index", span) if args else -1
        if not -len(xs) <= index < len(xs):
            self._fail("IndexError", f"array index out of range: {index}", span)
        return xs.pop(index)

    @staticmethod
    def _a_insert(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> None:
        xs.insert(self._require_int(args[0], "index", span), args[1])

    @staticmethod
    def _a_extend(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> None:
        if not isinstance(args[0], list):
            self._fail("TypeError", "extend() needs an array", span)
        xs.extend(list(args[0]))

    @staticmethod
    def _a_remove(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> None:
        i = self._index_of(xs, args[0], span)
        if i < 0:
            self._fail("ValueError", "value not found in array", span)
        del xs[i]

    @staticmethod
    def _a_index(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> int:
        i = self._index_of(xs, args[0], span)
        if i < 0:
            self._fail("ValueError", "value not found in array", span)
        return i

    @staticmethod
    def _a_contains(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> bool:
        return self._index_of(xs, args[0], span) >= 0

    @staticmethod
    def _a_count(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> int:
        return sum(1 for x in xs if self._interp._equals(x, args[0], span))

    @staticmethod
    def _a_reverse(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> None:
        xs.reverse()

    @staticmethod
    def _a_sort(self: "BuiltinMethods", xs: list, args: tuple, span: Any) -> None:
        try:
            xs.sort()
        except TypeError:
            self._fail("TypeError", "sort() needs elements that can be compared with each other", span)

    @staticmethod
    def _copy(self: "BuiltinMethods", value: Any, args: tuple, span: Any) -> Any:
        return value.copy()

    @staticmethod
    def _clear(self: "BuiltinMethods", value: Any, args: tuple, span: Any) -> None:
        value.clear()

    # ------------------------------------------------------------------------- maps
    @staticmethod
    def _m_keys(self: "BuiltinMethods", m: dict, args: tuple, span: Any) -> list:
        return list(m.keys())

    @staticmethod
    def _m_values(self: "BuiltinMethods", m: dict, args: tuple, span: Any) -> list:
        return list(m.values())

    @staticmethod
    def _m_items(self: "BuiltinMethods", m: dict, args: tuple, span: Any) -> list:
        return [(k, v) for k, v in m.items()]

    @staticmethod
    def _m_get(self: "BuiltinMethods", m: dict, args: tuple, span: Any) -> Any:
        self._interp._assert_hashable(args[0], span)
        return m.get(args[0], args[1] if len(args) > 1 else None)

    @staticmethod
    def _m_has(self: "BuiltinMethods", m: dict, args: tuple, span: Any) -> bool:
        self._interp._assert_hashable(args[0], span)
        return args[0] in m

    @staticmethod
    def _m_remove(self: "BuiltinMethods", m: dict, args: tuple, span: Any) -> None:
        self._interp._assert_hashable(args[0], span)
        if args[0] not in m:
            self._fail("KeyError", f"missing key: {args[0]}", span)
        del m[args[0]]


_S = BuiltinMethods._s_text

BuiltinMethods._STR = {
    "upper": (0, 0, _S("upper")),
    "lower": (0, 0, _S("lower")),
    "strip": (0, 1, _S("strip")),
    "lstrip": (0, 1, _S("lstrip")),
    "rstrip": (0, 1, _S("rstrip")),
    "split": (0, 1, BuiltinMethods._s_split),
    "join": (1, 1, BuiltinMethods._s_join),
    "replace": (2, 2, BuiltinMethods._s_replace),
    "find": (1, 1, _S("find")),
    "startswith": (1, 1, _S("startswith")),
    "endswith": (1, 1, _S("endswith")),
    "contains": (1, 1, _S("contains")),
}

BuiltinMethods._ARRAY = {
    "append": (1, 1, BuiltinMethods._a_append),
    "pop": (0, 1, BuiltinMethods._a_pop),
    "insert": (2, 2, BuiltinMethods._a_insert),
    "extend": (1, 1, BuiltinMethods._a_extend),
    "remove": (1, 1, BuiltinMethods._a_remove),
    "index": (1, 1, BuiltinMethods._a_index),
    "contains": (1, 1, BuiltinMethods._a_contains),
    "count": (1, 1, BuiltinMethods._a_count),
    "reverse": (0, 0, BuiltinMethods._a_reverse),
    "sort": (0, 0, BuiltinMethods._a_sort),
    "copy": (0, 0, BuiltinMethods._copy),
    "clear": (0, 0, BuiltinMethods._clear),
}

BuiltinMethods._MAP = {
    "keys": (0, 0, BuiltinMethods._m_keys),
    "values": (0, 0, BuiltinMethods._m_values),
    "items": (0, 0, BuiltinMethods._m_items),
    "get": (1, 2, BuiltinMethods._m_get),
    "has": (1, 1, BuiltinMethods._m_has),
    "remove": (1, 1, BuiltinMethods._m_remove),
    "copy": (0, 0, BuiltinMethods._copy),
    "clear": (0, 0, BuiltinMethods._clear),
}
