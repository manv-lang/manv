"""Raw memory for ManV's C-style pointers: allocation, typed loads and stores, and checking.

Why this exists:
- `docs/pointers.md` (decisions P-1, P-2): pointers address *real* memory so `extern "c"` keeps
  working, and every access is *checked* against a table of live blocks so a mistake raises a
  catchable `PointerError` instead of corrupting memory.
- It is the single definition of layout (sizes, alignment, struct offsets) that the interpreter,
  and later the C engine and the LLVM backend, must agree with.

Rules:
- Memory handed out by `alloc` is zero-filled (C leaves it undefined; zero is a legal choice and
  keeps programs deterministic).
- A pointer returned by foreign C code is `foreign`: it is not in the table, so accesses through
  it are not checked (and can crash). Everything else must point into a live block.
"""

from __future__ import annotations

import bisect
import ctypes
import struct
import weakref
from dataclasses import dataclass, field
from typing import Any

# name -> (size in bytes, struct format). `int` is 64-bit (TODO.md decision 5, recommended default).
SCALARS: dict[str, tuple[int, str]] = {
    "i8": (1, "b"), "u8": (1, "B"),
    "i16": (2, "h"), "u16": (2, "H"),
    "i32": (4, "i"), "u32": (4, "I"),
    "i64": (8, "q"), "u64": (8, "Q"),
    "int": (8, "q"), "usize": (8, "Q"),
    "f32": (4, "f"), "f64": (8, "d"), "float": (8, "d"),
    "bool": (1, "?"),
}
INT_TYPES = {"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "int", "usize"}
FLOAT_TYPES = {"f32", "f64", "float"}
POINTER_SIZE = 8
MAX_TOMBSTONES = 4096


class MemoryFault(Exception):
    """A memory error. `kind` is "null" (NullPointerError), "pointer" (PointerError) or "type" (TypeError)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def is_pointer_type(type_name: str | None) -> bool:
    return bool(type_name) and type_name.rstrip().endswith("*")


def pointee_of(type_name: str) -> str:
    """`int*` -> `int`, `const u8*` -> `u8`, `int**` -> `int*`."""
    text = type_name.strip()[:-1].strip()
    if text.startswith("const "):
        text = text[len("const "):].strip()
    return text or "void"


def canonical_pointer_type(type_name: str) -> str:
    """Spell `ptr` as `void*` and drop spaces before `*`; keeps `const`."""
    text = type_name.strip()
    if text == "ptr":
        return "void*"
    stars = 0
    while text.endswith("*"):
        stars += 1
        text = text[:-1].strip()
    return ("void" if text == "ptr" else text) + "*" * stars


@dataclass(frozen=True)
class Pointer:
    """A typed address. Equal when the addresses are equal; the pointee type decides the scale."""

    address: int
    pointee: str = "void"
    foreign: bool = field(default=False, compare=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Pointer) and other.address == self.address

    def __hash__(self) -> int:
        return hash(self.address)

    def __bool__(self) -> bool:
        return self.address != 0

    def retyped(self, pointee: str) -> "Pointer":
        return Pointer(self.address, pointee, self.foreign)


NULL = Pointer(0, "void")


@dataclass
class Field:
    name: str
    type_name: str
    count: int | None  # element count for a C-array field
    offset: int = 0


@dataclass
class StructDef:
    name: str
    fields: list[Field]
    size: int = 0
    align: int = 1

    def field(self, name: str) -> Field | None:
        return next((f for f in self.fields if f.name == name), None)


@dataclass
class _Block:
    base: int
    size: int
    buffer: Any  # keeps the ctypes buffer alive; None once freed
    live: bool = True
    heap: bool = False  # from malloc: the only blocks free() accepts


class StructRef:
    """A struct value: the address of its bytes. Owned values free their block when collected."""

    def __init__(self, memory: "Memory", address: int, sdef: StructDef, owned: bool):
        self.memory = memory
        self.address = address
        self.sdef = sdef
        if owned:
            weakref.finalize(self, memory.release, address)


class ArrayRef:
    """A C array (`int buf[16]`): contiguous elements with a known count. Decays to a pointer."""

    def __init__(self, memory: "Memory", address: int, elem_type: str, count: int, owned: bool):
        self.memory = memory
        self.address = address
        self.elem_type = elem_type
        self.count = count
        if owned:
            weakref.finalize(self, memory.release, address)


class Memory:
    def __init__(self) -> None:
        self.structs: dict[str, StructDef] = {}
        self._bases: list[int] = []
        self._blocks: dict[int, _Block] = {}
        self._tombstones: list[int] = []

    # ------------------------------------------------------------------ layout
    def sizeof(self, type_name: str) -> int:
        return self._layout(type_name)[0]

    def alignof(self, type_name: str) -> int:
        return self._layout(type_name)[1]

    def _layout(self, type_name: str) -> tuple[int, int]:
        type_name = type_name.strip()
        if is_pointer_type(type_name):
            return POINTER_SIZE, POINTER_SIZE
        if type_name in SCALARS:
            size = SCALARS[type_name][0]
            return size, size
        if type_name in self.structs:
            sdef = self.structs[type_name]
            return sdef.size, sdef.align
        if type_name == "void":
            raise MemoryFault("type", "void has no size")
        raise MemoryFault("type", f"unknown type '{type_name}'")

    def define_struct(self, name: str, fields: list[tuple[str, str, int | None]]) -> StructDef:
        """Lay out a struct the way C does: natural alignment, padded to a multiple of its alignment."""
        out: list[Field] = []
        offset = 0
        align = 1
        for fname, ftype, count in fields:
            size, falign = self._layout(ftype)
            offset = (offset + falign - 1) // falign * falign
            out.append(Field(fname, ftype, count, offset))
            offset += size * (count if count is not None else 1)
            align = max(align, falign)
        size = max((offset + align - 1) // align * align, 1) if out else 0
        sdef = StructDef(name, out, size, align)
        self.structs[name] = sdef
        return sdef

    # ------------------------------------------------------------------ allocation
    def alloc(self, size: int, heap: bool = False) -> Pointer:
        if size < 0:
            raise MemoryFault("pointer", f"cannot allocate {size} bytes")
        raw = ctypes.create_string_buffer(max(size, 1) + 16)
        base = (ctypes.addressof(raw) + 15) // 16 * 16
        self._drop_tombstones(base, max(size, 1))
        block = _Block(base=base, size=size, buffer=raw, heap=heap)
        bisect.insort(self._bases, base)
        self._blocks[base] = block
        return Pointer(base, "void")

    def free(self, ptr: Pointer) -> None:
        if ptr.address == 0:
            return
        block = self._blocks.get(ptr.address)
        if block is None:
            raise MemoryFault("pointer", "free() of a pointer that was not returned by malloc")
        if not block.live:
            raise MemoryFault("pointer", "double free")
        if not block.heap:
            raise MemoryFault("pointer", "free() of memory that was not allocated by malloc (a local, struct or array)")
        self._retire(block)

    def release(self, address: int) -> None:
        """Free an owned block (struct/array value or promoted local) without complaining if it is gone."""
        block = self._blocks.get(address)
        if block is not None and block.live:
            self._retire(block)

    def _retire(self, block: _Block) -> None:
        block.live = False
        block.buffer = None
        self._tombstones.append(block.base)
        while len(self._tombstones) > MAX_TOMBSTONES:
            old = self._tombstones.pop(0)
            stale = self._blocks.get(old)
            if stale is not None and not stale.live:
                self._remove(old)

    def _remove(self, base: int) -> None:
        self._blocks.pop(base, None)
        i = bisect.bisect_left(self._bases, base)
        if i < len(self._bases) and self._bases[i] == base:
            del self._bases[i]

    def _drop_tombstones(self, base: int, size: int) -> None:
        for other in [b for b in self._bases if base <= b < base + size or (b < base and self._blocks[b].base + max(self._blocks[b].size, 1) > base)]:
            if not self._blocks[other].live:
                self._remove(other)
                if other in self._tombstones:
                    self._tombstones.remove(other)

    def realloc(self, ptr: Pointer, size: int) -> Pointer:
        if ptr.address == 0:
            return self.alloc(size, heap=True)
        block = self._blocks.get(ptr.address)
        if block is None or not block.live or not block.heap:
            raise MemoryFault("pointer", "realloc() of a pointer that is not a live malloc allocation")
        new = self.alloc(size, heap=True)
        keep = min(block.size, size)
        if keep:
            ctypes.memmove(new.address, block.base, keep)
        self._retire(block)
        return new

    def usable_size(self, address: int) -> int | None:
        block = self._blocks.get(address)
        return block.size if block is not None and block.live else None

    # ------------------------------------------------------------------ checking
    def check(self, address: int, size: int, foreign: bool = False, align: int = 1) -> None:
        if address == 0:
            raise MemoryFault("null", "null pointer dereference")
        if foreign:
            return
        block = self._find(address)
        if block is None:
            raise MemoryFault("pointer", "access to memory that ManV did not allocate")
        if not block.live:
            raise MemoryFault("pointer", "use after free")
        if address < block.base or address + size > block.base + block.size:
            raise MemoryFault("pointer", f"out-of-bounds access ({size} byte(s) at offset {address - block.base} of a {block.size}-byte block)")
        if align > 1 and address % align:
            raise MemoryFault("pointer", f"misaligned access (address is not a multiple of {align})")

    def _find(self, address: int) -> _Block | None:
        i = bisect.bisect_right(self._bases, address) - 1
        if i < 0:
            return None
        block = self._blocks[self._bases[i]]
        if address < block.base + max(block.size, 1):
            return block
        return None

    # ------------------------------------------------------------------ raw bytes
    def read_bytes(self, address: int, size: int, foreign: bool = False) -> bytes:
        if size:
            self.check(address, size, foreign)
        return ctypes.string_at(address, size) if size else b""

    def write_bytes(self, address: int, data: bytes, foreign: bool = False) -> None:
        if data:
            self.check(address, len(data), foreign)
            ctypes.memmove(address, data, len(data))

    def move(self, dst: Pointer, src: Pointer, size: int) -> None:
        if size < 0:
            raise MemoryFault("pointer", "negative size")
        if size:
            self.check(src.address, size, src.foreign)
            self.check(dst.address, size, dst.foreign)
            ctypes.memmove(dst.address, src.address, size)

    def fill(self, dst: Pointer, byte: int, size: int) -> None:
        if size < 0:
            raise MemoryFault("pointer", "negative size")
        if size:
            self.check(dst.address, size, dst.foreign)
            ctypes.memset(dst.address, byte & 0xFF, size)

    def compare(self, a: Pointer, b: Pointer, size: int) -> int:
        if size < 0:
            raise MemoryFault("pointer", "negative size")
        if not size:
            return 0
        left = self.read_bytes(a.address, size, a.foreign)
        right = self.read_bytes(b.address, size, b.foreign)
        return (left > right) - (left < right)

    def read_cstring(self, ptr: Pointer, limit: int | None = None) -> bytes:
        """Bytes up to (not including) the NUL, or exactly `limit` bytes."""
        if limit is not None:
            return self.read_bytes(ptr.address, limit, ptr.foreign)
        if ptr.address == 0:
            raise MemoryFault("null", "null pointer dereference")
        if ptr.foreign:
            return ctypes.string_at(ptr.address)
        block = self._find(ptr.address)
        if block is None or not block.live:
            self.check(ptr.address, 1)
        available = block.base + block.size - ptr.address
        data = ctypes.string_at(ptr.address, max(available, 0))
        end = data.find(b"\0")
        if end < 0:
            raise MemoryFault("pointer", "string is not NUL-terminated inside its block")
        return data[:end]

    # ------------------------------------------------------------------ typed access
    def load(self, ptr: Pointer, type_name: str | None = None) -> Any:
        """Read a value of `type_name` (default: the pointer's pointee) from `ptr`."""
        type_name = (type_name or ptr.pointee).strip()
        if type_name == "void":
            raise MemoryFault("type", "cannot dereference a void*")
        if type_name in self.structs:
            self.check(ptr.address, self.structs[type_name].size, ptr.foreign, self.structs[type_name].align)
            return StructRef(self, ptr.address, self.structs[type_name], owned=False)
        size, align = self._layout(type_name)
        self.check(ptr.address, size, ptr.foreign, align)
        data = ctypes.string_at(ptr.address, size)
        if is_pointer_type(type_name):
            (address,) = struct.unpack("<Q", data)
            return Pointer(address, pointee_of(type_name), foreign=address != 0 and self._find(address) is None)
        return struct.unpack("<" + SCALARS[type_name][1], data)[0]

    def store(self, ptr: Pointer, value: Any, type_name: str | None = None) -> None:
        type_name = (type_name or ptr.pointee).strip()
        if type_name == "void":
            raise MemoryFault("type", "cannot store through a void*")
        if type_name in self.structs:
            sdef = self.structs[type_name]
            if not isinstance(value, StructRef) or value.sdef.name != sdef.name:
                raise MemoryFault("type", f"expected a {sdef.name} value")
            self.move(Pointer(ptr.address, type_name, ptr.foreign), Pointer(value.address, type_name), sdef.size)
            return
        size, align = self._layout(type_name)
        self.check(ptr.address, size, ptr.foreign, align)
        ctypes.memmove(ptr.address, self.encode(value, type_name), size)

    def encode(self, value: Any, type_name: str) -> bytes:
        if is_pointer_type(type_name):
            if isinstance(value, ArrayRef):
                value = Pointer(value.address, value.elem_type)
            if not isinstance(value, Pointer):
                raise MemoryFault("type", f"cannot store {type(value).__name__} in a {type_name} slot (use 'as' to convert)")
            return struct.pack("<Q", value.address)
        size, fmt = SCALARS[type_name]
        if type_name in INT_TYPES:
            if isinstance(value, bool) or not isinstance(value, int):
                if isinstance(value, bool):
                    value = int(value)
                else:
                    raise MemoryFault("type", f"cannot store {type(value).__name__} in a {type_name} slot")
            bits = size * 8
            value &= (1 << bits) - 1
            if fmt.islower() and value >= 1 << (bits - 1):
                value -= 1 << bits
            return struct.pack("<" + fmt, value)
        if type_name in FLOAT_TYPES:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MemoryFault("type", f"cannot store {type(value).__name__} in a {type_name} slot")
            try:
                return struct.pack("<" + fmt, float(value))
            except OverflowError:
                return struct.pack("<" + fmt, float("inf") if value > 0 else float("-inf"))
        return struct.pack("<?", bool(value))

    # ------------------------------------------------------------------ owned values
    def new_struct(self, name: str) -> StructRef:
        sdef = self.structs[name]
        ptr = self.alloc(sdef.size)
        return StructRef(self, ptr.address, sdef, owned=True)

    def new_array(self, elem_type: str, count: int) -> ArrayRef:
        ptr = self.alloc(self.sizeof(elem_type) * count)
        return ArrayRef(self, ptr.address, elem_type, count, owned=True)

    def copy_struct(self, value: StructRef) -> StructRef:
        out = self.new_struct(value.sdef.name)
        self.move(Pointer(out.address), Pointer(value.address), value.sdef.size)
        return out

    def field_pointer(self, ref: StructRef, name: str) -> tuple[Pointer, Field]:
        f = ref.sdef.field(name)
        if f is None:
            raise MemoryFault("type", f"struct {ref.sdef.name} has no field '{name}'")
        elem = f.type_name
        return Pointer(ref.address + f.offset, elem), f
