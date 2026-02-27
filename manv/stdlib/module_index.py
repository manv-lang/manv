from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


@dataclass(frozen=True)
class ModuleSpec:
    logical_name: str
    python_module: str
    tier: int
    stable: bool


_LOGICAL: list[ModuleSpec] = [
    ModuleSpec("std.builtins", "manv.stdlib.builtins", 1, True),
    ModuleSpec("std.errors", "manv.stdlib.errors", 1, True),
    ModuleSpec("std.protocols", "manv.stdlib.protocols", 1, True),
    ModuleSpec("std.types", "manv.stdlib.types", 1, True),
    ModuleSpec("std.iter", "manv.stdlib.iter", 1, True),
    ModuleSpec("std.functional", "manv.stdlib.functional", 1, True),
    ModuleSpec("std.collections", "manv.stdlib.collections", 1, True),
    ModuleSpec("std.collections.deque", "manv.stdlib.collections.deque", 1, True),
    ModuleSpec("std.collections.counter", "manv.stdlib.collections.counter", 1, True),
    ModuleSpec("std.collections.defaultmap", "manv.stdlib.collections.defaultmap", 1, True),
    ModuleSpec("std.collections.orderedmap", "manv.stdlib.collections.orderedmap", 1, True),
    ModuleSpec("std.collections.heap", "manv.stdlib.collections.heap", 1, True),
    ModuleSpec("std.collections.lru", "manv.stdlib.collections.lru", 1, True),
    ModuleSpec("std.array", "manv.stdlib.array", 1, True),
    ModuleSpec("std.buffer", "manv.stdlib.buffer", 1, True),
    ModuleSpec("std.bytes", "manv.stdlib.bytes", 1, True),
    ModuleSpec("std.math", "manv.stdlib.math", 1, True),
    ModuleSpec("std.random", "manv.stdlib.random", 1, True),
    ModuleSpec("std.statistics", "manv.stdlib.statistics", 1, True),
    ModuleSpec("std.decimal", "manv.stdlib.decimal", 1, True),
    ModuleSpec("std.big", "manv.stdlib.big", 1, True),
    ModuleSpec("std.string", "manv.stdlib.string", 1, True),
    ModuleSpec("std.re", "manv.stdlib.re", 1, True),
    ModuleSpec("std.text.tokenize", "manv.stdlib.text.tokenize", 1, True),
    ModuleSpec("std.text.normalize", "manv.stdlib.text.normalize", 1, True),
    ModuleSpec("std.json", "manv.stdlib.json", 1, True),
    ModuleSpec("std.csv", "manv.stdlib.csv", 1, True),
    ModuleSpec("std.config.ini", "manv.stdlib.config.ini", 2, True),
    ModuleSpec("std.config.toml", "manv.stdlib.config.toml", 2, True),
    ModuleSpec("std.config.env", "manv.stdlib.config.env", 2, True),
    ModuleSpec("std.io", "manv.stdlib.io", 1, True),
    ModuleSpec("std.io.text", "manv.stdlib.io.text", 1, True),
    ModuleSpec("std.io.binary", "manv.stdlib.io.binary", 1, True),
    ModuleSpec("std.io.buffered", "manv.stdlib.io.buffered", 1, True),
    ModuleSpec("std.path", "manv.stdlib.path", 1, True),
    ModuleSpec("std.fs", "manv.stdlib.fs", 1, True),
    ModuleSpec("std.os", "manv.stdlib.os", 1, True),
    ModuleSpec("std.sys", "manv.stdlib.sys", 1, True),
    ModuleSpec("std.process", "manv.stdlib.process", 1, True),
    ModuleSpec("std.signal", "manv.stdlib.signal", 1, True),
    ModuleSpec("std.platform", "manv.stdlib.platform", 1, True),
    ModuleSpec("std.socket", "manv.stdlib.socket", 1, True),
    ModuleSpec("std.net.url", "manv.stdlib.net.url", 1, True),
    ModuleSpec("std.net.http", "manv.stdlib.net.http", 1, True),
    ModuleSpec("std.net.http.server", "manv.stdlib.net.http.server", 2, True),
    ModuleSpec("std.net.http.client", "manv.stdlib.net.http.client", 1, True),
    ModuleSpec("std.net.http.headers", "manv.stdlib.net.http.headers", 1, True),
    ModuleSpec("std.net.http.cookie", "manv.stdlib.net.http.cookie", 1, True),
    ModuleSpec("std.async", "manv.stdlib.async_runtime", 2, True),
    ModuleSpec("std.async.loop", "manv.stdlib.async_runtime.loop", 2, True),
    ModuleSpec("std.async.future", "manv.stdlib.async_runtime.future", 2, True),
    ModuleSpec("std.async.task", "manv.stdlib.async_runtime.task", 2, True),
    ModuleSpec("std.async.streams", "manv.stdlib.async_runtime.streams", 2, True),
    ModuleSpec("std.threading", "manv.stdlib.threading", 1, True),
    ModuleSpec("std.threading.sync", "manv.stdlib.threading.sync", 1, True),
    ModuleSpec("std.threading.local", "manv.stdlib.threading.local", 1, True),
    ModuleSpec("std.multiprocessing", "manv.stdlib.multiprocessing", 2, True),
    ModuleSpec("std.ipc", "manv.stdlib.ipc", 2, True),
    ModuleSpec("std.binary.struct", "manv.stdlib.binary.struct", 1, True),
    ModuleSpec("std.serialization.pickle", "manv.stdlib.serialization.pickle", 1, True),
    ModuleSpec("std.compression.gzip", "manv.stdlib.compression.gzip", 1, True),
    ModuleSpec("std.compression.zip", "manv.stdlib.compression.zip", 1, True),
    ModuleSpec("std.logging", "manv.stdlib.logging_mod", 1, True),
    ModuleSpec("std.logging.handlers", "manv.stdlib.logging_handlers", 1, True),
    ModuleSpec("std.testing.unittest", "manv.stdlib.testing.unittest", 1, True),
    ModuleSpec("std.testing.property", "manv.stdlib.testing.property", 2, True),
    ModuleSpec("std.testing.bench", "manv.stdlib.testing.bench", 2, True),
    ModuleSpec("std.inspect", "manv.stdlib.inspect", 1, True),
    ModuleSpec("std.importlib", "manv.stdlib.importlib", 1, True),
    ModuleSpec("std.typing", "manv.stdlib.typing", 1, True),
    ModuleSpec("std.tensor", "manv.stdlib.tensor", 1, True),
    ModuleSpec("std.kernel", "manv.stdlib.kernel", 1, True),
    ModuleSpec("std.profiler", "manv.stdlib.profiler", 1, True),
    ModuleSpec("std.debug", "manv.stdlib.debug", 2, True),
    ModuleSpec("std.trace", "manv.stdlib.trace", 1, True),
    ModuleSpec("std.compat.python", "manv.stdlib.compat.python", 1, True),
]

LOGICAL_MODULES: dict[str, ModuleSpec] = {spec.logical_name: spec for spec in _LOGICAL}


def load_std_module(logical_name: str) -> Any:
    spec = LOGICAL_MODULES.get(logical_name)
    if spec is None:
        raise KeyError(f"unknown std module: {logical_name}")
    return importlib.import_module(spec.python_module)
