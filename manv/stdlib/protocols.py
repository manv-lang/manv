from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Iterable(Protocol):
    def __iter__(self) -> Any: ...


@runtime_checkable
class Iterator(Iterable, Protocol):
    def __next__(self) -> Any: ...


@runtime_checkable
class Sequence(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: Any) -> Any: ...


@runtime_checkable
class MutableSequence(Sequence, Protocol):
    def __setitem__(self, index: Any, value: Any) -> None: ...


@runtime_checkable
class Mapping(Protocol):
    def __getitem__(self, key: Any) -> Any: ...

    def keys(self) -> Any: ...

    def values(self) -> Any: ...

    def items(self) -> Any: ...


@runtime_checkable
class MutableMapping(Mapping, Protocol):
    def __setitem__(self, key: Any, value: Any) -> None: ...

    def __delitem__(self, key: Any) -> None: ...


@runtime_checkable
class Numeric(Protocol):
    def __add__(self, other: Any) -> Any: ...

    def __sub__(self, other: Any) -> Any: ...

    def __mul__(self, other: Any) -> Any: ...

    def __truediv__(self, other: Any) -> Any: ...


@runtime_checkable
class ContextManager(Protocol):
    def __enter__(self) -> Any: ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...


@runtime_checkable
class Callable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class AsyncIterable(Protocol):
    def __aiter__(self) -> Any: ...


@runtime_checkable
class AsyncIterator(AsyncIterable, Protocol):
    async def __anext__(self) -> Any: ...


@runtime_checkable
class Awaitable(Protocol):
    def __await__(self) -> Any: ...


@runtime_checkable
class Buffer(Protocol):
    @property
    def readonly(self) -> bool: ...

    @property
    def shape(self) -> tuple[int, ...] | None: ...

    @property
    def strides(self) -> tuple[int, ...] | None: ...
