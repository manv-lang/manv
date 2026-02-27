from __future__ import annotations


class BaseException_(Exception):
    """Language-level base exception alias for stdlib surfaces."""


class RuntimeError_(RuntimeError):
    pass


class ImportError_(RuntimeError_):
    pass


class IOError_(RuntimeError_):
    pass


class OSError_(IOError_):
    pass


class NetworkError(RuntimeError_):
    pass


class TimeoutError_(NetworkError):
    pass


class ConcurrencyError(RuntimeError_):
    pass


class SerializationError(RuntimeError_):
    pass


class CapabilityError(RuntimeError_):
    pass


class KernelError(RuntimeError_):
    pass


class GPUError(KernelError):
    pass
