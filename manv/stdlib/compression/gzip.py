from __future__ import annotations

import gzip as _gzip


def compress(data: bytes) -> bytes:
    return _gzip.compress(data)


def decompress(data: bytes) -> bytes:
    return _gzip.decompress(data)
