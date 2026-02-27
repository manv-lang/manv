from __future__ import annotations

import zipfile


class ZipWriter:
    def __init__(self, path: str):
        self._z = zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED)

    def add_bytes(self, name: str, payload: bytes) -> None:
        self._z.writestr(name, payload)

    def close(self) -> None:
        self._z.close()


class ZipReader:
    def __init__(self, path: str):
        self._z = zipfile.ZipFile(path, mode="r")

    def names(self) -> list[str]:
        return self._z.namelist()

    def read(self, name: str) -> bytes:
        return self._z.read(name)

    def close(self) -> None:
        self._z.close()
