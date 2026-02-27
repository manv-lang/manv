from __future__ import annotations

import csv as _csv
from typing import Iterable, TextIO

reader = _csv.reader
writer = _csv.writer
DictReader = _csv.DictReader
DictWriter = _csv.DictWriter


def register_dialect(name: str, **kwargs: object) -> None:
    _csv.register_dialect(name, **kwargs)


def write_rows(out: TextIO, rows: Iterable[Iterable[object]]) -> None:
    w = _csv.writer(out)
    for row in rows:
        w.writerow(row)
