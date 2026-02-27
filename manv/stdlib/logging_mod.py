from __future__ import annotations

import logging as _logging

DEBUG = _logging.DEBUG
INFO = _logging.INFO
WARNING = _logging.WARNING
ERROR = _logging.ERROR
CRITICAL = _logging.CRITICAL


def get_logger(name: str) -> _logging.Logger:
    return _logging.getLogger(name)


def basic_config(level: int = INFO) -> None:
    _logging.basicConfig(level=level)
