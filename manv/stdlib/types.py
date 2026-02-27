from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from socket import socket

BigInt = int


@dataclass
class Random:
    seed: int | None = None


@dataclass
class Future:
    done: bool = False
    value: object | None = None


@dataclass
class Task(Future):
    name: str = ""


PathType = Path
SocketType = socket
TensorType = object
DecimalType = Decimal
