from __future__ import annotations

import asyncio
from typing import Any


def run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


async def sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)
