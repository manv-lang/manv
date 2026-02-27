from __future__ import annotations

import asyncio
from typing import Any


def create_task(coro: Any) -> asyncio.Task[Any]:
    return asyncio.create_task(coro)


async def wait(tasks):
    return await asyncio.wait(tasks)


async def gather(*aws):
    return await asyncio.gather(*aws)
