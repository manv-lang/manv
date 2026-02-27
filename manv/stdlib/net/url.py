from __future__ import annotations

from urllib.parse import ParseResult, urlencode, urlparse, urlunparse


def parse_url(url: str) -> ParseResult:
    return urlparse(url)


def unparse_url(parts: ParseResult) -> str:
    return urlunparse(parts)


def encode_query(params: dict[str, object]) -> str:
    return urlencode(params)
