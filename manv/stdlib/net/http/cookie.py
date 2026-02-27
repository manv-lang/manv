from __future__ import annotations


def parse_set_cookie(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in value.split(";"):
        token = part.strip()
        if not token:
            continue
        if "=" in token:
            k, v = token.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[token] = ""
    return out
