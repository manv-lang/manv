from __future__ import annotations

import unicodedata


def format(template: str, **kwargs: object) -> str:
    return template.format(**kwargs)


def normalize(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text)


def tokenize_whitespace(text: str) -> list[str]:
    return text.split()
