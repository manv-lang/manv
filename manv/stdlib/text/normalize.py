from __future__ import annotations

import unicodedata


def normalize(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text)
