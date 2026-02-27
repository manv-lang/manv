from __future__ import annotations

import configparser


def loads(text: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read_string(text)
    return parser


def dumps(parser: configparser.ConfigParser) -> str:
    from io import StringIO

    out = StringIO()
    parser.write(out)
    return out.getvalue()
