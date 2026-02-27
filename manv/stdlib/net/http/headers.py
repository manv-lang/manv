from __future__ import annotations


class Headers(dict[str, str]):
    def normalized(self) -> dict[str, str]:
        return {k.lower(): v for k, v in self.items()}
