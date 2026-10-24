"""Compute counter: tokens processed, split by purpose so the cost table can separate them."""
from __future__ import annotations

from collections import Counter

CATEGORIES = ("train", "gate", "oracle", "tracking", "trigger")


class ComputeCounter:
    def __init__(self) -> None:
        self._tokens: Counter[str] = Counter()

    def add(self, category: str, n_tokens: int) -> None:
        if category not in CATEGORIES:
            raise KeyError(f"unknown category {category!r}; expected one of {CATEGORIES}")
        if n_tokens < 0:
            raise ValueError("n_tokens must be non-negative")
        self._tokens[category] += n_tokens

    def get(self, category: str) -> int:
        return self._tokens[category]

    def method_cost(self) -> int:
        """Tokens charged to the method: oracle and tracking are evaluation-only and excluded."""
        return self._tokens["train"] + self._tokens["gate"] + self._tokens["trigger"]

    def as_dict(self) -> dict[str, int]:
        return {c: self._tokens[c] for c in CATEGORIES}
