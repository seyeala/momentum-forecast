"""Internal validation helpers for momentum forecast modules."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal, TypeAlias

Side: TypeAlias = Literal["long", "short"]


def require_finite(value: float, name: str) -> float:
    """Return *value* as float after validating it is finite."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def require_positive(value: float, name: str) -> float:
    """Return *value* after validating it is finite and strictly positive."""
    value = require_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_non_negative(value: float, name: str) -> float:
    """Return *value* after validating it is finite and non-negative."""
    value = require_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def require_positive_int(value: int, name: str) -> int:
    """Return *value* after validating it is a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_side(side: str) -> Side:
    """Validate a trade side literal."""
    if side not in {"long", "short"}:
        raise ValueError("side must be either 'long' or 'short'")
    return side  # type: ignore[return-value]


def as_float_list(values: Iterable[float], name: str) -> list[float]:
    """Materialize an iterable of finite floats."""
    result = [require_finite(value, f"{name}[{index}]") for index, value in enumerate(values)]
    return result
