"""Internal statistical helpers for fitting modules."""

from __future__ import annotations

import math
from collections.abc import Iterable

from momentum_forecast._validation import as_float_list


def finite_float_list(values: Iterable[float], name: str) -> list[float]:
    return as_float_list(values, name)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two observations are required")
    m = mean(values)
    return sum((value - m) ** 2 for value in values) / (len(values) - 1)


def ols_line(values: list[float]) -> tuple[float, float, list[float], float]:
    """Return slope, intercept, residuals, and sum of centered time squares."""
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = mean(values)
    sxx = sum((index - x_mean) ** 2 for index in range(n))
    slope = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / sxx
    intercept = y_mean - slope * x_mean
    residuals = [value - (intercept + slope * index) for index, value in enumerate(values)]
    return slope, intercept, residuals, sxx


def safe_t_stat(estimate: float, standard_error: float | None) -> float | None:
    if standard_error is None or standard_error == 0.0:
        return None
    return estimate / standard_error


def sqrt(value: float) -> float:
    return math.sqrt(max(0.0, value))
