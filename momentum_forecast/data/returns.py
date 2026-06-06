"""Utilities for converting prices and returns into log-return units."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real

from momentum_forecast._validation import as_float_list, require_finite


def _is_scalar(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def log_prices(prices: float | Iterable[float]) -> float | list[float]:
    """Convert positive price levels to log-prices.

    A scalar input returns a scalar log-price.  An iterable input returns a list
    of log-prices with the same length.
    """
    if _is_scalar(prices):
        price = require_finite(float(prices), "price")
        if price <= 0:
            raise ValueError("prices must be positive")
        return math.log(price)

    values = as_float_list(prices, "prices")  # type: ignore[arg-type]
    if any(price <= 0 for price in values):
        raise ValueError("prices must be positive")
    return [math.log(price) for price in values]


def log_returns(prices: Iterable[float]) -> list[float]:
    """Return consecutive log returns from positive price levels."""
    logs = log_prices(prices)
    if not isinstance(logs, list):
        raise ValueError("prices must be an iterable with at least two values")
    if len(logs) < 2:
        raise ValueError("at least two prices are required")
    return [logs[index] - logs[index - 1] for index in range(1, len(logs))]


def simple_to_log_return(simple_return: float) -> float:
    """Convert a simple return, e.g. 0.01 for 1%, to a log return."""
    value = require_finite(simple_return, "simple_return")
    if value <= -1:
        raise ValueError("simple_return must be greater than -1")
    return math.log1p(value)


def bps_to_log_cost(bps: float) -> float:
    """Convert basis points to an approximately equivalent log-return cost."""
    bps = require_finite(bps, "bps")
    if bps < 0:
        raise ValueError("bps must be non-negative")
    return simple_to_log_return(bps / 10_000.0)
