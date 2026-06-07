"""Lightweight rolling-window fitting helpers.

Each result is aligned to the window endpoint.  For ``n`` observations and a
window of ``w``, the returned list has length ``n - w + 1``; output index 0 uses
exactly observations ``[0:w]`` and later outputs use only observations at or
before their endpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from momentum_forecast._validation import require_positive_int
from momentum_forecast.fitting._stats import finite_float_list
from momentum_forecast.fitting.ar1 import fit_ar1_phi
from momentum_forecast.fitting.drift import fit_mean_drift
from momentum_forecast.fitting.types import AR1Fit, DriftFit, VolatilityFit
from momentum_forecast.fitting.volatility import fit_return_variance

T = TypeVar("T")


def _rolling(values: Iterable[float], window: int, fitter: Callable[[list[float]], T]) -> list[T]:
    window = require_positive_int(window, "window")
    data = finite_float_list(values, "returns")
    if len(data) < window:
        raise ValueError("not enough data to produce a rolling fit")
    return [fitter(data[end - window : end]) for end in range(window, len(data) + 1)]


def rolling_mean_drift(returns: Iterable[float], window: int) -> list[DriftFit]:
    """Return rolling mean-drift fits aligned to each window endpoint."""
    return _rolling(returns, window, fit_mean_drift)


def rolling_return_variance(returns: Iterable[float], window: int) -> list[VolatilityFit]:
    """Return rolling sample-variance fits aligned to each window endpoint."""
    return _rolling(returns, window, fit_return_variance)


def rolling_ar1_phi(returns: Iterable[float], window: int) -> list[AR1Fit]:
    """Return rolling AR(1) fits aligned to each window endpoint."""
    return _rolling(returns, window, fit_ar1_phi)
