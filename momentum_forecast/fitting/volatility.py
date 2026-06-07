"""Volatility and variance estimators for return series."""

from __future__ import annotations

from collections.abc import Iterable

from momentum_forecast._validation import require_finite
from momentum_forecast.fitting._stats import finite_float_list, sample_variance, sqrt
from momentum_forecast.fitting.types import VolatilityFit


def fit_return_variance(returns: Iterable[float]) -> VolatilityFit:
    """Estimate sample return variance using ``n - 1`` degrees of freedom."""
    values = finite_float_list(returns, "returns")
    if len(values) < 2:
        raise ValueError("at least two returns are required")
    variance = sample_variance(values)
    return VolatilityFit(variance=variance, volatility=sqrt(variance), lookback=len(values), method="sample_variance")


def fit_return_volatility(returns: Iterable[float]) -> VolatilityFit:
    """Estimate sample return volatility using ``n - 1`` degrees of freedom."""
    fit = fit_return_variance(returns)
    return VolatilityFit(
        variance=fit.variance,
        volatility=fit.volatility,
        lookback=fit.lookback,
        method="sample_volatility",
    )


def fit_ewma_volatility(returns: Iterable[float], alpha: float) -> VolatilityFit:
    """Estimate EWMA volatility.

    Initialization behavior: the first EWMA variance is initialized to the first
    squared return, then each following return applies
    ``variance_t = alpha * return_t**2 + (1 - alpha) * variance_{t-1}``.
    """
    alpha = require_finite(alpha, "alpha")
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    values = finite_float_list(returns, "returns")
    if not values:
        raise ValueError("at least one return is required")

    variance = values[0] ** 2
    for value in values[1:]:
        variance = alpha * value**2 + (1.0 - alpha) * variance
    return VolatilityFit(
        variance=variance,
        volatility=sqrt(variance),
        lookback=len(values),
        method=f"ewma_alpha={alpha:g}",
    )
