"""Drift estimators for returns and log-price lines."""

from __future__ import annotations

from collections.abc import Iterable

from momentum_forecast.fitting._stats import finite_float_list, mean, ols_line, safe_t_stat, sample_variance, sqrt
from momentum_forecast.fitting.types import DriftFit


def fit_mean_drift(returns: Iterable[float]) -> DriftFit:
    """Estimate per-bar drift as the sample mean of returns.

    The standard error uses sample variance with ``n - 1`` degrees of freedom;
    therefore at least two returns are required.
    """
    values = finite_float_list(returns, "returns")
    if len(values) < 2:
        raise ValueError("at least two returns are required")

    drift = mean(values)
    standard_error = sqrt(sample_variance(values) / len(values))
    return DriftFit(
        drift=drift,
        standard_error=standard_error,
        t_stat=safe_t_stat(drift, standard_error),
        lookback=len(values),
        method="sample_mean",
    )


def fit_ols_line_drift(log_prices: Iterable[float]) -> DriftFit:
    """Estimate per-bar drift as the OLS slope of log prices on time index."""
    values = finite_float_list(log_prices, "log_prices")
    if len(values) < 2:
        raise ValueError("at least two log prices are required")

    slope, _intercept, residuals, sxx = ols_line(values)
    standard_error = None
    if len(values) > 2:
        residual_variance = sum(residual * residual for residual in residuals) / (len(values) - 2)
        standard_error = sqrt(residual_variance / sxx)

    return DriftFit(
        drift=slope,
        standard_error=standard_error,
        t_stat=safe_t_stat(slope, standard_error),
        lookback=len(values),
        method="ols_log_price_line",
    )
