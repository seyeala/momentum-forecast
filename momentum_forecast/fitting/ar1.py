"""AR(1) persistence fitting."""

from __future__ import annotations

from collections.abc import Iterable

from momentum_forecast.fitting._stats import finite_float_list, safe_t_stat, sqrt
from momentum_forecast.fitting.types import AR1Fit
from momentum_forecast.models import ar1_half_life


def fit_ar1_phi(returns: Iterable[float]) -> AR1Fit:
    """Fit ``r[t + 1] = phi * r[t] + eta[t + 1]`` by OLS through the origin."""
    values = finite_float_list(returns, "returns")
    if len(values) < 3:
        raise ValueError("at least three returns are required")

    x_values = values[:-1]
    y_values = values[1:]
    sum_x2 = sum(x * x for x in x_values)
    phi = sum(x * y for x, y in zip(x_values, y_values)) / sum_x2 if sum_x2 > 0.0 else 0.0
    residuals = [y - phi * x for x, y in zip(x_values, y_values)]
    degrees_of_freedom = len(residuals) - 1 if sum_x2 > 0.0 else len(residuals)
    innovation_variance = sum(residual * residual for residual in residuals) / degrees_of_freedom

    standard_error = sqrt(innovation_variance / sum_x2) if sum_x2 > 0.0 else None
    return AR1Fit(
        phi=phi,
        innovation_variance=innovation_variance,
        half_life=ar1_half_life(phi),
        standard_error=standard_error,
        t_stat=safe_t_stat(phi, standard_error),
        lookback=len(values),
    )
