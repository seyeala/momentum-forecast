"""Log-price channel and residual-band fitting."""

from __future__ import annotations

import math
from collections.abc import Iterable

from momentum_forecast._validation import require_finite, require_positive
from momentum_forecast.fitting._stats import finite_float_list, ols_line, sqrt
from momentum_forecast.fitting.types import ChannelFit
from momentum_forecast.models import expected_channel_life


def fit_channel_band_from_residual_quantile(residuals: Iterable[float], quantile: float = 0.95) -> float:
    """Fit a symmetric channel band from an absolute-residual quantile.

    The quantile uses a nearest-rank rule over sorted absolute residuals.
    """
    quantile = require_finite(quantile, "quantile")
    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError("quantile must satisfy 0 < quantile < 1")
    values = [abs(value) for value in finite_float_list(residuals, "residuals")]
    if not values:
        raise ValueError("at least one residual is required")
    values.sort()
    index = math.ceil(quantile * len(values)) - 1
    return values[max(0, min(index, len(values) - 1))]


def fit_channel_band_from_volatility(residual_volatility: float, multiplier: float = 2.0) -> float:
    """Fit a symmetric channel band as ``multiplier * residual_volatility``."""
    residual_volatility = require_positive(residual_volatility, "residual_volatility")
    multiplier = require_positive(multiplier, "multiplier")
    return multiplier * residual_volatility


def fit_line_residuals(log_prices: Iterable[float]) -> ChannelFit:
    """Fit an OLS log-price line and residual channel.

    Default band rule: use ``2 * residual_volatility``.  If the series is a
    perfect line with zero residual volatility, the band is zero and the expected
    channel life is mathematically infinite.
    """
    values = finite_float_list(log_prices, "log_prices")
    if len(values) < 3:
        raise ValueError("at least three log prices are required")

    slope, intercept, residuals, _sxx = ols_line(values)
    residual_variance = sum(residual * residual for residual in residuals) / (len(values) - 2)
    if residual_variance <= 1e-30:
        residual_variance = 0.0
    residual_volatility = sqrt(residual_variance)
    if residual_volatility == 0.0:
        band = 0.0
        life = math.inf
    else:
        band = fit_channel_band_from_volatility(residual_volatility)
        life = expected_channel_life(band, residual_volatility)

    return ChannelFit(
        slope=slope,
        intercept=intercept,
        residual_variance=residual_variance,
        residual_volatility=residual_volatility,
        band=band,
        expected_life=life,
        lookback=len(values),
        method="ols_line_residuals_band=2sigma",
    )
