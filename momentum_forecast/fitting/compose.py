"""Composition helpers that bridge fitted parameters to Phase 1 formulas."""

from __future__ import annotations

from collections.abc import Iterable

from momentum_forecast._validation import Side
from momentum_forecast.fitting.drift import fit_mean_drift
from momentum_forecast.fitting.types import DriftFit, VolatilityFit
from momentum_forecast.fitting.volatility import fit_return_variance
from momentum_forecast.models import estimated_drift_profit_probability


def fit_estimated_drift_inputs(returns: Iterable[float]) -> tuple[DriftFit, VolatilityFit]:
    """Fit mean drift and return variance for the estimated-drift formula."""
    values = list(returns)
    return fit_mean_drift(values), fit_return_variance(values)


def estimated_drift_probability_from_fits(
    drift_fit: DriftFit,
    volatility_fit: VolatilityFit,
    horizon: float,
    cost: float,
    side: Side = "long",
) -> float:
    """Compute estimated-drift probability from fit outputs without trading rules."""
    if drift_fit.lookback != volatility_fit.lookback:
        raise ValueError("drift and volatility fits must use the same lookback")
    return estimated_drift_profit_probability(
        mu_hat=drift_fit.drift,
        q_hat=volatility_fit.variance,
        lookback=drift_fit.lookback,
        horizon=horizon,
        cost=cost,
        side=side,
    )
