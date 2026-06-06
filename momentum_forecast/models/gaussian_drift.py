"""Known-drift Gaussian momentum probability formulas."""

from __future__ import annotations

import math

from momentum_forecast._validation import Side, require_finite, require_non_negative, require_positive, validate_side


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def known_drift_profit_probability(
    drift: float,
    volatility: float,
    horizon: float,
    cost: float,
    side: Side = "long",
) -> float:
    """Probability that a known-drift Gaussian move beats cost."""
    drift = require_finite(drift, "drift")
    volatility = require_positive(volatility, "volatility")
    horizon = require_positive(horizon, "horizon")
    cost = require_non_negative(cost, "cost")
    side = validate_side(side)

    directional_mean = drift * horizon if side == "long" else -drift * horizon
    z = (directional_mean - cost) / (volatility * math.sqrt(horizon))
    return _normal_cdf(z)
