"""Estimated-drift Gaussian momentum probability formulas."""

from __future__ import annotations

import math

from momentum_forecast._validation import Side, require_finite, require_non_negative, require_positive, require_positive_int, validate_side
from momentum_forecast.models.gaussian_drift import _normal_cdf


def estimated_drift_profit_probability(
    mu_hat: float,
    q_hat: float,
    lookback: int,
    horizon: float,
    cost: float,
    side: Side = "long",
) -> float:
    """Probability that an estimated-drift Gaussian move beats cost.

    Includes the slope-estimation uncertainty penalty ``1 + horizon / lookback``.
    """
    mu_hat = require_finite(mu_hat, "mu_hat")
    q_hat = require_positive(q_hat, "q_hat")
    lookback = require_positive_int(lookback, "lookback")
    horizon = require_positive(horizon, "horizon")
    cost = require_non_negative(cost, "cost")
    side = validate_side(side)

    directional_mean = mu_hat * horizon if side == "long" else -mu_hat * horizon
    variance = q_hat * horizon * (1.0 + horizon / lookback)
    z = (directional_mean - cost) / math.sqrt(variance)
    return _normal_cdf(z)
