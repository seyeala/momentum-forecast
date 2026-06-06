"""Brownian residual channel-life formulas."""

from __future__ import annotations

import math

from momentum_forecast._validation import require_positive, require_positive_int


def expected_channel_life(band: float, residual_volatility: float) -> float:
    """Expected first-exit time from a symmetric Brownian channel."""
    band = require_positive(band, "band")
    residual_volatility = require_positive(residual_volatility, "residual_volatility")
    return band**2 / residual_volatility**2


def channel_survival_probability(
    horizon: float,
    band: float,
    residual_volatility: float,
    terms: int = 100,
) -> float:
    """Probability a Brownian residual remains inside ``[-band, band]``."""
    horizon = require_positive(horizon, "horizon")
    band = require_positive(band, "band")
    residual_volatility = require_positive(residual_volatility, "residual_volatility")
    terms = require_positive_int(terms, "terms")

    total = 0.0
    scale = math.pi**2 * residual_volatility**2 * horizon / (8.0 * band**2)
    for k in range(terms):
        odd = 2 * k + 1
        total += ((-1.0) ** k) / odd * math.exp(-(odd**2) * scale)
    probability = 4.0 / math.pi * total
    return min(1.0, max(0.0, probability))
