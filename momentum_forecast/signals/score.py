"""Momentum signal-to-noise scoring."""

from __future__ import annotations

import math
from typing import Literal

from momentum_forecast._validation import require_finite, require_positive, require_positive_int

MomentumStrength = Literal["weak", "moderate", "strong", "extreme"]


def momentum_z_score(mu_hat: float, q_hat: float, lookback: int) -> float:
    """Return observed directional move divided by return noise."""
    mu_hat = require_finite(mu_hat, "mu_hat")
    q_hat = require_positive(q_hat, "q_hat")
    lookback = require_positive_int(lookback, "lookback")
    return mu_hat * math.sqrt(lookback) / math.sqrt(q_hat)


def classify_momentum_strength(z: float) -> MomentumStrength:
    """Classify signal strength using absolute z-score thresholds."""
    abs_z = abs(require_finite(z, "z"))
    if abs_z < 1.0:
        return "weak"
    if abs_z < 2.0:
        return "moderate"
    if abs_z < 3.0:
        return "strong"
    return "extreme"
