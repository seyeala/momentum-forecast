"""Constant-hazard regime lifetime formulas."""

from __future__ import annotations

import math

from momentum_forecast._validation import require_finite, require_non_negative, require_positive


def regime_survival_probability(hazard: float, horizon: float) -> float:
    """Survival probability under a constant regime-death hazard."""
    hazard = require_positive(hazard, "hazard")
    horizon = require_non_negative(horizon, "horizon")
    return math.exp(-hazard * horizon)


def expected_regime_life(hazard: float) -> float:
    """Expected lifetime under a constant regime-death hazard."""
    hazard = require_positive(hazard, "hazard")
    return 1.0 / hazard


def hazard_adjusted_drift_gain(drift: float, hazard: float, horizon: float) -> float:
    """Expected drift contribution when a trend can die with constant hazard."""
    drift = require_finite(drift, "drift")
    hazard = require_positive(hazard, "hazard")
    horizon = require_non_negative(horizon, "horizon")
    return drift * (-math.expm1(-hazard * horizon)) / hazard
