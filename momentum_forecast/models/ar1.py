"""AR(1) momentum persistence formulas."""

from __future__ import annotations

import math

from momentum_forecast._validation import require_finite, require_positive_int


def ar1_expected_return_at_horizon(current_return: float, phi: float, h: int) -> float:
    """Expected return h bars ahead for an AR(1) process."""
    current_return = require_finite(current_return, "current_return")
    phi = require_finite(phi, "phi")
    h = require_positive_int(h, "h")
    return (phi**h) * current_return


def ar1_expected_cumulative_return(current_return: float, phi: float, horizon: int) -> float:
    """Expected cumulative AR(1) return over the next horizon bars."""
    current_return = require_finite(current_return, "current_return")
    phi = require_finite(phi, "phi")
    horizon = require_positive_int(horizon, "horizon")
    if phi == 1.0:
        return current_return * horizon
    return current_return * phi * (1.0 - phi**horizon) / (1.0 - phi)


def ar1_half_life(phi: float) -> float | None:
    """Return AR(1) half-life when 0 < phi < 1; otherwise return None."""
    phi = require_finite(phi, "phi")
    if phi <= 0.0 or phi >= 1.0:
        return None
    return math.log(0.5) / math.log(phi)
