"""Constant-hazard lifetime fitting."""

from __future__ import annotations

from collections.abc import Iterable

from momentum_forecast.fitting._stats import finite_float_list, mean
from momentum_forecast.fitting.types import HazardFit
from momentum_forecast.models import expected_regime_life


def _positive_lifetimes(lifetimes: Iterable[float]) -> list[float]:
    values = finite_float_list(lifetimes, "lifetimes")
    if not values:
        raise ValueError("at least one lifetime is required")
    if any(value <= 0.0 for value in values):
        raise ValueError("lifetimes must be positive")
    return values


def fit_constant_hazard(lifetimes: Iterable[float]) -> HazardFit:
    """Fit complete-observation hazard as ``1 / mean(lifetimes)``."""
    values = _positive_lifetimes(lifetimes)
    hazard = 1.0 / mean(values)
    return HazardFit(
        hazard=hazard,
        expected_life=expected_regime_life(hazard),
        n_events=len(values),
        n_censored=0,
        method="complete_lifetime_mle",
    )


def fit_constant_hazard_with_censoring(lifetimes: Iterable[float], observed: Iterable[bool]) -> HazardFit:
    """Fit right-censored constant hazard as events divided by time at risk."""
    values = _positive_lifetimes(lifetimes)
    observed_values = list(observed)
    if len(values) != len(observed_values):
        raise ValueError("lifetimes and observed must have equal lengths")
    if any(not isinstance(value, bool) for value in observed_values):
        raise ValueError("observed values must be booleans")

    n_events = sum(1 for value in observed_values if value)
    if n_events <= 0:
        raise ValueError("at least one observed event is required")
    n_censored = len(observed_values) - n_events
    hazard = n_events / sum(values)
    return HazardFit(
        hazard=hazard,
        expected_life=expected_regime_life(hazard),
        n_events=n_events,
        n_censored=n_censored,
        method="right_censored_mle",
    )
