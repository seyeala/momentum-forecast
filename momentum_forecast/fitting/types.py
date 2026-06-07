"""Structured fit result types for empirical parameter estimates."""

from __future__ import annotations

from dataclasses import dataclass

import math

from momentum_forecast._validation import require_finite, require_non_negative, require_positive, require_positive_int


def _require_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_method(value: str) -> str:
    if not value:
        raise ValueError("method must be non-empty")
    return value


@dataclass(frozen=True)
class DriftFit:
    """Estimated per-bar drift and optional uncertainty diagnostics."""

    drift: float
    standard_error: float | None
    t_stat: float | None
    lookback: int
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "drift", require_finite(self.drift, "drift"))
        if self.standard_error is not None:
            object.__setattr__(self, "standard_error", require_non_negative(self.standard_error, "standard_error"))
        if self.t_stat is not None:
            object.__setattr__(self, "t_stat", require_finite(self.t_stat, "t_stat"))
        object.__setattr__(self, "lookback", require_positive_int(self.lookback, "lookback"))
        object.__setattr__(self, "method", _require_method(self.method))


@dataclass(frozen=True)
class VolatilityFit:
    """Estimated per-bar return variance and volatility."""

    variance: float
    volatility: float
    lookback: int
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "variance", require_non_negative(self.variance, "variance"))
        object.__setattr__(self, "volatility", require_non_negative(self.volatility, "volatility"))
        object.__setattr__(self, "lookback", require_positive_int(self.lookback, "lookback"))
        object.__setattr__(self, "method", _require_method(self.method))


@dataclass(frozen=True)
class AR1Fit:
    """OLS-through-origin AR(1) persistence estimate."""

    phi: float
    innovation_variance: float
    half_life: float | None
    standard_error: float | None
    t_stat: float | None
    lookback: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "phi", require_finite(self.phi, "phi"))
        object.__setattr__(
            self, "innovation_variance", require_non_negative(self.innovation_variance, "innovation_variance")
        )
        if self.half_life is not None:
            object.__setattr__(self, "half_life", require_positive(self.half_life, "half_life"))
        if self.standard_error is not None:
            object.__setattr__(self, "standard_error", require_non_negative(self.standard_error, "standard_error"))
        if self.t_stat is not None:
            object.__setattr__(self, "t_stat", require_finite(self.t_stat, "t_stat"))
        object.__setattr__(self, "lookback", require_positive_int(self.lookback, "lookback"))


@dataclass(frozen=True)
class ChannelFit:
    """OLS log-price channel fit with residual-width diagnostics."""

    slope: float
    intercept: float
    residual_variance: float
    residual_volatility: float
    band: float
    expected_life: float
    lookback: int
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "slope", require_finite(self.slope, "slope"))
        object.__setattr__(self, "intercept", require_finite(self.intercept, "intercept"))
        object.__setattr__(self, "residual_variance", require_non_negative(self.residual_variance, "residual_variance"))
        object.__setattr__(self, "residual_volatility", require_non_negative(self.residual_volatility, "residual_volatility"))
        object.__setattr__(self, "band", require_non_negative(self.band, "band"))
        if not (math.isfinite(self.expected_life) or self.expected_life == math.inf):
            raise ValueError("expected_life must be finite or positive infinity")
        if self.expected_life <= 0:
            raise ValueError("expected_life must be positive")
        object.__setattr__(self, "lookback", require_positive_int(self.lookback, "lookback"))
        object.__setattr__(self, "method", _require_method(self.method))


@dataclass(frozen=True)
class HazardFit:
    """Constant-hazard lifetime estimate."""

    hazard: float
    expected_life: float
    n_events: int
    n_censored: int
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "hazard", require_positive(self.hazard, "hazard"))
        object.__setattr__(self, "expected_life", require_positive(self.expected_life, "expected_life"))
        object.__setattr__(self, "n_events", _require_non_negative_int(self.n_events, "n_events"))
        object.__setattr__(self, "n_censored", _require_non_negative_int(self.n_censored, "n_censored"))
        if self.n_events <= 0:
            raise ValueError("n_events must be positive")
        object.__setattr__(self, "method", _require_method(self.method))
