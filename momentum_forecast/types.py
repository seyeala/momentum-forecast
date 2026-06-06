"""Shared public result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from momentum_forecast._validation import require_finite, require_positive, validate_side


@dataclass(frozen=True)
class MomentumForecast:
    """Container for a single momentum forecast result."""

    model: str
    side: str
    horizon: float
    expected_return: float
    cost: float
    net_expected_return: float
    probability_profit: float | None = None
    z_score: float | None = None
    expected_life: float | None = None
    survival_probability: float | None = None
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        object.__setattr__(self, "side", validate_side(self.side))
        object.__setattr__(self, "horizon", require_positive(self.horizon, "horizon"))
        object.__setattr__(self, "expected_return", require_finite(self.expected_return, "expected_return"))
        object.__setattr__(self, "cost", require_finite(self.cost, "cost"))
        object.__setattr__(self, "net_expected_return", require_finite(self.net_expected_return, "net_expected_return"))

        for field_name in ("probability_profit", "z_score", "expected_life", "survival_probability"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_finite(value, field_name))

        diagnostics = {
            str(key): require_finite(value, f"diagnostics[{key!r}]") for key, value in self.diagnostics.items()
        }
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))
