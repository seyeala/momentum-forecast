"""Trading cost representation in log-return-compatible units."""

from __future__ import annotations

from dataclasses import dataclass

from momentum_forecast._validation import require_non_negative


@dataclass(frozen=True)
class TradingCost:
    """Immutable decomposition of per-trade costs."""

    spread: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    impact: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("spread", "fees", "slippage", "impact"):
            object.__setattr__(self, field_name, require_non_negative(getattr(self, field_name), field_name))

    @property
    def total(self) -> float:
        """Total cost in return units."""
        return self.spread + self.fees + self.slippage + self.impact
