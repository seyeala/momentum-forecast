# Phase 1 Tasks: Core Math and Data Structures

Phase 1 builds the deterministic foundation for the momentum-line forecasting library. The goal is to implement pure, testable formula modules and shared data structures before adding statistical fitting, walk-forward validation, or strategy optimization.

## Scope

Phase 1 includes:

- Return and log-price utilities.
- Trading cost representation.
- Shared forecast result types.
- Known-drift Gaussian profit probability.
- Estimated-drift Gaussian profit probability using a supplied drift estimate and variance.
- Momentum signal-to-noise scoring.
- Brownian channel-life formulas.
- Constant-hazard regime-life formulas.
- AR(1) momentum persistence formulas.
- Unit tests for formula correctness, edge cases, and monotonic behavior.

Phase 1 does not include historical parameter fitting, backtesting, probability calibration, empirical survival fitting, or strategy threshold optimization. Those belong to later phases.

## Task 1: Package Scaffold

Create the initial Python package layout.

Suggested files:

```text
momentum_forecast/
  __init__.py
  types.py
  data/
    __init__.py
    returns.py
  costs/
    __init__.py
    model.py
  models/
    __init__.py
    gaussian_drift.py
    estimated_drift.py
    channel_life.py
    hazard.py
    ar1.py
  signals/
    __init__.py
    score.py
tests/
  test_returns.py
  test_costs.py
  test_gaussian_drift.py
  test_estimated_drift.py
  test_channel_life.py
  test_hazard.py
  test_ar1.py
  test_score.py
pyproject.toml
README.md
```

Acceptance criteria:

- Package imports successfully with `python -c "import momentum_forecast"`.
- Test runner discovers all tests.
- Public exports are defined in `momentum_forecast/__init__.py`.

## Task 2: Return and Log-Price Utilities

Implement conversion utilities in `momentum_forecast/data/returns.py`.

Functions:

```python
log_prices(prices)
log_returns(prices)
simple_to_log_return(simple_return)
bps_to_log_cost(bps)
```

Acceptance criteria:

- Reject non-positive prices for log-price conversion.
- Return `n - 1` log returns for `n` prices.
- Convert 2 basis points to approximately `0.0002` in return units.
- Tests cover scalar and sequence inputs where appropriate.

## Task 3: Trading Cost Model

Implement a reusable cost object in `momentum_forecast/costs/model.py`.

Suggested type:

```python
@dataclass(frozen=True)
class TradingCost:
    spread: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    impact: float = 0.0

    @property
    def total(self) -> float:
        return self.spread + self.fees + self.slippage + self.impact
```

Acceptance criteria:

- Cost components must be non-negative.
- `total` returns the sum of all components.
- The object is immutable after creation.

## Task 4: Shared Forecast Result Types

Implement shared result objects in `momentum_forecast/types.py`.

Suggested type:

```python
@dataclass(frozen=True)
class MomentumForecast:
    model: str
    side: Literal["long", "short"]
    horizon: float
    expected_return: float
    cost: float
    net_expected_return: float
    probability_profit: float | None = None
    z_score: float | None = None
    expected_life: float | None = None
    survival_probability: float | None = None
    diagnostics: Mapping[str, float] = field(default_factory=dict)
```

Acceptance criteria:

- Validate `side` is either `long` or `short`.
- Validate `horizon` is positive.
- Store diagnostics without allowing accidental mutation of the forecast object.

## Task 5: Known-Drift Gaussian Model

Implement the known-drift Gaussian probability module in `momentum_forecast/models/gaussian_drift.py`.

Functions:

```python
known_drift_profit_probability(
    drift: float,
    volatility: float,
    horizon: float,
    cost: float,
    side: Literal["long", "short"] = "long",
) -> float
```

Formula for long trades:

```text
Phi((drift * horizon - cost) / (volatility * sqrt(horizon)))
```

Formula for short trades:

```text
Phi((-drift * horizon - cost) / (volatility * sqrt(horizon)))
```

Acceptance criteria:

- Return exactly `0.5` when expected gross return equals cost.
- Probability increases as long-side drift increases.
- Probability decreases as cost increases.
- Reject non-positive volatility and horizon.
- Reject negative cost.
- Reject unsupported side values.

## Task 6: Estimated-Drift Gaussian Model

Implement the estimated-drift model in `momentum_forecast/models/estimated_drift.py`.

Functions:

```python
estimated_drift_profit_probability(
    mu_hat: float,
    q_hat: float,
    lookback: int,
    horizon: float,
    cost: float,
    side: Literal["long", "short"] = "long",
) -> float
```

Formula for long trades:

```text
Phi((mu_hat * horizon - cost) / sqrt(q_hat * horizon * (1 + horizon / lookback)))
```

Formula for short trades:

```text
Phi((-mu_hat * horizon - cost) / sqrt(q_hat * horizon * (1 + horizon / lookback)))
```

Acceptance criteria:

- Return `0.5` when expected gross return equals cost.
- Include the slope-uncertainty penalty term `(1 + horizon / lookback)`.
- Reject non-positive `q_hat`, `lookback`, and `horizon`.
- Reject negative cost.
- Reject unsupported side values.

## Task 7: Momentum Signal-to-Noise Score

Implement signal scoring in `momentum_forecast/signals/score.py`.

Functions:

```python
momentum_z_score(mu_hat: float, q_hat: float, lookback: int) -> float
classify_momentum_strength(z: float) -> Literal["weak", "moderate", "strong", "extreme"]
```

Formula:

```text
mu_hat * sqrt(lookback) / sqrt(q_hat)
```

Default classification thresholds:

```text
abs(z) < 1.0       weak
1.0 <= abs(z) < 2 moderate
2.0 <= abs(z) < 3 strong
abs(z) >= 3       extreme
```

Acceptance criteria:

- Reject non-positive `q_hat` and `lookback`.
- Classification uses absolute z-score so long and short momentum are treated symmetrically.
- Tests cover all threshold boundaries.

## Task 8: Channel-Life Model

Implement Brownian channel-life formulas in `momentum_forecast/models/channel_life.py`.

Functions:

```python
expected_channel_life(band: float, residual_volatility: float) -> float
channel_survival_probability(
    horizon: float,
    band: float,
    residual_volatility: float,
    terms: int = 100,
) -> float
```

Expected life formula:

```text
band ** 2 / residual_volatility ** 2
```

Acceptance criteria:

- Reject non-positive band, residual volatility, horizon, and terms.
- Expected life equals `band ** 2 / residual_volatility ** 2`.
- Survival probability is constrained to `[0, 1]` after numeric computation.
- Survival probability decreases as horizon increases for fixed band and volatility.

## Task 9: Constant-Hazard Regime-Life Model

Implement regime hazard formulas in `momentum_forecast/models/hazard.py`.

Functions:

```python
regime_survival_probability(hazard: float, horizon: float) -> float
expected_regime_life(hazard: float) -> float
hazard_adjusted_drift_gain(drift: float, hazard: float, horizon: float) -> float
```

Formulas:

```text
survival = exp(-hazard * horizon)
expected_life = 1 / hazard
drift_gain = drift / hazard * (1 - exp(-hazard * horizon))
```

Acceptance criteria:

- Reject non-positive hazard.
- Reject negative horizon.
- Survival is `1.0` at horizon `0`.
- Expected life equals `1 / hazard`.
- Drift gain approaches `drift * horizon` for very small hazard.

## Task 10: AR(1) Momentum Persistence Model

Implement AR(1) formulas in `momentum_forecast/models/ar1.py`.

Functions:

```python
ar1_expected_return_at_horizon(current_return: float, phi: float, h: int) -> float
ar1_expected_cumulative_return(current_return: float, phi: float, horizon: int) -> float
ar1_half_life(phi: float) -> float | None
```

Formulas:

```text
E[r_{t+h} | r_t] = phi ** h * current_return
E[sum_{h=1}^H r_{t+h} | r_t] = current_return * phi * (1 - phi ** horizon) / (1 - phi), phi != 1
T_1/2 = log(0.5) / log(phi), 0 < phi < 1
```

Acceptance criteria:

- Reject non-positive horizon values.
- Handle `phi == 1` in cumulative return as `current_return * horizon`.
- Return `None` for half-life when `phi <= 0` or `phi >= 1`.
- Tests cover `phi = 0`, `0 < phi < 1`, `phi = 1`, and `phi > 1`.

## Task 11: Public API Exports

Expose stable v1 functions and types in package `__init__.py` files.

Acceptance criteria:

- Users can import core objects from `momentum_forecast`.
- Module-level imports remain lightweight.
- No imports are wrapped in try/except blocks.

## Task 12: Unit Test Suite

Create a pytest suite covering all Phase 1 formulas and validations.

Acceptance criteria:

- `pytest` passes.
- Formula tests use deterministic known values.
- Edge cases are explicit.
- Monotonicity tests are included for probability and survival functions.

## Task 13: Documentation

Update `README.md` with Phase 1 usage examples.

Include examples for:

- Converting prices to log returns.
- Constructing a `TradingCost`.
- Computing known-drift profit probability.
- Computing estimated-drift profit probability.
- Computing AR(1) half-life.

Acceptance criteria:

- Examples are short and runnable.
- README states that Phase 1 modules do not fit parameters from historical data.
- README points to Phase 2 for statistical fitting.

## Task 14: Phase 1 Completion Checklist

Phase 1 is complete when:

- Core package scaffold exists.
- Formula modules are implemented.
- Shared data types are implemented.
- Cost model is implemented.
- Unit tests pass.
- README examples are present.
- No statistical fitting or backtesting logic has leaked into Phase 1 modules.
