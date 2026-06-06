# Phase 2 Tasks: Statistical Parameter Fitting

Phase 2 turns the deterministic Phase 1 formula modules into empirically usable components by estimating their required parameters from historical data. The goal is to fit drift, variance, AR(1) persistence, channel residuals, and simple regime hazard parameters without introducing trading rules, probability calibration, or full backtesting yet.

## Scope

Phase 2 includes:

- Historical drift estimation from return windows and log-price lines.
- Historical volatility and variance estimation.
- AR(1) persistence fitting.
- Channel residual and band fitting.
- Constant-hazard fitting from historical line lifetimes.
- Lightweight rolling estimators that use only past observations.
- Synthetic-data tests that verify estimator recovery against known parameters.
- Documentation examples showing how fitted parameters feed Phase 1 formulas.

Phase 2 does not include strategy optimization, trade threshold learning, full walk-forward backtesting, probability calibration, survival machine learning, GARCH, HMMs, Hawkes processes, or execution-cost impact modeling. Those belong to later phases.

## Design Principles

- Keep formula modules and fitting modules separate.
- Every estimator must use only data passed to it; no estimator may reach forward in time.
- Prefer simple, transparent estimators before complex models.
- Return structured fit objects with diagnostics instead of raw floats where uncertainty matters.
- Tests should include synthetic data with known ground-truth parameters.
- Estimators should validate inputs and fail loudly on insufficient data.

## Task 1: Fitting Package Scaffold

Create the Phase 2 fitting package and tests.

Suggested files:

```text
momentum_forecast/
  fitting/
    __init__.py
    drift.py
    volatility.py
    ar1.py
    channel.py
    hazard.py
    rolling.py

tests/
  test_fit_drift.py
  test_fit_volatility.py
  test_fit_ar1.py
  test_fit_channel.py
  test_fit_hazard.py
  test_rolling_fits.py
```

Acceptance criteria:

- `momentum_forecast.fitting` imports successfully.
- Public fitting exports are available from `momentum_forecast.fitting`.
- Phase 1 model modules do not import Phase 2 fitting modules.
- Tests can be discovered by `pytest`.

## Task 2: Shared Fit Result Types

Add fit result dataclasses to `momentum_forecast/types.py` or a dedicated `momentum_forecast/fitting/types.py` module.

Suggested types:

```python
@dataclass(frozen=True)
class DriftFit:
    drift: float
    standard_error: float | None
    t_stat: float | None
    lookback: int
    method: str

@dataclass(frozen=True)
class VolatilityFit:
    variance: float
    volatility: float
    lookback: int
    method: str

@dataclass(frozen=True)
class AR1Fit:
    phi: float
    innovation_variance: float
    half_life: float | None
    standard_error: float | None
    t_stat: float | None
    lookback: int

@dataclass(frozen=True)
class ChannelFit:
    slope: float
    intercept: float
    residual_variance: float
    residual_volatility: float
    band: float
    expected_life: float
    lookback: int
    method: str

@dataclass(frozen=True)
class HazardFit:
    hazard: float
    expected_life: float
    n_events: int
    n_censored: int
    method: str
```

Acceptance criteria:

- Fit result objects are immutable.
- Required numeric fields are finite.
- `lookback`, `n_events`, and `n_censored` are validated where present.
- Fit result objects can be used directly as inputs to Phase 1 formula functions.

## Task 3: Mean Drift Estimator

Implement sample-mean drift fitting in `momentum_forecast/fitting/drift.py`.

Functions:

```python
fit_mean_drift(returns) -> DriftFit
```

Estimator:

```text
drift = mean(returns)
standard_error = sample_std(returns) / sqrt(n)
t_stat = drift / standard_error
```

Acceptance criteria:

- Reject empty inputs.
- Require at least two observations when standard error is requested.
- Use sample variance with `n - 1` degrees of freedom.
- Return `lookback == len(returns)`.
- Synthetic tests recover a known Gaussian drift within a tolerance that scales with sample size.

## Task 4: OLS Log-Price Line Drift Estimator

Implement line-slope fitting for log prices in `momentum_forecast/fitting/drift.py`.

Functions:

```python
fit_ols_line_drift(log_prices) -> DriftFit
```

Estimator:

```text
log_price_t = intercept + slope * t + residual_t
```

Acceptance criteria:

- Reject fewer than two log prices.
- Reject non-finite values.
- Return slope as drift per bar.
- Include a slope standard error and t-stat when enough observations are available.
- Synthetic tests recover a known linear log-price slope with Gaussian noise.

## Task 5: Return Variance and Volatility Estimator

Implement return noise fitting in `momentum_forecast/fitting/volatility.py`.

Functions:

```python
fit_return_variance(returns) -> VolatilityFit
fit_return_volatility(returns) -> VolatilityFit
```

Estimator:

```text
variance = sample variance of returns using n - 1 degrees of freedom
volatility = sqrt(variance)
```

Acceptance criteria:

- Require at least two observations.
- Reject non-finite returns.
- Return variance and volatility in per-bar return units.
- Synthetic tests recover known Gaussian variance within sampling tolerance.

## Task 6: EWMA Volatility Estimator

Implement exponentially weighted volatility fitting in `momentum_forecast/fitting/volatility.py`.

Functions:

```python
fit_ewma_volatility(returns, alpha: float) -> VolatilityFit
```

Estimator:

```text
variance_t = alpha * return_t^2 + (1 - alpha) * variance_{t-1}
```

Acceptance criteria:

- Require `0 < alpha <= 1`.
- Require at least one observation.
- Document initialization behavior.
- Return method metadata that includes the alpha value.
- Tests cover alpha boundary validation and simple hand-computable sequences.

## Task 7: AR(1) Parameter Fitting

Implement AR(1) persistence fitting in `momentum_forecast/fitting/ar1.py`.

Functions:

```python
fit_ar1_phi(returns) -> AR1Fit
```

Model:

```text
r[t + 1] = phi * r[t] + eta[t + 1]
```

Acceptance criteria:

- Require at least three returns.
- Estimate `phi` by ordinary least squares through the origin unless an intercept option is explicitly added.
- Estimate innovation variance from residuals.
- Populate half-life using the Phase 1 `ar1_half_life` formula.
- Include standard error and t-stat for `phi` when denominator variance is non-zero.
- Synthetic AR(1) tests recover a known `phi` within tolerance.

## Task 8: Channel Residual Fitting

Implement channel residual fitting in `momentum_forecast/fitting/channel.py`.

Functions:

```python
fit_line_residuals(log_prices) -> ChannelFit
```

Workflow:

```text
1. Fit OLS line to log prices.
2. Compute residuals around the fitted line.
3. Estimate residual variance and residual volatility.
4. Set a default channel band from residual volatility or residual quantile.
5. Compute expected channel life using the Phase 1 channel-life formula.
```

Acceptance criteria:

- Require at least three log prices.
- Reject non-finite values.
- Return slope, intercept, residual variance, residual volatility, band, and expected life.
- Default band rule is documented.
- Tests cover a perfect line, a noisy line, and invalid inputs.

## Task 9: Channel Band Estimators

Implement explicit channel band helpers in `momentum_forecast/fitting/channel.py`.

Functions:

```python
fit_channel_band_from_residual_quantile(residuals, quantile: float = 0.95) -> float
fit_channel_band_from_volatility(residual_volatility: float, multiplier: float = 2.0) -> float
```

Acceptance criteria:

- Quantile band uses absolute residuals.
- Require `0 < quantile < 1`.
- Require positive residual volatility for volatility-multiplier bands.
- Require positive multiplier.
- Tests cover deterministic residual arrays and invalid parameters.

## Task 10: Constant-Hazard Fitting

Implement simple hazard fitting in `momentum_forecast/fitting/hazard.py`.

Functions:

```python
fit_constant_hazard(lifetimes) -> HazardFit
fit_constant_hazard_with_censoring(lifetimes, observed) -> HazardFit
```

Estimators:

```text
complete observations: hazard = 1 / mean(lifetimes)
right-censored observations: hazard = number_of_events / total_time_at_risk
```

Acceptance criteria:

- Reject non-positive lifetimes.
- Require at least one observed event for hazard estimation.
- Validate that `lifetimes` and `observed` have equal lengths for censored fitting.
- Populate `n_events`, `n_censored`, and expected life.
- Synthetic exponential-lifetime tests recover known hazard within tolerance.

## Task 11: Rolling Window Fit Helpers

Implement lightweight rolling estimators in `momentum_forecast/fitting/rolling.py`.

Functions:

```python
rolling_mean_drift(returns, window: int)
rolling_return_variance(returns, window: int)
rolling_ar1_phi(returns, window: int)
```

Acceptance criteria:

- Require positive integer window sizes.
- Require enough data to produce at least one fit.
- Each rolling estimate uses only observations at or before the current endpoint.
- Output length and alignment are documented.
- Tests verify that the first rolling result uses exactly the first `window` observations.

## Task 12: Fit-to-Forecast Composition Helpers

Add small helper functions that connect Phase 2 fit outputs to Phase 1 formula modules without creating a strategy layer.

Suggested files:

```text
momentum_forecast/fitting/compose.py
```

Functions:

```python
fit_estimated_drift_inputs(returns) -> tuple[DriftFit, VolatilityFit]
estimated_drift_probability_from_fits(
    drift_fit: DriftFit,
    volatility_fit: VolatilityFit,
    horizon: float,
    cost: float,
    side: Literal["long", "short"] = "long",
) -> float
```

Acceptance criteria:

- Composition helpers do not decide whether to trade.
- Composition helpers do not optimize thresholds.
- Probability output matches direct calls to Phase 1 `estimated_drift_profit_probability`.
- Tests verify the bridge from fitted `drift` and `variance` to probability calculation.

## Task 13: Synthetic Data Test Utilities

Create deterministic synthetic data helpers for tests.

Suggested files:

```text
tests/helpers.py
```

Functions:

```python
make_gaussian_returns(mu: float, sigma: float, n: int, seed: int)
make_linear_log_prices(slope: float, sigma: float, n: int, seed: int)
make_ar1_returns(phi: float, innovation_sigma: float, n: int, seed: int)
make_exponential_lifetimes(hazard: float, n: int, seed: int)
```

Acceptance criteria:

- Helpers use deterministic random seeds.
- Helpers are used only in tests, not production code.
- Tests document tolerances to avoid brittle failures.

## Task 14: Phase 2 Documentation

Update `README.md` or add `docs/phase2_usage.md` with fitting examples.

Include examples for:

- Fitting mean drift from returns.
- Fitting return volatility.
- Fitting AR(1) persistence and half-life.
- Fitting a channel from log prices.
- Fitting a constant hazard from line lifetimes.
- Passing fitted values into Phase 1 probability formulas.

Acceptance criteria:

- Examples use small arrays or synthetic data.
- Documentation clearly states that fitted parameters are in-sample estimates unless evaluated out of sample.
- Documentation points to Phase 3 for walk-forward validation and backtesting.

## Task 15: Phase 2 Completion Checklist

Phase 2 is complete when:

- Fitting package scaffold exists.
- Drift, volatility, AR(1), channel, and hazard estimators are implemented.
- Rolling fit helpers are implemented.
- Fit-to-forecast composition helpers are implemented.
- Synthetic-data recovery tests pass.
- Documentation examples are present.
- No strategy threshold optimization, probability calibration, or full backtesting logic has leaked into Phase 2 modules.
