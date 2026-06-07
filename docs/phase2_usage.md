# Phase 2 Usage: Fitting Formula Inputs

Phase 2 fitting utilities estimate the parameters consumed by the deterministic
Phase 1 formulas.  They do **not** fetch market data, choose trades, optimize
thresholds, calibrate probabilities, or backtest strategies.

## Data and unit assumptions

- Inputs are in chronological order.
- Price examples use adjusted, positive price levels converted to log prices.
- Return examples use per-bar log returns.
- Drift, volatility, variance, costs, and horizons must use compatible per-bar
  units.
- Fitted parameters are in-sample estimates unless evaluated with a separate
  out-of-sample or walk-forward process.
- Rolling helpers are endpoint-aligned and use only observations in the current
  window.

## Fit mean drift and return volatility

```python
from momentum_forecast.fitting import fit_mean_drift, fit_return_volatility

returns = [0.001, 0.002, -0.001, 0.003]
drift_fit = fit_mean_drift(returns)
volatility_fit = fit_return_volatility(returns)
```

## Fit EWMA volatility

```python
from momentum_forecast.fitting import fit_ewma_volatility

returns = [0.001, -0.002, 0.003]
ewma_fit = fit_ewma_volatility(returns, alpha=0.3)
```

The EWMA estimator initializes variance with the first squared return, then
updates with `alpha * return_t**2 + (1 - alpha) * variance_{t-1}`.

## Fit an OLS log-price drift

```python
from momentum_forecast.data import log_prices
from momentum_forecast.fitting import fit_ols_line_drift

prices = [100.0, 101.0, 102.5, 104.0]
log_price_values = log_prices(prices)
drift_fit = fit_ols_line_drift(log_price_values)
```

## Fit AR(1) persistence and half-life

```python
from momentum_forecast.fitting import fit_ar1_phi

returns = [0.01, 0.006, 0.004, 0.002, 0.001]
ar1_fit = fit_ar1_phi(returns)
phi = ar1_fit.phi
half_life = ar1_fit.half_life
```

## Fit a log-price channel

```python
from momentum_forecast.data import log_prices
from momentum_forecast.fitting import fit_line_residuals

log_price_values = log_prices([100.0, 101.0, 102.0, 101.5, 103.0])
channel_fit = fit_line_residuals(log_price_values)
```

By default, channel fitting uses an OLS line and sets the symmetric band to
`2 * residual_volatility`. A perfect line has zero residual volatility and an
infinite expected channel life.

## Fit a constant hazard

```python
from momentum_forecast.fitting import fit_constant_hazard, fit_constant_hazard_with_censoring

complete_fit = fit_constant_hazard([5.0, 7.0, 4.0, 6.0])
censored_fit = fit_constant_hazard_with_censoring([5.0, 7.0, 4.0, 6.0], [True, False, True, True])
```

## Bridge fitted values into Phase 1 formulas

```python
from momentum_forecast.fitting import fit_estimated_drift_inputs, estimated_drift_probability_from_fits

returns = [0.001, 0.002, -0.001, 0.003, 0.002]
drift_fit, volatility_fit = fit_estimated_drift_inputs(returns)
probability = estimated_drift_probability_from_fits(
    drift_fit,
    volatility_fit,
    horizon=5,
    cost=0.0005,
    side="long",
)
```

For walk-forward validation, strategy backtesting, and probability calibration,
use a later validation/backtesting phase rather than these fitting helpers.
