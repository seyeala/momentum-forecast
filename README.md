# Momentum Forecast

Momentum Forecast is a small Python package for the deterministic formula layer of momentum-line forecasting.  The first phase focuses on log-return units, cost accounting, Gaussian continuation probabilities, signal-to-noise scores, Brownian channel lifetime, constant-hazard regime lifetime, and AR(1) persistence.

The package intentionally separates **formula modules** from later **statistical fitting**, walk-forward validation, and strategy backtesting.

## Install for local development

```bash
python -m pip install -e .
python -m pytest
```

## Log prices and log returns

Use log-prices and log returns so slopes, volatility, and trading costs are all in compatible return units.

```python
from momentum_forecast import log_returns

returns = log_returns([100.0, 101.0, 102.01])
# approximately [0.00995, 0.00995]
```

## Trading costs

Costs are represented as non-negative return-unit components.

```python
from momentum_forecast import TradingCost, bps_to_log_cost

cost = TradingCost(
    spread=bps_to_log_cost(1.0),
    fees=bps_to_log_cost(0.5),
    slippage=bps_to_log_cost(0.25),
    impact=bps_to_log_cost(0.25),
)

total_cost = cost.total
```

## Known-drift Gaussian probability

When drift and volatility are supplied directly, compute the probability that the directional return beats cost over a horizon.

```python
from momentum_forecast import known_drift_profit_probability

probability = known_drift_profit_probability(
    drift=0.0005,
    volatility=0.002,
    horizon=10,
    cost=0.0002,
    side="long",
)
```

## Estimated-drift Gaussian probability

When drift is estimated from a lookback window, use the estimated-drift formula.  It includes the slope-uncertainty penalty `1 + horizon / lookback`.

```python
from momentum_forecast import estimated_drift_profit_probability

probability = estimated_drift_profit_probability(
    mu_hat=0.0005,
    q_hat=0.000004,
    lookback=20,
    horizon=10,
    cost=0.0002,
    side="long",
)
```

## Momentum signal-to-noise score

```python
from momentum_forecast import classify_momentum_strength, momentum_z_score

z = momentum_z_score(mu_hat=0.0005, q_hat=0.000004, lookback=20)
strength = classify_momentum_strength(z)
```

## Channel and regime lifetime

```python
from momentum_forecast import expected_channel_life, expected_regime_life

channel_life = expected_channel_life(band=0.01, residual_volatility=0.002)
regime_life = expected_regime_life(hazard=0.05)
```

## AR(1) persistence

```python
from momentum_forecast import ar1_expected_cumulative_return, ar1_half_life

expected_return = ar1_expected_cumulative_return(current_return=0.001, phi=0.6, horizon=5)
half_life = ar1_half_life(0.6)
```
