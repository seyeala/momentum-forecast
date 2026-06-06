from dataclasses import FrozenInstanceError

import pytest

from momentum_forecast.costs import TradingCost


def test_trading_cost_total():
    cost = TradingCost(spread=0.0001, fees=0.00005, slippage=0.00002, impact=0.00003)
    assert cost.total == pytest.approx(0.0002)


def test_trading_cost_rejects_negative_components():
    with pytest.raises(ValueError):
        TradingCost(spread=-0.1)


def test_trading_cost_is_immutable():
    cost = TradingCost()
    with pytest.raises(FrozenInstanceError):
        cost.spread = 1.0
