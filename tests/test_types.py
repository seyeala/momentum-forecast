from dataclasses import FrozenInstanceError

import pytest

from momentum_forecast.types import MomentumForecast


def test_momentum_forecast_validates_and_freezes_diagnostics():
    source = {"z": 1.5}
    forecast = MomentumForecast(
        model="known_drift",
        side="long",
        horizon=5,
        expected_return=0.003,
        cost=0.001,
        net_expected_return=0.002,
        diagnostics=source,
    )
    source["z"] = 99.0
    assert forecast.diagnostics["z"] == 1.5
    with pytest.raises(TypeError):
        forecast.diagnostics["z"] = 2.0
    with pytest.raises(FrozenInstanceError):
        forecast.horizon = 10


@pytest.mark.parametrize("kwargs", [{"side": "flat"}, {"horizon": 0.0}, {"model": ""}])
def test_momentum_forecast_rejects_invalid_inputs(kwargs):
    base = dict(model="m", side="long", horizon=1.0, expected_return=0.0, cost=0.0, net_expected_return=0.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        MomentumForecast(**base)
