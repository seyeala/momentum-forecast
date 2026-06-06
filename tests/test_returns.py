import math

import pytest

from momentum_forecast.data import bps_to_log_cost, log_prices, log_returns, simple_to_log_return


def test_log_prices_scalar_and_sequence():
    assert log_prices(math.e) == pytest.approx(1.0)
    assert log_prices([1.0, math.e, math.e**2]) == pytest.approx([0.0, 1.0, 2.0])


@pytest.mark.parametrize("prices", [[1.0, 0.0], [-1.0], 0.0])
def test_log_prices_reject_non_positive(prices):
    with pytest.raises(ValueError):
        log_prices(prices)


def test_log_returns_returns_n_minus_one_values():
    assert log_returns([100.0, 105.0, 110.25]) == pytest.approx([math.log(1.05), math.log(1.05)])


def test_log_returns_require_two_prices():
    with pytest.raises(ValueError):
        log_returns([100.0])


def test_simple_to_log_return():
    assert simple_to_log_return(0.05) == pytest.approx(math.log(1.05))
    with pytest.raises(ValueError):
        simple_to_log_return(-1.0)


def test_bps_to_log_cost_is_in_return_units():
    assert bps_to_log_cost(2.0) == pytest.approx(0.0002, rel=1e-4)
    with pytest.raises(ValueError):
        bps_to_log_cost(-1.0)
