import pytest

from momentum_forecast.fitting import rolling_ar1_phi, rolling_mean_drift, rolling_return_variance


def test_rolling_mean_drift_alignment_uses_first_window_exactly():
    fits = rolling_mean_drift([1.0, 2.0, 100.0], window=2)
    assert len(fits) == 2
    assert fits[0].drift == pytest.approx(1.5)
    assert fits[1].drift == pytest.approx(51.0)


def test_rolling_return_variance_alignment_uses_first_window_exactly():
    fits = rolling_return_variance([1.0, 2.0, 100.0], window=2)
    assert fits[0].variance == pytest.approx(0.5)


def test_rolling_ar1_phi_alignment_uses_first_window_exactly():
    fits = rolling_ar1_phi([1.0, 0.5, 0.25, 100.0], window=3)
    assert fits[0].phi == pytest.approx(0.5)


@pytest.mark.parametrize("window", [0, -1])
def test_rolling_rejects_invalid_window(window):
    with pytest.raises(ValueError):
        rolling_mean_drift([1.0, 2.0], window)


def test_rolling_rejects_too_little_data():
    with pytest.raises(ValueError):
        rolling_mean_drift([1.0], 2)
