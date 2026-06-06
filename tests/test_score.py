import pytest

from momentum_forecast.signals import classify_momentum_strength, momentum_z_score


def test_momentum_z_score_formula():
    assert momentum_z_score(0.002, 0.0001, 25) == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs", [{"q_hat": 0.0}, {"lookback": 0}])
def test_momentum_z_score_rejects_invalid_inputs(kwargs):
    base = dict(mu_hat=0.0, q_hat=0.0001, lookback=10)
    base.update(kwargs)
    with pytest.raises(ValueError):
        momentum_z_score(**base)


@pytest.mark.parametrize(
    ("z", "strength"),
    [
        (0.999, "weak"),
        (1.0, "moderate"),
        (-1.5, "moderate"),
        (2.0, "strong"),
        (-2.5, "strong"),
        (3.0, "extreme"),
        (-4.0, "extreme"),
    ],
)
def test_classify_momentum_strength_boundaries(z, strength):
    assert classify_momentum_strength(z) == strength
