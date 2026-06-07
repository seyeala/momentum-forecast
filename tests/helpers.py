"""Deterministic synthetic data helpers for fitting tests."""

from __future__ import annotations

import random


def make_gaussian_returns(mu: float, sigma: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def make_linear_log_prices(slope: float, sigma: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [slope * index + rng.gauss(0.0, sigma) for index in range(n)]


def make_ar1_returns(phi: float, innovation_sigma: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    values = [rng.gauss(0.0, innovation_sigma)]
    for _ in range(1, n):
        values.append(phi * values[-1] + rng.gauss(0.0, innovation_sigma))
    return values


def make_exponential_lifetimes(hazard: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.expovariate(hazard) for _ in range(n)]
