"""Тесты CUPED θ-корректировки (ядро, извлечённое из VarWar)."""

from __future__ import annotations

import numpy as np

from rnd_reports.variance_reduction.cuped import cuped_adjust, cuped_theta
from rnd_reports.variance_reduction.metrics import variance_reduction_pct


def _correlated_xy(n: int = 2000, rho: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    cov = [[1.0, rho], [rho, 1.0]]
    xy = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    import pandas as pd

    return pd.Series(xy[:, 0]), pd.Series(xy[:, 1])  # y, prediction


def test_cuped_reduces_variance_for_correlated_prediction() -> None:
    y, pred = _correlated_xy()
    y_adj = cuped_adjust(y, pred)
    assert y_adj.var() < y.var()


def test_variance_reduction_pct_positive_and_bounded() -> None:
    y, pred = _correlated_xy(rho=0.8)
    vr = variance_reduction_pct(y, pred)
    assert 0.0 <= vr <= 100.0
    # при rho≈0.8 ожидаем заметное снижение (R²≈0.64 → ~64%)
    assert vr > 30.0


def test_degenerate_prediction_gives_zero() -> None:
    import pandas as pd

    y = pd.Series(np.arange(100, dtype=float))
    const_pred = np.ones(100)
    assert cuped_theta(y, const_pred) == 0.0
    assert variance_reduction_pct(y, const_pred) == 0.0
