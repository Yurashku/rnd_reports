"""Классическая CUPED-корректировка (θ-резидуализация целевой метрики).

Извлечено из CUPAC-реализации VarWar (`autocupac.py`) без изменения численного
поведения: используется линейная поправка ``Ỹ = Y − θ·(pred − E[pred])`` с
``θ = Cov(Y, pred)/Var(pred)``. См. docs/variance_reduction_methodology.md.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def cuped_theta(y, prediction) -> float:
    """Коэффициент θ для CUPED-поправки.

    Вырожденный случай (почти нулевая дисперсия предсказания) → θ=0.
    Численно эквивалентно исходному коду VarWar (cov с ddof=1, var с ddof=0).
    """
    pred = np.asarray(prediction, dtype=float)
    pred_centered = pred - pred.mean()
    var = pred_centered.var()
    if var < 1e-10:
        return 0.0
    return float(np.cov(y, pred_centered)[0, 1] / var)

def cuped_adjust(y, prediction, theta: Optional[float] = None):
    """Применить CUPED-поправку к ``y`` по предсказанию ``prediction``.

    Сохраняет тип ``y`` (для ``pd.Series`` вернётся ``pd.Series``).
    """
    pred = np.asarray(prediction, dtype=float)
    pred_centered = pred - pred.mean()
    if theta is None:
        theta = cuped_theta(y, prediction)
    return y - theta * pred_centered
