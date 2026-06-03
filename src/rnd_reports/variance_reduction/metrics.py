"""Метрики снижения дисперсии для CUPED/CUPAC.

Извлечено из CUPAC-реализации VarWar (`autocupac.py`) без изменения поведения.
См. docs/variance_reduction_methodology.md.
"""

from __future__ import annotations

from .cuped import cuped_adjust, cuped_theta


def variance_reduction_pct(y, prediction) -> float:
    """Процент снижения дисперсии ``y`` после CUPED-поправки по ``prediction``.

    Возвращает ``max(0, (1 − Var(Ỹ)/Var(Y)) · 100)``; при вырожденном
    предсказании — ``0.0`` (как в исходном коде VarWar).
    """
    theta = cuped_theta(y, prediction)
    if theta == 0.0:
        return 0.0
    y_adj = cuped_adjust(y, prediction, theta=theta)
    return max(0.0, (1 - y_adj.var() / y.var()) * 100)
