"""Безопасная in-time коррекция метрики линейным second-stage (R&D-6, Step 5–6).

Поверх CUPAC-скорректированного исхода применяем линейную (OLS) коррекцию **только**
по допустимым in-time признакам: класс B (expert-safe) и класс C (прошедшие
balance/missingness gate). Снижает остаточную дисперсию без смещения — при условии,
что признаки безопасны (см. реестр/политику и diagnostics). Не использует treatment.

``adj = target − (ŷ − E[ŷ])``, где ``ŷ = OLS(target ~ safe_features)``.

См. docs/variance_reduction_methodology.md, docs/feature_safety_policy.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .metrics import variance_reduction_pct


def safe_intime_linear_adjustment(
    df: pd.DataFrame, target_col: str, safe_features: list[str]
):
    """Линейная second-stage коррекция ``target_col`` по safe in-time признакам.

    Возвращает ``(adjusted: pd.Series, info: dict)``. При пустом списке признаков
    возвращает исходный таргет без изменений.

    NaN-политика: fail-fast с понятным ``ValueError``, если в таргете или safe-признаках
    есть пропуски. Реальные датасеты могут содержать NaN — их обработку (импутация/
    исключение) добавим отдельно; пока не допускаем «молчаливого» поведения.
    """
    y = df[target_col].reset_index(drop=True)
    if not safe_features:
        return y.copy(), {"used_features": [], "variance_reduction": 0.0}

    if y.isna().any():
        raise ValueError(
            f"safe_intime_linear_adjustment: целевая колонка '{target_col}' содержит NaN"
        )
    na_cols = [c for c in safe_features if df[c].isna().any()]
    if na_cols:
        raise ValueError(
            f"safe_intime_linear_adjustment: safe-признаки содержат NaN: {na_cols}"
        )

    X = df[safe_features].reset_index(drop=True)
    model = LinearRegression().fit(X, y)
    pred = np.asarray(model.predict(X), dtype=float)
    adjusted = y - (pred - pred.mean())
    return adjusted, {
        "used_features": list(safe_features),
        "variance_reduction": variance_reduction_pct(y, pred),
    }
