"""Диагностики безопасности признаков и balance/missingness gate (R&D-6, Step 6).

Проверяют согласованность заявленной роли признака с данными и реализуют
практический gate для класса C: безопасный in-time признак не должен быть
разбалансирован между treatment/control. Медиатор/leakage ожидаемо разбалансированы —
это наглядно показывает, почему их нельзя использовать.

См. docs/feature_safety_policy.md, docs/06_safe_intime_cupac_context.md §3C.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from . import rules
from .contracts import FeatureClass

# Роли (таксономия A–F), для которых дисбаланс по treatment — тревожный сигнал.

_SHOULD_BE_BALANCED = {
    FeatureClass.A_PRE_TREATMENT,
    FeatureClass.B_EXPERT_SAFE_INTIME,
    FeatureClass.C_BALANCE_GATED_INTIME,
}


def standardized_mean_diff(df: pd.DataFrame, treatment_col: str, feature: str) -> float:
    """Стандартизованная разность средних (SMD) признака между группами."""
    t = df.loc[df[treatment_col] == 1, feature]
    c = df.loc[df[treatment_col] == 0, feature]
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled_sd) or pooled_sd < 1e-12:
        return 0.0
    return float((t.mean() - c.mean()) / pooled_sd)


def missingness_diff(df: pd.DataFrame, treatment_col: str, feature: str) -> float:
    """|разность долей пропусков признака между treatment/control|."""
    t = df.loc[df[treatment_col] == 1, feature]
    c = df.loc[df[treatment_col] == 0, feature]
    return float(abs(t.isna().mean() - c.isna().mean()))


def balance_row(df: pd.DataFrame, treatment_col: str, feature: str) -> dict:
    """SMD + Welch p-value + дисбаланс пропусков для одного признака."""
    t = df.loc[df[treatment_col] == 1, feature]
    c = df.loc[df[treatment_col] == 0, feature]
    _, p_value = stats.ttest_ind(t, c, equal_var=False, nan_policy="omit")
    smd = standardized_mean_diff(df, treatment_col, feature)
    miss = missingness_diff(df, treatment_col, feature)
    return {
        "feature": feature,
        "smd": smd,
        "p_value": float(p_value),
        "missingness_diff": miss,
    }


def balance_gate(
    df: pd.DataFrame,
    treatment_col: str,
    features: list[str],
    **thresholds,
) -> tuple[list[str], pd.DataFrame]:
    """Применить balance/missingness gate к набору признаков.

    Возвращает ``(passed, report)``: список прошедших признаков и таблицу
    диагностики с колонкой ``gate_pass``.
    """
    rows = []
    passed: list[str] = []
    for f in features:
        r = balance_row(df, treatment_col, f)
        ok = rules.passes_balance_gate(r["smd"], r["missingness_diff"], **thresholds)
        r["gate_pass"] = bool(ok)
        rows.append(r)
        if ok:
            passed.append(f)
    report = pd.DataFrame(rows, columns=["feature", "smd", "p_value", "missingness_diff", "gate_pass"])
    return passed, report


def diagnose(
    df: pd.DataFrame,
    treatment_col: str,
    registry=None,
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Таблица диагностики по признакам реестра/списка.

    Колонки: ``feature, feature_class, smd, p_value, missingness_diff, gate_pass,
    role_consistent``. ``role_consistent=False`` — признак, который по политике должен
    быть сбалансирован, но разбалансирован (или наоборот).
    """
    if features is None:
        if registry is None:
            raise ValueError("Нужен либо registry, либо явный список features")
        features = registry.names()

    classes = registry.classes() if registry is not None else {}
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        r = balance_row(df, treatment_col, f)
        gate_ok = rules.passes_balance_gate(r["smd"], r["missingness_diff"])
        fc = classes.get(f)
        if fc in _SHOULD_BE_BALANCED:
            role_consistent = gate_ok
        else:
            role_consistent = True  # для risky/mediator/leakage баланс не требуется
        rows.append(
            {
                "feature": f,
                "feature_class": str(fc) if fc is not None else "",
                "smd": round(r["smd"], 4),
                "p_value": round(r["p_value"], 4),
                "missingness_diff": round(r["missingness_diff"], 4),
                "gate_pass": bool(gate_ok),
                "role_consistent": bool(role_consistent),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "feature_class",
            "smd",
            "p_value",
            "missingness_diff",
            "gate_pass",
            "role_consistent",
        ],
    )
