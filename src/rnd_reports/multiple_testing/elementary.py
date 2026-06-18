"""Контракт A/B-таблицы и элементарные тесты по каждой target-метрике.

Таблица имеет формат ``id, treatment, target_1, ..., target_n``: каждая ``target_*``
трактуется как отдельная гипотеза о treatment effect (разница средних treatment − control).

Элементарный тест — двусторонний Welch t-test через готовый
:func:`scipy.stats.ttest_ind` (``equal_var=False``); доверительный интервал берётся
из ``TtestResult.confidence_interval``. Своя формула не нужна — это ровно то, что
scipy уже реализует.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def infer_target_columns(
    df: pd.DataFrame,
    id_col: str = "id",
    treatment_col: str = "treatment",
    target_cols: Optional[List[str]] = None,
) -> List[str]:
    """Определить target-колонки по контракту таблицы (всё, кроме id/treatment)."""
    if target_cols is not None:
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"В таблице нет target-колонок: {missing}")
        return list(target_cols)

    excluded = {id_col, treatment_col}
    inferred = [c for c in df.columns if c not in excluded]
    if not inferred:
        raise ValueError("Не найдено ни одной target-колонки")
    return inferred


def validate_input_table(
    df: pd.DataFrame,
    id_col: str = "id",
    treatment_col: str = "treatment",
    target_cols: Optional[List[str]] = None,
) -> List[str]:
    """Проверить минимальные требования к таблице и вернуть список target-колонок."""
    for col in (id_col, treatment_col):
        if col not in df.columns:
            raise ValueError(f"Нет обязательной колонки: {col}")

    targets = infer_target_columns(df, id_col=id_col, treatment_col=treatment_col, target_cols=target_cols)

    unique_treatments = set(pd.Series(df[treatment_col]).dropna().unique())
    if not unique_treatments.issubset({0, 1}):
        raise ValueError(f"{treatment_col} должен содержать только 0/1; найдено {sorted(unique_treatments)}")

    non_numeric = [c for c in targets if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Все target-колонки должны быть числовыми; не числовые: {non_numeric}")

    group_sizes = df[treatment_col].value_counts(dropna=False).to_dict()
    if 0 not in group_sizes or 1 not in group_sizes:
        raise ValueError("В таблице должны присутствовать обе группы: control=0 и treatment=1")

    return targets


def compute_elementary_tests(
    df: pd.DataFrame,
    target_cols: List[str],
    treatment_col: str = "treatment",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Welch t-test (treatment − control) по каждой target-метрике через scipy.

    Возвращает таблицу с колонками ``target, effect, std_error, t_stat, p_value,
    ci_low, ci_high`` (отсортированной как пришли target-колонки).
    """
    mask_t = df[treatment_col].to_numpy() == 1
    rows = []
    for target in target_cols:
        x1 = df.loc[mask_t, target].to_numpy(dtype=float)
        x0 = df.loc[~mask_t, target].to_numpy(dtype=float)
        # Готовый Welch t-test scipy; nan_policy='omit' выбрасывает пропуски.
        res = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
        ci = res.confidence_interval(confidence_level=1 - alpha)
        effect = float(np.nanmean(x1) - np.nanmean(x0))
        t_stat = float(res.statistic)
        # std_error восстанавливаем из определения t = effect / se.
        se = float(effect / t_stat) if np.isfinite(t_stat) and t_stat != 0.0 else float("nan")
        rows.append(
            {
                "target": target,
                "effect": effect,
                "std_error": se,
                "t_stat": t_stat,
                "p_value": float(res.pvalue),
                "ci_low": float(ci.low),
                "ci_high": float(ci.high),
            }
        )
    return pd.DataFrame(rows)


def compute_vectorized_t_stats(y: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Векторные Welch t-статистики по всем колонкам ``y`` для бинарного ``treatment``.

    Используется внутри перестановочного цикла (Westfall–Young / Romano–Wolf): тысячи
    перестановок × n метрик, поэтому поячейный вызов scipy в цикле слишком медленный —
    здесь это осознанная векторная оптимизация той же Welch-статистики.
    """
    treatment = np.asarray(treatment).astype(int)
    mask_t = treatment == 1
    y_t = y[mask_t]
    y_c = y[~mask_t]
    n_t, n_c = y_t.shape[0], y_c.shape[0]
    if n_t < 2 or n_c < 2:
        raise ValueError("В обеих группах должно быть не меньше двух наблюдений")

    mean_t = np.nanmean(y_t, axis=0)
    mean_c = np.nanmean(y_c, axis=0)
    var_t = np.nanvar(y_t, axis=0, ddof=1)
    var_c = np.nanvar(y_c, axis=0, ddof=1)
    se = np.sqrt(var_t / n_t + var_c / n_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mean_t - mean_c) / se
    return np.where(np.isfinite(t), t, 0.0)
