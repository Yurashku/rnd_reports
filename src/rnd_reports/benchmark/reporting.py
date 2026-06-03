"""Сборка результатов бенчмарка R&D-6 в таблицу (Step 3).

Превращает список ``MethodResult`` в ``pandas.DataFrame`` в порядке колонок
``RESULT_COLUMNS`` (§10 контекста) и форматирует числовые поля для отображения.
График ATE±CI и markdown-отчёт — последующие шаги.
"""

from __future__ import annotations

import pandas as pd

from .contracts import RESULT_COLUMNS, MethodResult

# Числовые колонки и округление при показе.
_ROUND = {
    "ate": 4,
    "se": 4,
    "p_value": 4,
    "ci_low": 4,
    "ci_high": 4,
    "adjusted_target_variance": 4,
    "variance_reduction_vs_ab_pct": 2,
    "sample_size_reduction_vs_ab_pct": 2,
    "variance_reduction_vs_sklearn_cupac_pct": 2,
    "sample_size_reduction_vs_sklearn_cupac_pct": 2,
    "incremental_variance_reduction_vs_predecessor_pct": 2,
    "incremental_sample_size_reduction_vs_predecessor_pct": 2,
}


def results_to_frame(results: list[MethodResult]) -> pd.DataFrame:
    """Список результатов → DataFrame в порядке ``RESULT_COLUMNS``."""
    return pd.DataFrame([r.as_row() for r in results], columns=RESULT_COLUMNS)


def format_results_table(results_or_frame) -> pd.DataFrame:
    """Округлить числовые колонки и привести список признаков к строке."""
    if isinstance(results_or_frame, pd.DataFrame):
        out = results_or_frame.copy()
    else:
        out = results_to_frame(results_or_frame)
    for col, ndigits in _ROUND.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(ndigits)
    if "feature_groups_used" in out.columns:
        out["feature_groups_used"] = out["feature_groups_used"].apply(
            lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v
        )
    return out
