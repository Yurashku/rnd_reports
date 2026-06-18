"""Сравнение методов на одной таблице (single-table) + оценка против известной правды.

:func:`run_comparison` работает на ЛЮБОЙ таблице контракта ``id, treatment, target_*``:
синтетике или своей реальной. Если ``true_effects`` известны (синтетика) — добавляет
оценку TP/FP/FDR/power по каждому методу; на реальных данных правда неизвестна, и этот
блок пропускается.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .elementary import compute_elementary_tests, validate_input_table
from .methods import METHOD_ERROR, METHOD_NAMES, add_corrections, p_adj_columns, reject_columns


@dataclass
class ComparisonResult:
    """Результат сравнения методов на одной таблице."""

    table: pd.DataFrame              # target × (effect, p_value, p_adj_*, reject_*)
    method_summary: pd.DataFrame     # method × (error, n_rejected)
    truth_evaluation: Optional[pd.DataFrame]  # синтетика: method × (TP/FP/FDR/power)
    pvalue_profile: pd.DataFrame     # method × (mean_p_adj_signal/null, conservativeness_gap)
    target_cols: List[str]


def summarize_rejections(table: pd.DataFrame) -> pd.DataFrame:
    """Сводка «сколько target-метрик отвергнуто каждым методом»."""
    rows = []
    for name, col in reject_columns().items():
        rows.append(
            {
                "method": name,
                "error_control": METHOD_ERROR[name],
                "n_rejected": int(table[col].sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_against_truth(
    table: pd.DataFrame,
    true_effects: Dict[str, float],
    effect_tol: float = 1e-12,
) -> pd.DataFrame:
    """Оценка методов против известной правды: TP/FP, realized FDR, power, FWER-событие."""
    truth = table["target"].map(lambda t: abs(float(true_effects.get(t, 0.0))) > effect_tol).to_numpy()
    n_non_null = int(truth.sum())

    rows = []
    for name, col in reject_columns().items():
        reject = table[col].to_numpy(dtype=bool)
        tp = int(np.sum(reject & truth))
        fp = int(np.sum(reject & ~truth))
        discoveries = int(np.sum(reject))
        rows.append(
            {
                "method": name,
                "error_control": METHOD_ERROR[name],
                "true_positives": tp,
                "false_positives": fp,
                "realized_fdr": fp / discoveries if discoveries > 0 else 0.0,
                "power": tp / n_non_null if n_non_null > 0 else np.nan,
                "fwer_event": bool(fp >= 1),
            }
        )
    return pd.DataFrame(rows)


def summarize_pvalue_profile(
    table: pd.DataFrame,
    true_effects: Optional[Dict[str, float]] = None,
    effect_tol: float = 1e-12,
) -> pd.DataFrame:
    """Скалярные сводки методов «через p-values».

    По каждому методу из :data:`METHOD_NAMES` считает:

    - ``mean_p_adj_signal`` — средний adjusted p-value по истинно-ненулевым target
      (сила детекции; NaN, если правда неизвестна или нет ненулевых эффектов);
    - ``mean_p_adj_null`` — средний adjusted p-value по нулевым target (калибровка;
      NaN без правды);
    - ``conservativeness_gap`` — среднее ``p_adj - p_value`` по всем target (инфляция
      над raw p-value; считается всегда, работает и на реальной таблице).

    Колонка ``error_control`` (FWER/FDR) намеренно сохранена: значения FWER-методов
    сравнимы между собой напрямую, а Benjamini–Hochberg — q-value «другой валюты»
    ошибки, поэтому строки FWER и FDR читаются раздельно, а не пулятся.
    """
    raw = table["p_value"].to_numpy(dtype=float)
    if true_effects is not None:
        truth = table["target"].map(
            lambda t: abs(float(true_effects.get(t, 0.0))) > effect_tol
        ).to_numpy()
    else:
        truth = None

    rows = []
    for name, col in p_adj_columns().items():
        p_adj = table[col].to_numpy(dtype=float)
        if truth is not None:
            signal = p_adj[truth]
            null = p_adj[~truth]
            mean_signal = float(np.mean(signal)) if signal.size else float("nan")
            mean_null = float(np.mean(null)) if null.size else float("nan")
        else:
            mean_signal = float("nan")
            mean_null = float("nan")
        rows.append(
            {
                "method": name,
                "error_control": METHOD_ERROR[name],
                "mean_p_adj_signal": mean_signal,
                "mean_p_adj_null": mean_null,
                "conservativeness_gap": float(np.mean(p_adj - raw)),
            }
        )
    return pd.DataFrame(rows)


def run_comparison(
    df: pd.DataFrame,
    id_col: str = "id",
    treatment_col: str = "treatment",
    target_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    random_state: int = 42,
    true_effects: Optional[Dict[str, float]] = None,
) -> ComparisonResult:
    """Прогнать все методы на одной таблице и собрать сравнительные артефакты."""
    targets = validate_input_table(df, id_col=id_col, treatment_col=treatment_col, target_cols=target_cols)
    tests = compute_elementary_tests(df, targets, treatment_col=treatment_col, alpha=alpha)
    corrected = add_corrections(
        tests, df, targets, treatment_col=treatment_col, alpha=alpha,
        n_resamples=n_resamples, random_state=random_state,
    )

    # Компактная таблица для просмотра глазами: эффект с ДИ + adj p-value + reject-флаги.
    # ci_low/ci_high из Welch-теста несём дальше — нужны для forest-плота эффектов.
    cols = ["target", "effect", "ci_low", "ci_high", "p_value"]
    for name in METHOD_NAMES:
        cols += [f"p_adj_{name}", f"reject_{name}"]
    table = corrected[cols].sort_values("p_value").reset_index(drop=True)

    method_summary = summarize_rejections(corrected)
    truth_eval = evaluate_against_truth(corrected, true_effects) if true_effects is not None else None
    pvalue_profile = summarize_pvalue_profile(corrected, true_effects)

    return ComparisonResult(
        table=table,
        method_summary=method_summary,
        truth_evaluation=truth_eval,
        pvalue_profile=pvalue_profile,
        target_cols=targets,
    )


# Реэкспорт для удобства витрины.
__all__ = [
    "ComparisonResult",
    "run_comparison",
    "summarize_rejections",
    "evaluate_against_truth",
    "summarize_pvalue_profile",
    "p_adj_columns",
    "reject_columns",
]
