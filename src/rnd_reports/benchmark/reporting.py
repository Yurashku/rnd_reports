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


def plot_ate_ci(results: list[MethodResult], ax=None, true_ate: float | None = None):
    """График ATE ± доверительный интервал по всем методам (обязательная виз. §11).

    unsafe_demo выделяется цветом/маркером, чтобы визуально отделить демонстрацию
    от кандидатных методов. Требует matplotlib (импорт ленивый).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.6 * len(results) + 1.5))

    ys = list(range(len(results)))
    for y, r in zip(ys, results):
        ate = r.ate
        if ate is None:
            continue
        lo = r.ci_low if r.ci_low is not None else ate
        hi = r.ci_high if r.ci_high is not None else ate
        is_demo = r.safety_status == "unsafe_demo"
        color = "tab:red" if is_demo else ("tab:gray" if r.safety_status == "reference_only" else "tab:blue")
        ax.errorbar(
            ate, y, xerr=[[ate - lo], [hi - ate]], fmt="o", color=color,
            capsize=4, label=None,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([r.method for r in results])
    ax.invert_yaxis()
    if true_ate is not None:
        ax.axvline(true_ate, color="green", linestyle="--", alpha=0.7, label=f"true ATE={true_ate:g}")
        ax.legend(loc="best")
    ax.set_xlabel("ATE (± 95% CI)")
    ax.set_title("ATE ± CI по методам (синий — кандидаты, серый — reference, красный — unsafe_demo)")
    ax.grid(axis="x", alpha=0.3)
    return ax


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
