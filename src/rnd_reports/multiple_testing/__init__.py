"""R&D-8: сравнение методов множественного тестирования в OR-регионе A/B-пилота.

Таблица контракта ``id, treatment, target_1, ..., target_n``: каждая target-метрика —
отдельная гипотеза о treatment effect, всё семейство обслуживает один OR-claim
(«есть ли хотя бы один надёжный сигнал и какие именно метрики его дали»).

Сравниваемые методы (см. :mod:`~rnd_reports.multiple_testing.methods`):
Bonferroni, Holm, Westfall–Young (maxT), Romano–Wolf (stepdown) — контроль FWER;
Benjamini–Hochberg — единственный FDR-ориентир. Реализация максимально опирается на
готовые решения scipy/statsmodels; свой код — только Romano–Wolf stepdown.

Два уровня сравнения:

- :func:`run_comparison` — single-table: adjusted p-value и reject-флаги по всем методам
  на ОДНОЙ таблице (синтетике или своей реальной); при известной правде — TP/FP/FDR/power;
- :func:`operating_characteristics` — Monte-Carlo: эмпирический FWER и power по методам на
  сетке сценариев (главный — зависимость power от корреляции метрик ``rho``).
"""

from __future__ import annotations

from .elementary import (
    compute_elementary_tests,
    compute_vectorized_t_stats,
    infer_target_columns,
    validate_input_table,
)
from .methods import (
    METHOD_ERROR,
    METHOD_NAMES,
    add_corrections,
    p_adj_columns,
    reject_columns,
    romano_wolf_stepdown_adjusted_pvalues,
    westfall_young_maxT_adjusted_pvalues,
)
from .pipeline import (
    ComparisonResult,
    evaluate_against_truth,
    run_comparison,
    summarize_pvalue_profile,
    summarize_rejections,
)
from .simulation import Scenario, operating_characteristics, rho_grid
from .synthetic import make_ab_table, make_equicorrelation_cov

__all__ = [
    # contract & elementary tests
    "validate_input_table",
    "infer_target_columns",
    "compute_elementary_tests",
    "compute_vectorized_t_stats",
    # methods
    "METHOD_NAMES",
    "METHOD_ERROR",
    "add_corrections",
    "westfall_young_maxT_adjusted_pvalues",
    "romano_wolf_stepdown_adjusted_pvalues",
    "p_adj_columns",
    "reject_columns",
    # single-table comparison
    "ComparisonResult",
    "run_comparison",
    "summarize_rejections",
    "evaluate_against_truth",
    "summarize_pvalue_profile",
    # synthetic
    "make_ab_table",
    "make_equicorrelation_cov",
    # Monte-Carlo operating characteristics
    "Scenario",
    "operating_characteristics",
    "rho_grid",
]
