"""Контракты результатов бенчмарка R&D-6 (единая таблица метрик).

Фундамент Step 1: только схема результата (§10 контекста) и dataclass под одну
строку. **Без оценочной логики** — расчёт метрик появится в последующих шагах.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Точный порядок колонок единой таблицы результатов (см. контекст §10).
RESULT_COLUMNS: list[str] = [
    "hypothesis_name",
    "method",
    "predecessor_method",
    "dataset_type",
    "dataset_name",
    "target",
    "n",
    "ate",
    "se",
    "p_value",
    "ci_low",
    "ci_high",
    "adjusted_target_variance",
    "variance_reduction_vs_ab_pct",
    "sample_size_reduction_vs_ab_pct",
    "variance_reduction_vs_sklearn_cupac_pct",
    "sample_size_reduction_vs_sklearn_cupac_pct",
    "incremental_variance_reduction_vs_predecessor_pct",
    "incremental_sample_size_reduction_vs_predecessor_pct",
    "feature_groups_used",
    "n_features_used",
    "safety_status",
    "diagnostic_notes",
]

# Допустимые значения safety_status.
SAFETY_OK = "ok"
SAFETY_REFERENCE = "reference_only"
SAFETY_UNSAFE_DEMO = "unsafe_demo"
SAFETY_UNAVAILABLE = "unavailable"
SAFETY_STATUSES = (SAFETY_OK, SAFETY_REFERENCE, SAFETY_UNSAFE_DEMO, SAFETY_UNAVAILABLE)


@dataclass
class MethodResult:
    """Одна строка таблицы результатов бенчмарка (контракт; без вычислений)."""

    method: str
    dataset_name: str
    dataset_type: str
    target: str = "target"
    hypothesis_name: str = ""
    predecessor_method: str | None = None
    n: int | None = None
    ate: float | None = None
    se: float | None = None
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    adjusted_target_variance: float | None = None
    variance_reduction_vs_ab_pct: float | None = None
    sample_size_reduction_vs_ab_pct: float | None = None
    variance_reduction_vs_sklearn_cupac_pct: float | None = None
    sample_size_reduction_vs_sklearn_cupac_pct: float | None = None
    incremental_variance_reduction_vs_predecessor_pct: float | None = None
    incremental_sample_size_reduction_vs_predecessor_pct: float | None = None
    feature_groups_used: list[str] = field(default_factory=list)
    n_features_used: int = 0
    safety_status: str = SAFETY_OK
    diagnostic_notes: str = ""

    def as_row(self) -> dict:
        """Представление строкой в порядке ``RESULT_COLUMNS``."""
        return {col: getattr(self, col) for col in RESULT_COLUMNS}
