"""Тулкит-функции R&D-7: эмбеддинги клиента как adjustment set.

Функции-«переходники» поверх pyspark-датасета со схемой
``epk_id, report_dt, emb_0_val, emb_1_val, ..., emb_n_val`` (поддерживается и
легаси-формат ``col_000, ...``). Каждая принимает один ``DataFrame`` и возвращает его же
с **добавленными** колонками:

- :func:`reduce_embeddings` — снижает размерность эмбеддингов до ``red_size``
  (StandardScaler + PCA), добавляет колонки ``red_0, ..., red_{red_size-1}``;
- :func:`add_propensity_score` — сводит эмбеддинги к одному ``prop_score`` = P(treatment=1)
  и добавляет его; трит — опциональная колонка того же датафрейма, при отсутствии
  генерируется случайно (``random_state``). Из LogisticRegression / GBTClassifier берётся
  модель с лучшим ROC-AUC (инженерная selection-эвристика, не causal-критерий качества).

pyspark — **опциональная** зависимость (extra ``spark``). Импорт самого пакета её
не требует: контрактные хелперы доступны всегда, а Spark-адаптеры подгружаются лениво.
Сам causal-эксперимент R&D-7 реализован в numpy/sklearn-слое
(:mod:`rnd_reports.embeddings.experiment`, :mod:`rnd_reports.embeddings.synthetic`):
оценка ATE с поправкой и диагностика баланса/overlap — pyspark для него не нужен.
"""

from __future__ import annotations

from .contracts import (
    EMBEDDING_PREFIX,
    EPK_ID,
    KEY_COLUMNS,
    PROPENSITY_SCORE,
    REDUCED_PREFIX,
    REPORT_DT,
    TREATMENT,
    embedding_feature_columns,
    reduced_column_names,
    validate_embedding_schema,
)

__all__ = [
    "EPK_ID",
    "REPORT_DT",
    "TREATMENT",
    "KEY_COLUMNS",
    "EMBEDDING_PREFIX",
    "REDUCED_PREFIX",
    "PROPENSITY_SCORE",
    "embedding_feature_columns",
    "reduced_column_names",
    "validate_embedding_schema",
    "reduce_embeddings",
    "add_propensity_score",
    # causal-эксперимент (numpy/sklearn, без pyspark)
    "make_embedding_observational_scenario",
    "estimate_ate_with_adjustment",
    "covariate_balance_after_adjustment",
    "overlap_diagnostics",
    "evaluate_adjustment_set_quality",
    "fit_propensity",
]

# causal-слой не требует pyspark — импортируем сразу.
from .experiment import (  # noqa: E402
    covariate_balance_after_adjustment,
    estimate_ate_with_adjustment,
    evaluate_adjustment_set_quality,
    fit_propensity,
    overlap_diagnostics,
)
from .synthetic import make_embedding_observational_scenario  # noqa: E402


def __getattr__(name: str):
    # Ленивая загрузка Spark-зависимых функций: ``import rnd_reports.embeddings``
    # не должен требовать установленного pyspark (см. test_rnd7_imports).
    if name == "reduce_embeddings":
        from .reducer import reduce_embeddings

        return reduce_embeddings
    if name == "add_propensity_score":
        from .propensity import add_propensity_score

        return add_propensity_score
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
