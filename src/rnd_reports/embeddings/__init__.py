"""Тулкит-адаптеры R&D-7: эмбеддинги клиента как adjustment set.

Адаптеры-«переходники» поверх pyspark-датасета со схемой
``epk_id, report_dt, emb_0_val, emb_1_val, ..., emb_n_val`` (поддерживается и
легаси-формат ``col_000, ...``):

- :class:`EmbeddingReducer` — снижает размерность эмбеддингов до ``red_size``
  (StandardScaler + PCA), отдаёт ``[epk_id, report_dt, red_0, ...]``;
- :class:`PropensityScorer` — джойнит с таблицей псевдо-трита и сводит эмбеддинги к
  одному ``prop_score`` = P(treatment=1); из LogisticRegression / GBTClassifier берётся
  модель с лучшим ROC-AUC (инженерная selection-эвристика, не causal-критерий качества).

Оба адаптера поддерживают **in-time safety**: обучение только на ``report_dt <= cutoff``
и применение к более поздним срезам (без утечки из будущего).

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
    validate_treatment_schema,
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
    "validate_treatment_schema",
    "EmbeddingReducer",
    "PropensityScorer",
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
    # Ленивая загрузка Spark-зависимых адаптеров: ``import rnd_reports.embeddings``
    # не должен требовать установленного pyspark (см. test_rnd7_imports).
    if name == "EmbeddingReducer":
        from .reducer import EmbeddingReducer

        return EmbeddingReducer
    if name == "PropensityScorer":
        from .propensity import PropensityScorer

        return PropensityScorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
