"""Скрипт 2 (R&D-7): propensity score (prop_score) из эмбеддингов.

``add_propensity_score`` обучает классификатор ``P(treatment=1 | эмбеддинги)`` и
**добавляет** к исходному датафрейму колонку ``prop_score`` (все исходные колонки
сохраняются). Если колонки ``treatment`` в датафрейме нет — она генерируется случайно
(Bernoulli с фиксированным ``random_state``) и тоже добавляется к выходу.

Выбор модели: сравниваем **LogisticRegression** и **GBTClassifier** (``pyspark.ml``)
по ROC-AUC на отложенной выборке и в ``prop_score`` отдаём модель с лучшим AUC.

Важно: ROC-AUC здесь — **инженерная selection-эвристика** (чтобы не зашивать одну модель
руками), а **не** мера качества propensity для causal-задачи. В PSM/IPW высокий AUC сам по
себе нежелателен: он означает, что трит почти детерминирован эмбеддингами, и тянет за собой
плохой overlap и экстремальные propensity/веса. Финальный causal-критерий — баланс ковариат
(``|SMD|``) и overlap/positivity **после** matching/IPW; эти диагностики живут в
numpy/sklearn-слое (:func:`rnd_reports.embeddings.overlap_diagnostics`,
:func:`rnd_reports.embeddings.covariate_balance_after_adjustment`).

Требует pyspark (extra ``spark``); импортируется лениво из пакета ``embeddings``.
"""

from __future__ import annotations

from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from . import contracts

_FEATURES_COL = "__prop_features__"
_LABEL_COL = "__prop_label__"
_PROBABILITY_COL = "__prop_probability__"
_RAW_COL = "__prop_raw__"
_ARRAY_COL = "__prop_array__"
_HASH_MOD = 1000  # гранулярность детерминированного псевдо-трита


def _random_treatment(epk_col, random_state: int, treatment_share: float):
    """Детерминированный псевдо-случайный трит из хэша ``epk_id`` + сид.

    ``F.rand`` нельзя: он недетерминирован между материализациями, поэтому метка «уехала»
    бы между fit и оценкой (и относительно ``randomSplit``). Хэш стабилен для строки при
    любом партиционировании/пересчёте.
    """
    threshold = int(round(treatment_share * _HASH_MOD))
    bucket = F.pmod(F.hash(epk_col, F.lit(random_state)), F.lit(_HASH_MOD))
    return (bucket < F.lit(threshold)).cast("int")


def _candidates(
    max_iter: int, reg_param: float, gbt_max_depth: int, gbt_max_iter: int, seed: int
) -> dict[str, object]:
    """Кандидаты-классификаторы с общими колонками features/label/probability/raw.

    ``probabilityCol``/``rawPredictionCol`` задаём сеттерами: ``GBTClassifier`` не
    принимает их как kwargs конструктора, хотя сами колонки умеет выдавать.
    """
    estimators = {
        "logreg": LogisticRegression(
            featuresCol=_FEATURES_COL, labelCol=_LABEL_COL,
            maxIter=max_iter, regParam=reg_param,
        ),
        "gbt": GBTClassifier(
            featuresCol=_FEATURES_COL, labelCol=_LABEL_COL,
            maxDepth=gbt_max_depth, maxIter=gbt_max_iter, seed=seed,
        ),
    }
    for est in estimators.values():
        est.setProbabilityCol(_PROBABILITY_COL).setRawPredictionCol(_RAW_COL)
    return estimators


def add_propensity_score(
    df: DataFrame,
    *,
    treatment_col: str = contracts.TREATMENT,
    score_col: str = contracts.PROPENSITY_SCORE,
    random_state: int = 42,
    treatment_share: float = 0.5,
    max_iter: int = 100,
    reg_param: float = 0.0,
    gbt_max_depth: int = 5,
    gbt_max_iter: int = 20,
    val_fraction: float = 0.3,
) -> DataFrame:
    """Добавить ``score_col`` = P(treatment=1 | эмбеддинги) к исходному датафрейму.

    Если колонки ``treatment_col`` нет — она генерируется случайно
    (``Bernoulli(treatment_share)`` с сидом ``random_state``) и добавляется к выходу.
    Из LogReg/GBT берётся модель с лучшим ROC-AUC на отложенной доле ``val_fraction``;
    выбор печатается одной строкой. Все исходные колонки сохраняются.
    """
    feature_cols = contracts.validate_embedding_schema(df)

    # Трит — опциональная колонка того же датафрейма. Нет → детерминированный псевдо-случайный
    # трит из хэша epk_id + random_state (стабилен между материализациями, см. _random_treatment).
    if treatment_col not in df.columns:
        df = df.withColumn(
            treatment_col,
            _random_treatment(F.col(contracts.EPK_ID), random_state, treatment_share),
        )

    labelled = df.withColumn(_LABEL_COL, F.col(treatment_col).cast("double"))
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=_FEATURES_COL)
    assembled = assembler.transform(labelled)

    # Честная оценка AUC на отложенной выборке; затем дообучаем лучшую модель на всех данных.
    train, val = assembled.randomSplit(
        [1.0 - val_fraction, val_fraction], seed=random_state
    )
    evaluator = BinaryClassificationEvaluator(
        labelCol=_LABEL_COL, rawPredictionCol=_RAW_COL, metricName="areaUnderROC"
    )
    candidates = _candidates(max_iter, reg_param, gbt_max_depth, gbt_max_iter, random_state)
    metrics = {
        name: float(evaluator.evaluate(est.fit(train).transform(val)))
        for name, est in candidates.items()
    }
    best_name = max(metrics, key=metrics.get)
    print(
        "propensity ROC-AUC:",
        {k: round(v, 4) for k, v in metrics.items()},
        "| выбрана:", best_name,
    )

    fresh = _candidates(max_iter, reg_param, gbt_max_depth, gbt_max_iter, random_state)
    model = fresh[best_name].fit(assembled)

    scored = model.transform(assembled).withColumn(
        _ARRAY_COL, vector_to_array(F.col(_PROBABILITY_COL))
    )
    scored = scored.withColumn(score_col, F.col(_ARRAY_COL)[1])
    return scored.drop(
        _FEATURES_COL, _LABEL_COL, _PROBABILITY_COL, _RAW_COL, _ARRAY_COL, "prediction"
    )
