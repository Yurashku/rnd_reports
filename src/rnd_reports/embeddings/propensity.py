"""Скрипт 2 (R&D-7): propensity score (prop_score) из эмбеддингов.

``PropensityScorer`` джойнит сырой эмбеддинг-датасет (``emb_{i}_val``) с таблицей
псевдо-трита и обучает классификатор ``P(treatment=1 | эмбеддинги)``. ``transform``
сводит эмбеддинги к одной колонке ``prop_score`` — она дальше пойдёт в
propensity-score matching по Рубину (сам матчинг здесь не реализуется).

«Лучшая методология»: сравниваем **LogisticRegression** и **GBTClassifier**
(``pyspark.ml``) по ROC-AUC на отложенной выборке и в ``prop_score`` отдаём лучшую
модель. AUC обеих сохраняются в ``metrics_``, имя выбранной — в ``best_model_name_``.

Джойн трита: по ``(epk_id, report_dt)`` если в таблице трита есть ``report_dt``,
иначе по ``epk_id`` (таблица псевдо-разбиения ``epk_id × treatment``).

In-time safety: ``fit(..., cutoff=...)`` обучается только на ``report_dt <= cutoff``.

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


class PropensityScorer:
    """prop_score = P(treatment=1 | эмбеддинги); лучший из LogReg / GBT по ROC-AUC.

    Параметры:
        max_iter / reg_param: гиперпараметры ``LogisticRegression``;
        gbt_max_depth / gbt_max_iter: гиперпараметры ``GBTClassifier``;
        val_fraction: доля отложенной выборки для честной оценки AUC;
        seed: сид сплита и GBT.

    Использование: ``fit(embeddings_df, treatment_df, cutoff)`` → ``transform(embeddings_df)``.
    После ``fit`` доступны ``metrics_`` (AUC обеих моделей) и ``best_model_name_``.
    """

    def __init__(
        self,
        max_iter: int = 100,
        reg_param: float = 0.0,
        gbt_max_depth: int = 5,
        gbt_max_iter: int = 20,
        val_fraction: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.gbt_max_depth = gbt_max_depth
        self.gbt_max_iter = gbt_max_iter
        self.val_fraction = val_fraction
        self.seed = seed
        self._assembler: VectorAssembler | None = None
        self._model = None
        self.metrics_: dict[str, float] = {}
        self.best_model_name_: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _candidates(self) -> dict[str, object]:
        """Кандидаты-классификаторы с общими колонками features/label/probability/raw.

        ``probabilityCol``/``rawPredictionCol`` задаём сеттерами: ``GBTClassifier`` не
        принимает их как kwargs конструктора, хотя сами колонки умеет выдавать.
        """
        estimators = {
            "logreg": LogisticRegression(
                featuresCol=_FEATURES_COL, labelCol=_LABEL_COL,
                maxIter=self.max_iter, regParam=self.reg_param,
            ),
            "gbt": GBTClassifier(
                featuresCol=_FEATURES_COL, labelCol=_LABEL_COL,
                maxDepth=self.gbt_max_depth, maxIter=self.gbt_max_iter, seed=self.seed,
            ),
        }
        for est in estimators.values():
            est.setProbabilityCol(_PROBABILITY_COL).setRawPredictionCol(_RAW_COL)
        return estimators

    def fit(
        self,
        embeddings_df: DataFrame,
        treatment_df: DataFrame,
        cutoff=None,
    ) -> "PropensityScorer":
        """Джойн с тритом + выбор лучшего классификатора (LogReg/GBT) по ROC-AUC.

        При ``cutoff`` обучение только на ``report_dt <= cutoff`` (in-time safety).
        """
        feature_cols = contracts.validate_embedding_schema(embeddings_df)
        contracts.validate_treatment_schema(treatment_df)

        join_keys = contracts.treatment_join_keys(treatment_df)
        treatment = treatment_df.select(*join_keys, contracts.TREATMENT)
        joined = embeddings_df.join(treatment, join_keys, how="inner")
        if cutoff is not None:
            joined = joined.filter(F.col(contracts.REPORT_DT) <= F.lit(cutoff))
        labelled = joined.withColumn(_LABEL_COL, F.col(contracts.TREATMENT).cast("double"))

        self._assembler = VectorAssembler(inputCols=feature_cols, outputCol=_FEATURES_COL)
        assembled = self._assembler.transform(labelled)

        # Честная оценка AUC на отложенной выборке; затем дообучаем лучшую модель на всех данных.
        train, val = assembled.randomSplit([1.0 - self.val_fraction, self.val_fraction], seed=self.seed)
        evaluator = BinaryClassificationEvaluator(
            labelCol=_LABEL_COL, rawPredictionCol=_RAW_COL, metricName="areaUnderROC"
        )
        for name, estimator in self._candidates().items():
            scored = estimator.fit(train).transform(val)
            self.metrics_[name] = float(evaluator.evaluate(scored))

        self.best_model_name_ = max(self.metrics_, key=self.metrics_.get)
        self._model = self._candidates()[self.best_model_name_].fit(assembled)
        return self

    def transform(self, embeddings_df: DataFrame) -> DataFrame:
        """Вернуть ``[epk_id, report_dt, prop_score]`` (P(treatment=1) ∈ [0, 1])."""
        if not self.is_fitted:
            raise RuntimeError("PropensityScorer не обучен: вызовите fit(...) до transform(...)")
        contracts.validate_embedding_schema(embeddings_df)

        assembled = self._assembler.transform(embeddings_df)
        scored = self._model.transform(assembled)
        with_array = scored.withColumn(_ARRAY_COL, vector_to_array(F.col(_PROBABILITY_COL)))
        return with_array.select(
            F.col(contracts.EPK_ID),
            F.col(contracts.REPORT_DT),
            F.col(_ARRAY_COL)[1].alias(contracts.PROPENSITY_SCORE),
        )
