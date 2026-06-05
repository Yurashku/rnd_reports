"""Адаптер 2 (R&D-7): propensity score из эмбеддингов.

``PropensityScorer`` джойнит эмбеддинг-датасет с датасетом трита по ключу
``(epk_id, report_dt)`` и обучает ``LogisticRegression`` (``pyspark.ml``) на **сырых**
эмбеддингах ``col_*``, предсказывая ``P(treatment=1)``. ``transform`` сводит эмбеддинги
к одной колонке ``propensity_score``.

In-time safety: ``fit(..., cutoff=...)`` обучается только на ``report_dt <= cutoff``;
``transform`` применяется к любому срезу.

Требует pyspark (extra ``spark``); импортируется лениво из пакета ``embeddings``.
"""

from __future__ import annotations

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from . import contracts

_FEATURES_COL = "__prop_features__"
_LABEL_COL = "__prop_label__"
_PROBABILITY_COL = "__prop_probability__"
_ARRAY_COL = "__prop_array__"


class PropensityScorer:
    """Propensity score P(treatment=1 | эмбеддинги) на сырых ``col_*`` (LogisticRegression).

    Параметры:
        max_iter / reg_param: гиперпараметры ``LogisticRegression`` (детерминирован).

    Использование: ``fit(embeddings_df, treatment_df, cutoff)`` → ``transform(embeddings_df)``.
    """

    def __init__(self, max_iter: int = 100, reg_param: float = 0.0) -> None:
        self.max_iter = max_iter
        self.reg_param = reg_param
        self._assembler: VectorAssembler | None = None
        self._model = None
        self._feature_cols: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        embeddings_df: DataFrame,
        treatment_df: DataFrame,
        cutoff=None,
    ) -> "PropensityScorer":
        """Джойн по ``(epk_id, report_dt)`` + обучение LogReg на сырых эмбеддингах.

        При ``cutoff`` обучение только на ``report_dt <= cutoff`` (in-time safety).
        """
        feature_cols = contracts.validate_embedding_schema(embeddings_df)
        contracts.validate_treatment_schema(treatment_df)

        treatment = treatment_df.select(
            contracts.EPK_ID, contracts.REPORT_DT, contracts.TREATMENT
        )
        joined = embeddings_df.join(treatment, list(contracts.KEY_COLUMNS), how="inner")
        if cutoff is not None:
            joined = joined.filter(F.col(contracts.REPORT_DT) <= F.lit(cutoff))
        labelled = joined.withColumn(
            _LABEL_COL, F.col(contracts.TREATMENT).cast("double")
        )

        self._assembler = VectorAssembler(inputCols=feature_cols, outputCol=_FEATURES_COL)
        assembled = self._assembler.transform(labelled)
        lr = LogisticRegression(
            featuresCol=_FEATURES_COL,
            labelCol=_LABEL_COL,
            probabilityCol=_PROBABILITY_COL,
            maxIter=self.max_iter,
            regParam=self.reg_param,
        )
        self._model = lr.fit(assembled)
        self._feature_cols = feature_cols
        return self

    def transform(self, embeddings_df: DataFrame) -> DataFrame:
        """Вернуть ``[epk_id, report_dt, propensity_score]`` (P(treatment=1) ∈ [0, 1])."""
        if not self.is_fitted:
            raise RuntimeError("PropensityScorer не обучен: вызовите fit(...) до transform(...)")
        contracts.validate_embedding_schema(embeddings_df)

        assembled = self._assembler.transform(embeddings_df)
        scored = self._model.transform(assembled)
        with_array = scored.withColumn(
            _ARRAY_COL, vector_to_array(F.col(_PROBABILITY_COL))
        )
        return with_array.select(
            F.col(contracts.EPK_ID),
            F.col(contracts.REPORT_DT),
            F.col(_ARRAY_COL)[1].alias(contracts.PROPENSITY_SCORE),
        )
