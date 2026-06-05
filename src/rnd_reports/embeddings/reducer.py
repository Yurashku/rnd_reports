"""Адаптер 1 (R&D-7): снижение размерности эмбеддингов через PCA.

``EmbeddingReducer`` обучает PCA (``pyspark.ml.feature.PCA``) на эмбеддинг-колонках
``col_*`` и отдаёт компактное представление ``[epk_id, report_dt, emb_000, ...]``.

In-time safety: ``fit(df, cutoff=...)`` обучает PCA только на срезе
``report_dt <= cutoff``; ``transform`` применяется к любому (в т.ч. более позднему)
срезу — это исключает утечку из будущего при оценке adjustment set.

Требует pyspark (extra ``spark``); импортируется лениво из пакета ``embeddings``.
"""

from __future__ import annotations

from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from . import contracts

_FEATURES_COL = "__emb_features__"
_PCA_COL = "__emb_pca__"
_ARRAY_COL = "__emb_array__"


class EmbeddingReducer:
    """Снижение размерности эмбеддингов до ``reducted_shape`` (PCA).

    Параметры:
        reducted_shape: число выходных компонент (по умолчанию 5);
        seed: зарезервирован для совместимости API — PCA детерминирован (SVD).

    Использование: ``fit(df, cutoff)`` → ``transform(df)`` (либо ``fit_transform``).
    """

    def __init__(self, reducted_shape: int = 5, seed: int = 42) -> None:
        if reducted_shape < 1:
            raise ValueError(
                f"reducted_shape должен быть >= 1, получено {reducted_shape}"
            )
        self.reducted_shape = reducted_shape
        self.seed = seed
        self._assembler: VectorAssembler | None = None
        self._model = None
        self._feature_cols: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, df: DataFrame, cutoff=None) -> "EmbeddingReducer":
        """Обучить PCA на эмбеддингах; при ``cutoff`` — только на ``report_dt <= cutoff``."""
        feature_cols = contracts.validate_embedding_schema(df)
        train = df if cutoff is None else df.filter(F.col(contracts.REPORT_DT) <= F.lit(cutoff))

        self._assembler = VectorAssembler(inputCols=feature_cols, outputCol=_FEATURES_COL)
        assembled = self._assembler.transform(train)
        pca = PCA(k=self.reducted_shape, inputCol=_FEATURES_COL, outputCol=_PCA_COL)
        self._model = pca.fit(assembled)
        self._feature_cols = feature_cols
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """Применить обученный PCA; вернуть ``[epk_id, report_dt, emb_000, ...]``."""
        if not self.is_fitted:
            raise RuntimeError("EmbeddingReducer не обучен: вызовите fit(...) до transform(...)")
        contracts.validate_embedding_schema(df)

        assembled = self._assembler.transform(df)
        scored = self._model.transform(assembled)
        with_array = scored.withColumn(_ARRAY_COL, vector_to_array(F.col(_PCA_COL)))

        out_names = contracts.reduced_column_names(self.reducted_shape)
        select_exprs = [F.col(contracts.EPK_ID), F.col(contracts.REPORT_DT)]
        select_exprs += [F.col(_ARRAY_COL)[i].alias(name) for i, name in enumerate(out_names)]
        return with_array.select(*select_exprs)

    def fit_transform(self, df: DataFrame, cutoff=None) -> DataFrame:
        """Удобный шорткат: обучить на ``df`` (с ``cutoff``) и применить к нему же."""
        return self.fit(df, cutoff=cutoff).transform(df)
