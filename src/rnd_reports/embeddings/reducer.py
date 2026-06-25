"""Скрипт 1 (R&D-7): понижение размерности эмбеддингов через PCA.

``EmbeddingReducer`` обучает ``StandardScaler`` + ``PCA`` (``pyspark.ml``) на
эмбеддинг-колонках ``emb_{i}_val`` и отдаёт компактное представление
``[epk_id, report_dt, red_0, ..., red_{red_size-1}]``.

Стандартизация перед PCA обязательна: PCA чувствителен к масштабу признаков, а
сырые компоненты эмбеддингов могут иметь разные дисперсии.

In-time safety: ``fit(df, cutoff=...)`` обучает преобразование только на срезе
``report_dt <= cutoff``; ``transform`` применяется к любому (в т.ч. более позднему)
срезу — это исключает утечку из будущего.

Требует pyspark (extra ``spark``); импортируется лениво из пакета ``embeddings``.
"""

from __future__ import annotations

from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from . import contracts

_FEATURES_COL = "__emb_features__"
_SCALED_COL = "__emb_scaled__"
_PCA_COL = "__emb_pca__"
_ARRAY_COL = "__emb_array__"


class EmbeddingReducer:
    """Снижение размерности эмбеддингов до ``red_size`` (StandardScaler + PCA).

    Параметры:
        red_size: число выходных компонент (по умолчанию 5);
        seed: зарезервирован для совместимости API — PCA детерминирован (SVD).

    Использование: ``fit(df, cutoff)`` → ``transform(df)`` (либо ``fit_transform``).
    """

    def __init__(self, red_size: int = 5, seed: int = 42) -> None:
        if red_size < 1:
            raise ValueError(f"red_size должен быть >= 1, получено {red_size}")
        self.red_size = red_size
        self.seed = seed
        self._assembler: VectorAssembler | None = None
        self._scaler_model = None
        self._pca_model = None
        self._feature_cols: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._pca_model is not None

    def fit(self, df: DataFrame, cutoff=None) -> "EmbeddingReducer":
        """Обучить StandardScaler+PCA; при ``cutoff`` — только на ``report_dt <= cutoff``."""
        feature_cols = contracts.validate_embedding_schema(df)
        train = df if cutoff is None else df.filter(F.col(contracts.REPORT_DT) <= F.lit(cutoff))

        self._assembler = VectorAssembler(inputCols=feature_cols, outputCol=_FEATURES_COL)
        assembled = self._assembler.transform(train)
        scaler = StandardScaler(
            inputCol=_FEATURES_COL, outputCol=_SCALED_COL, withMean=True, withStd=True
        )
        self._scaler_model = scaler.fit(assembled)
        scaled = self._scaler_model.transform(assembled)
        pca = PCA(k=self.red_size, inputCol=_SCALED_COL, outputCol=_PCA_COL)
        self._pca_model = pca.fit(scaled)
        self._feature_cols = feature_cols
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """Применить обученное преобразование; вернуть ``[epk_id, report_dt, red_0, ...]``."""
        if not self.is_fitted:
            raise RuntimeError("EmbeddingReducer не обучен: вызовите fit(...) до transform(...)")
        contracts.validate_embedding_schema(df)

        assembled = self._assembler.transform(df)
        scaled = self._scaler_model.transform(assembled)
        scored = self._pca_model.transform(scaled)
        with_array = scored.withColumn(_ARRAY_COL, vector_to_array(F.col(_PCA_COL)))

        out_names = contracts.reduced_column_names(self.red_size)
        select_exprs = [F.col(contracts.EPK_ID), F.col(contracts.REPORT_DT)]
        select_exprs += [F.col(_ARRAY_COL)[i].alias(name) for i, name in enumerate(out_names)]
        return with_array.select(*select_exprs)

    def fit_transform(self, df: DataFrame, cutoff=None) -> DataFrame:
        """Удобный шорткат: обучить на ``df`` (с ``cutoff``) и применить к нему же."""
        return self.fit(df, cutoff=cutoff).transform(df)
