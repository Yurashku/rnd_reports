"""Скрипт 1 (R&D-7): понижение размерности эмбеддингов через PCA.

``reduce_embeddings`` обучает ``StandardScaler`` + ``PCA`` (``pyspark.ml``) на
эмбеддинг-колонках ``emb_{i}_val`` и **добавляет** к исходному датафрейму компактные
колонки ``red_0, ..., red_{red_size-1}`` (все исходные колонки сохраняются).

Стандартизация перед PCA обязательна: PCA чувствителен к масштабу признаков, а сырые
компоненты эмбеддингов могут иметь разные дисперсии. PCA детерминирован (SVD), поэтому
сид не нужен.

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


def reduce_embeddings(df: DataFrame, red_size: int = 5) -> DataFrame:
    """Понизить размерность эмбеддингов до ``red_size`` (StandardScaler + PCA).

    На вход — pyspark ``DataFrame`` со схемой ``epk_id, report_dt, emb_{i}_val`` (легаси
    ``col_{i}`` тоже понимается). Возвращает **тот же** датафрейм с добавленными колонками
    ``red_0 ... red_{red_size-1}`` (исходные колонки, включая сырые эмбеддинги, сохраняются).
    """
    feature_cols = contracts.validate_embedding_schema(df)
    out_names = contracts.reduced_column_names(red_size)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol=_FEATURES_COL)
    assembled = assembler.transform(df)
    scaler_model = StandardScaler(
        inputCol=_FEATURES_COL, outputCol=_SCALED_COL, withMean=True, withStd=True
    ).fit(assembled)
    scaled = scaler_model.transform(assembled)
    pca_model = PCA(k=red_size, inputCol=_SCALED_COL, outputCol=_PCA_COL).fit(scaled)

    scored = pca_model.transform(scaled).withColumn(_ARRAY_COL, vector_to_array(F.col(_PCA_COL)))
    for i, name in enumerate(out_names):
        scored = scored.withColumn(name, F.col(_ARRAY_COL)[i])
    return scored.drop(_FEATURES_COL, _SCALED_COL, _PCA_COL, _ARRAY_COL)
