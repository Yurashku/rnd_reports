"""R&D-7: адаптеры на локальном Spark.

Пропускается, если pyspark не установлен ИЛИ локальный Spark/JVM не стартует (нет JAVA_HOME,
нет Java и т.п.) — чтобы базовый прогон тестов оставался зелёным в любом окружении.
"""

from __future__ import annotations

import glob
import os
import shutil

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from rnd_reports.embeddings import (  # noqa: E402
    EmbeddingReducer,
    PropensityScorer,
    contracts,
)


def _ensure_java_home() -> None:
    """Подставить JAVA_HOME, если не задан (pyspark требует его для запуска JVM)."""
    if os.environ.get("JAVA_HOME") and os.path.isdir(os.environ["JAVA_HOME"]):
        return
    candidates = sorted(glob.glob("/usr/lib/jvm/*"), reverse=True)
    java = shutil.which("java")
    if java:
        # /usr/lib/jvm/<jdk>/jre/bin/java -> /usr/lib/jvm/<jdk>
        home = os.path.realpath(java)
        for _ in range(3):
            home = os.path.dirname(home)
        if os.path.isdir(home):
            candidates.insert(0, home)
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "bin")):
            os.environ["JAVA_HOME"] = cand
            return


@pytest.fixture(scope="module")
def spark():
    _ensure_java_home()
    try:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("rnd7-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
    except Exception as exc:  # noqa: BLE001 — нет JVM/JAVA_HOME → пропускаем, не падаем
        pytest.skip(f"локальный Spark недоступен: {type(exc).__name__}: {exc}")
    yield session
    session.stop()


N_FEATURES = 6


def _embeddings_df(spark):
    # 12 наблюдений за 3 месяца; эмбеддинги слегка коррелируют между собой.
    rows = []
    for i in range(12):
        month = f"2024-0{1 + i % 3}-01"
        base = float(i)
        feats = [base + j * 0.5 + (i % 4) * 0.1 for j in range(N_FEATURES)]
        rows.append((1000 + i, month, *feats))
    cols = ["epk_id", "report_dt"] + [f"col_{j:03d}" for j in range(N_FEATURES)]
    return spark.createDataFrame(rows, cols)


def _treatment_df(spark):
    rows = [(1000 + i, f"2024-0{1 + i % 3}-01", i % 2) for i in range(12)]
    return spark.createDataFrame(rows, ["epk_id", "report_dt", "treatment"])


def test_reducer_output_schema_and_count(spark) -> None:
    df = _embeddings_df(spark)
    reduced = EmbeddingReducer(reducted_shape=3).fit_transform(df)
    assert reduced.columns == ["epk_id", "report_dt", "emb_000", "emb_001", "emb_002"]
    assert reduced.count() == df.count()


def test_reducer_default_shape(spark) -> None:
    reduced = EmbeddingReducer().fit_transform(_embeddings_df(spark))
    emb_cols = [c for c in reduced.columns if c.startswith(contracts.REDUCED_PREFIX)]
    assert len(emb_cols) == 5


def test_reducer_in_time_fit_then_transform_later(spark) -> None:
    df = _embeddings_df(spark)
    reducer = EmbeddingReducer(reducted_shape=2).fit(df, cutoff="2024-01-01")
    # обучились только на январе, применяем ко всем срезам
    out = reducer.transform(df)
    assert out.columns == ["epk_id", "report_dt", "emb_000", "emb_001"]
    assert out.count() == df.count()


def test_reducer_transform_before_fit_raises(spark) -> None:
    with pytest.raises(RuntimeError):
        EmbeddingReducer().transform(_embeddings_df(spark))


def test_propensity_join_and_range(spark) -> None:
    emb = _embeddings_df(spark)
    trt = _treatment_df(spark)
    scores = PropensityScorer().fit(emb, trt).transform(emb)
    assert scores.columns == ["epk_id", "report_dt", "propensity_score"]
    assert scores.count() == emb.count()

    from pyspark.sql import functions as F

    bounds = scores.select(
        F.min("propensity_score").alias("lo"), F.max("propensity_score").alias("hi")
    ).collect()[0]
    assert 0.0 <= bounds["lo"] <= bounds["hi"] <= 1.0


def test_propensity_in_time_cutoff(spark) -> None:
    emb = _embeddings_df(spark)
    trt = _treatment_df(spark)
    scorer = PropensityScorer().fit(emb, trt, cutoff="2024-02-01")
    scores = scorer.transform(emb)
    assert scores.count() == emb.count()


def test_reducer_missing_embedding_columns_raises(spark) -> None:
    bad = spark.createDataFrame([(1, "2024-01-01")], ["epk_id", "report_dt"])
    with pytest.raises(ValueError):
        EmbeddingReducer().fit(bad)
