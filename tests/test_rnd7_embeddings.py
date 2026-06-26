"""R&D-7: функции на локальном Spark.

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
from pyspark.sql import functions as F  # noqa: E402

from rnd_reports.embeddings import (  # noqa: E402
    add_propensity_score,
    contracts,
    reduce_embeddings,
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


def _embeddings_df(spark, n: int = 60, with_treatment: bool = False):
    # n наблюдений за 3 месяца; эмбеддинги слегка коррелируют между собой.
    rows = []
    for i in range(n):
        month = f"2024-0{1 + i % 3}-01"
        base = float(i)
        feats = [base + j * 0.5 + (i % 4) * 0.1 for j in range(N_FEATURES)]
        extra = (i % 2,) if with_treatment else ()
        rows.append((1000 + i, month, *feats, *extra))
    cols = ["epk_id", "report_dt"] + [f"emb_{j}_val" for j in range(N_FEATURES)]
    if with_treatment:
        cols.append("treatment")
    return spark.createDataFrame(rows, cols)


def test_reduce_embeddings_appends_columns(spark) -> None:
    df = _embeddings_df(spark)
    reduced = reduce_embeddings(df, red_size=3)
    # Исходные колонки сохранены, добавлены только red_*.
    assert reduced.columns == df.columns + ["red_0", "red_1", "red_2"]
    assert reduced.count() == df.count()


def test_reduce_embeddings_default_shape(spark) -> None:
    reduced = reduce_embeddings(_embeddings_df(spark))
    red_cols = [c for c in reduced.columns if c.startswith(contracts.REDUCED_PREFIX)]
    assert len(red_cols) == 5


def test_reduce_embeddings_accepts_legacy_col_format(spark) -> None:
    # Витрина R&D-7 подаёт легаси-формат col_000 — детектор обязан его понимать.
    rows = [(1000 + i, "2024-01-01", float(i), float(i) + 0.5) for i in range(8)]
    legacy = spark.createDataFrame(rows, ["epk_id", "report_dt", "col_000", "col_001"])
    out = reduce_embeddings(legacy, red_size=2)
    assert out.columns == legacy.columns + ["red_0", "red_1"]


def test_reduce_embeddings_missing_embedding_columns_raises(spark) -> None:
    bad = spark.createDataFrame([(1, "2024-01-01")], ["epk_id", "report_dt"])
    with pytest.raises(ValueError):
        reduce_embeddings(bad)


def test_add_propensity_score_uses_existing_treatment(spark) -> None:
    df = _embeddings_df(spark, with_treatment=True)
    scored = add_propensity_score(df, random_state=0)
    # treatment уже был → добавлена только prop_score, исходные колонки сохранены.
    assert scored.columns == df.columns + ["prop_score"]
    assert scored.count() == df.count()

    bounds = scored.select(
        F.min("prop_score").alias("lo"), F.max("prop_score").alias("hi")
    ).collect()[0]
    assert 0.0 <= bounds["lo"] <= bounds["hi"] <= 1.0


def test_add_propensity_score_generates_treatment_when_absent(spark) -> None:
    df = _embeddings_df(spark)  # без treatment
    scored = add_propensity_score(df, random_state=0)
    # treatment сгенерирован и добавлен вместе с prop_score.
    assert "treatment" in scored.columns
    assert "prop_score" in scored.columns
    assert scored.count() == df.count()


def test_add_propensity_score_random_treatment_is_deterministic(spark) -> None:
    df = _embeddings_df(spark)
    a = add_propensity_score(df, random_state=7).select("epk_id", "treatment")
    b = add_propensity_score(df, random_state=7).select("epk_id", "treatment")
    # Одинаковый random_state → одинаковое назначение трита.
    diff = a.join(b, "epk_id").filter(a["treatment"] != b["treatment"]).count()
    assert diff == 0


def test_add_propensity_score_custom_score_col(spark) -> None:
    df = _embeddings_df(spark, with_treatment=True)
    scored = add_propensity_score(df, score_col="pscore", random_state=0)
    assert "pscore" in scored.columns
