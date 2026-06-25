"""R&D-7: импорт пакета и контракты схем работают БЕЗ установленного pyspark."""

from __future__ import annotations

import types

import pytest

from rnd_reports import embeddings
from rnd_reports.embeddings import contracts


def _fake_df(columns: list[str]):
    """Минимальный объект-заглушка с атрибутом ``.columns`` (как у pyspark DataFrame)."""
    return types.SimpleNamespace(columns=columns)


def test_package_imports_without_pyspark() -> None:
    # Сам пакет и контрактные хелперы доступны без pyspark.
    assert hasattr(embeddings, "embedding_feature_columns")
    assert callable(embeddings.validate_embedding_schema)


def test_embedding_feature_columns_natural_order() -> None:
    # Основной формат emb_{i}_val: сортировка по числовому индексу.
    df = _fake_df(["epk_id", "report_dt", "emb_2_val", "emb_10_val", "emb_1_val"])
    assert contracts.embedding_feature_columns(df) == ["emb_1_val", "emb_2_val", "emb_10_val"]


def test_embedding_feature_columns_legacy_col_format() -> None:
    # Легаси-формат col_000 (витрина R&D-7) тоже распознаётся.
    df = _fake_df(["epk_id", "report_dt", "col_002", "col_010", "col_001"])
    assert contracts.embedding_feature_columns(df) == ["col_001", "col_002", "col_010"]


def test_reduced_column_names() -> None:
    assert contracts.reduced_column_names(3) == ["red_0", "red_1", "red_2"]
    with pytest.raises(ValueError):
        contracts.reduced_column_names(0)


def test_validate_embedding_schema_ok_and_errors() -> None:
    good = _fake_df(["epk_id", "report_dt", "emb_0_val", "emb_1_val"])
    assert contracts.validate_embedding_schema(good) == ["emb_0_val", "emb_1_val"]

    with pytest.raises(ValueError):  # нет ключевых колонок
        contracts.validate_embedding_schema(_fake_df(["emb_0_val"]))
    with pytest.raises(ValueError):  # нет ни одной эмбеддинг-колонки
        contracts.validate_embedding_schema(_fake_df(["epk_id", "report_dt"]))


def test_validate_treatment_schema_and_join_keys() -> None:
    # report_dt опционален: таблица псевдо-разбиения может быть epk_id × treatment.
    with_dt = _fake_df(["epk_id", "report_dt", "treatment"])
    no_dt = _fake_df(["epk_id", "treatment"])
    assert contracts.validate_treatment_schema(with_dt) is None
    assert contracts.validate_treatment_schema(no_dt) is None
    assert contracts.treatment_join_keys(with_dt) == ["epk_id", "report_dt"]
    assert contracts.treatment_join_keys(no_dt) == ["epk_id"]

    with pytest.raises(ValueError):  # нет treatment
        contracts.validate_treatment_schema(_fake_df(["epk_id", "report_dt"]))


def test_lazy_adapters_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = embeddings.NotAnAdapter
