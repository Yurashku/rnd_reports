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
    df = _fake_df(["epk_id", "report_dt", "col_002", "col_010", "col_001"])
    assert contracts.embedding_feature_columns(df) == ["col_001", "col_002", "col_010"]


def test_reduced_column_names() -> None:
    assert contracts.reduced_column_names(3) == ["emb_000", "emb_001", "emb_002"]
    with pytest.raises(ValueError):
        contracts.reduced_column_names(0)


def test_validate_embedding_schema_ok_and_errors() -> None:
    good = _fake_df(["epk_id", "report_dt", "col_000", "col_001"])
    assert contracts.validate_embedding_schema(good) == ["col_000", "col_001"]

    with pytest.raises(ValueError):  # нет ключевых колонок
        contracts.validate_embedding_schema(_fake_df(["col_000"]))
    with pytest.raises(ValueError):  # нет ни одной col_*
        contracts.validate_embedding_schema(_fake_df(["epk_id", "report_dt"]))


def test_validate_treatment_schema_errors() -> None:
    ok = _fake_df(["epk_id", "report_dt", "treatment"])
    assert contracts.validate_treatment_schema(ok) is None
    with pytest.raises(ValueError):
        contracts.validate_treatment_schema(_fake_df(["epk_id", "report_dt"]))


def test_lazy_adapters_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = embeddings.NotAnAdapter
