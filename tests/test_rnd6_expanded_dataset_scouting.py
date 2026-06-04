"""Тесты расширенного датасет-скаутинга R&D-6 (без скачиваний).

``schema_summary``/``classify_columns`` — чистые функции на крошечном DataFrame;
проверки документа/каталога статичны; адаптеры обязаны падать понятной ошибкой при
отсутствии локальных данных. Реальные датасеты НЕ качаются.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rnd_reports.datasets import expanded_adapters as ea
from rnd_reports.datasets.catalog import CATALOG
from rnd_reports.datasets.inspect import classify_columns, schema_summary

ROOT = Path(__file__).resolve().parents[1]

_BASE_COLS = [
    "column", "dtype", "missing_pct", "n_unique", "example_values",
    "looks_like_datetime", "looks_like_id", "looks_like_binary",
    "looks_like_target_or_treatment",
]


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signup_date": ["2021-01-01", "2021-02-15", None],
            "treatment": [0, 1, 0],
            "f3": [1.2, 3.4, 5.6],
            "client_id": ["a", "b", "c"],
            "amount": [1.5, 2.0, 3.5],
        }
    )


def test_schema_summary_default_is_backward_compatible() -> None:
    out = schema_summary(_tiny_df())
    assert list(out.columns) == _BASE_COLS
    assert len(out) == 5


def test_schema_summary_optional_columns() -> None:
    out = schema_summary(_tiny_df(), include_feature_family=True, include_notes=True)
    assert "feature_family" in out.columns and "notes" in out.columns
    fam = out.set_index("column")["feature_family"]
    assert fam["f3"] == "anonymized"
    assert fam["client_id"] == "id"
    assert fam["signup_date"] == "datetime"
    assert fam["amount"] == "named"


def test_classify_columns_groups_roles() -> None:
    groups = classify_columns(_tiny_df())
    assert "f3" in groups["anonymized"]
    assert "client_id" in groups["id"]
    assert "signup_date" in groups["datetime"]
    assert "treatment" in groups["target_or_treatment"]
    assert "amount" in groups["named"]


def test_scouting_doc_exists_and_lists_datasets() -> None:
    doc = ROOT / "docs" / "06_expanded_dataset_scouting.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for name in ["Hillstrom", "Lenta", "Criteo", "X5", "MegaFon", "Orange",
                 "Open Bandit", "dunnhumby", "CriteoPrivateAd"]:
        assert name in text, f"в scouting-доке нет {name}"
    # честная формулировка вывода должна присутствовать
    assert "A+B+C-валидация на публичных реальных датасетах не подтверждена" in text


def test_catalog_contains_new_sandbox_candidates() -> None:
    for name in ["open_bandit", "dunnhumby_complete_journey", "criteo_private_ad"]:
        assert name in CATALOG, f"в каталоге нет {name}"
        assert CATALOG[name].source_url


def test_expanded_adapters_fail_clearly_without_local_data() -> None:
    with pytest.raises(FileNotFoundError):
        ea.load_lenta_benchmark_dataset(path="/nonexistent/lenta.csv.gz")
    with pytest.raises(FileNotFoundError):
        ea.load_orange_belgium_benchmark_dataset(path="/nonexistent/orange.csv")
    with pytest.raises(FileNotFoundError):
        ea.load_lazada_descn_benchmark_dataset(path="/nonexistent/lazada.csv")
    with pytest.raises(FileNotFoundError):
        ea.load_x5_a_only_benchmark_dataset(data_dir="/nonexistent/x5")


def test_research_dataset_sandbox_stubs_raise_or_require_data() -> None:
    # observational / advertising «песочницы» не имеют clean A/B-контракта.
    with pytest.raises((FileNotFoundError, NotImplementedError)):
        ea.load_completejourney_research_dataset(data_dir="/nonexistent/dunnhumby")
    with pytest.raises((FileNotFoundError, NotImplementedError)):
        ea.load_criteo_private_ad_research_dataset(data_dir="/nonexistent/criteo_priv")


_DELTA_COLUMNS = [
    "dataset", "validation_level", "method", "predecessor_method", "n", "target",
    "treatment_col", "ate", "se", "ci_low", "ci_high", "adjusted_target_variance",
    "variance_reduction_vs_ab_pct", "sample_size_reduction_vs_ab_pct",
    "variance_reduction_vs_cupac_A_pct", "sample_size_reduction_vs_cupac_A_pct",
    "incremental_variance_reduction_vs_predecessor_pct",
    "incremental_sample_size_reduction_vs_predecessor_pct",
    "feature_groups_used", "n_features_used", "safety_status", "diagnostic_notes",
]


def test_delta_csv_has_required_columns_if_present() -> None:
    csv = ROOT / "rnd" / "06_safe_intime_cupac" / "expanded_dataset_delta_results.csv"
    if not csv.exists():
        pytest.skip("delta CSV ещё не сгенерирован")
    df = pd.read_csv(csv)
    assert list(df.columns) == _DELTA_COLUMNS


def test_criteo_targets_multi_target_safe() -> None:
    # DataFrame с двумя исходами → оба извлекаются (без молчаливой потери conversion).
    multi = pd.DataFrame({"visit": [0, 1, 1], "conversion": [0, 0, 1]})
    got = ea._criteo_targets(multi)
    assert set(got) == {"visit", "conversion"}
    assert list(got["conversion"]) == [0, 0, 1]
    # named Series → используется её имя.
    assert set(ea._criteo_targets(pd.Series([0, 1], name="conversion"))) == {"conversion"}
    # «голый» ndarray → дефолтный visit.
    assert set(ea._criteo_targets(np.array([0, 1, 0]))) == {"visit"}


def test_no_raw_data_tracked_by_git() -> None:
    # data/ должна быть gitignored — ни один сырой файл не отслеживается git.
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "data/"], cwd=ROOT, capture_output=True, text=True,
            timeout=30,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pytest.skip("git недоступен")
    assert tracked == "", f"в git попали сырые данные: {tracked.splitlines()[:5]}"
