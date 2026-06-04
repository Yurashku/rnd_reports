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
    # observational / advertising «песочницы» требуют локальных данных (нет clean A/B).
    with pytest.raises((FileNotFoundError, NotImplementedError)):
        ea.load_completejourney_research_dataset(data_dir="/nonexistent/dunnhumby")
    with pytest.raises((FileNotFoundError, NotImplementedError)):
        ea.load_criteo_private_ad_research_dataset(data_dir="/nonexistent/criteo_priv")


def _write_tiny_dunnhumby(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tx = pd.DataFrame({
        "household_id": [1, 1, 1, 2, 2, 3, 4, 4],
        "basket_id":    [1, 2, 3, 4, 5, 6, 7, 8],
        "product_id":   [1, 2, 3, 1, 2, 1, 3, 1],
        "quantity":     [1, 1, 2, 1, 1, 3, 1, 1],
        "sales_value":  [10, 5, 20, 8, 12, 30, 7, 9],
        "coupon_disc":  [0, 0, 1, 0, 0, 2, 0, 0],
        "transaction_timestamp": pd.to_datetime([
            "2017-02-01", "2017-04-02", "2017-04-15",  # hh1: pre, cwin, post
            "2017-03-01", "2017-04-20",                # hh2: pre, post
            "2017-01-15",                              # hh3: pre only
            "2017-04-03", "2017-04-25",                # hh4: cwin, post
        ]),
    })
    tx.to_csv(root / "transactions.csv", index=False)
    pd.DataFrame({
        "campaign_id": [1, 2],
        "campaign_type": ["Type A", "Type B"],
        "start_date": ["2017-04-01", "2017-02-15"],
        "end_date": ["2017-05-01", "2017-03-01"],
    }).to_csv(root / "campaign_descriptions.csv", index=False)
    # анкер-кампания 1 получили hh1,hh2; прежняя кампания 2 — hh1.
    pd.DataFrame({
        "campaign_id": [1, 1, 2],
        "household_id": [1, 2, 1],
    }).to_csv(root / "campaigns.csv", index=False)
    pd.DataFrame({
        "household_id": [1, 2],
        "age": ["25-34", "45-54"],
        "income": ["35-49K", "50-74K"],
    }).to_csv(root / "demographics.csv", index=False)
    pd.DataFrame({
        "household_id": [1],
        "coupon_upc": [111],
        "campaign_id": [1],
        "redemption_date": ["2017-04-16"],
    }).to_csv(root / "coupon_redemptions.csv", index=False)


def test_dunnhumby_adapter_builds_anchor_research_dataset(tmp_path) -> None:
    from rnd_reports.feature_safety.contracts import FeatureClass

    root = tmp_path / "dunnhumby"
    _write_tiny_dunnhumby(root)
    rd = ea.load_completejourney_research_dataset(
        data_dir=str(root), anchor_campaign_id=1, cutoff_days=7, horizon_days=30,
    )
    assert isinstance(rd, ea.ResearchDataset)
    assert rd.dataset_type == "research_sandbox"
    b = rd.benchmark
    # все 4 домохозяйства; анкер-кампанию получили hh1,hh2.
    assert b.n == 4
    assert b.data.set_index("id")["treatment"].to_dict() == {1: 1, 2: 1, 3: 0, 4: 0}
    # target = post-window spend (hh1=20, hh2=12, hh4=9, hh3=0).
    assert b.data.set_index("id")["target"].to_dict() == {1: 20.0, 2: 12.0, 3: 0.0, 4: 9.0}
    reg = b.feature_registry
    assert reg.by_class(FeatureClass.A_PRE_TREATMENT)            # pre_* + демография
    assert "cwin_spend" in reg.by_class(FeatureClass.C_BALANCE_GATED_INTIME)
    assert "n_prior_campaigns" in reg.by_class(FeatureClass.D_DAG_REQUIRED)
    assert "redemptions_in_window" in reg.by_class(FeatureClass.E_MEDIATOR_RISK)
    # observational: helper не должен молча выдавать C за реальную валидацию.
    assert "demo" in rd.notes.lower() or "observational" in rd.notes.lower()


def test_criteo_private_ad_adapter_builds_event_log_sandbox(tmp_path) -> None:
    from rnd_reports.feature_safety.contracts import FeatureClass

    root = tmp_path / "criteo_private_ad"
    root.mkdir(parents=True, exist_ok=True)
    n = 6
    df = pd.DataFrame({
        "id": [f"h{i}" for i in range(n)],
        "user_id": [f"u{i}" for i in range(n)],
        "display_order": [1, 2, 1, 3, 1, 2],
        "campaign_id": [10, 11, 10, 12, 11, 10],
        "publisher_id": [5, 6, 5, 7, 6, 5],
        "is_clicked": [1, 0, 1, 0, 1, 0],
        "is_visit": [0, 0, 1, 0, 1, 0],
        "is_click_landed": [1, 0, 0, 0, 1, 0],
        "nb_sales": [None, None, 1.0, None, 2.0, None],
        "features_ctx_not_constrained_0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "features_browser_bits_constrained_0": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "features_kv_bits_constrained_0": [3, 1, 4, 1, 5, 9],
        "features_kv_not_constrained_0": [2, 7, 1, 8, 2, 8],
        "features_not_available_0": [None] * n,        # плейсхолдер — исключается
        "sale_delay_after_display_array": [[], [1], [1, 2], [], [3], []],
    })
    df.to_parquet(root / "sample_10k.parquet", index=False)

    rd = ea.load_criteo_private_ad_research_dataset(data_dir=str(root))
    assert rd.dataset_type == "event_log_sandbox"
    b = rd.benchmark
    assert b.n == n
    # treatment = топ-показ display_order==1.
    assert b.data["treatment"].tolist() == [1, 0, 1, 0, 1, 0]
    assert b.data["target"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # is_clicked
    reg = b.feature_registry
    a = reg.by_class(FeatureClass.A_PRE_TREATMENT)
    d = reg.by_class(FeatureClass.D_DAG_REQUIRED)
    f = reg.by_class(FeatureClass.F_LEAKAGE)
    assert "features_ctx_not_constrained_0" in a
    assert "features_browser_bits_constrained_0" in a
    assert "campaign_id" in d and "features_kv_bits_constrained_0" in d
    assert "is_visit" in f and "sale_delay_after_display_count" in f
    # плейсхолдер и id не попадают в реестр признаков.
    assert "features_not_available_0" not in reg.names()


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
