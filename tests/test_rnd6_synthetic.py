"""Step 2: синтетический генератор + интеграция с BenchmarkDataset."""

from __future__ import annotations

import pandas as pd

from rnd_reports.datasets.adapters import (
    SyntheticAdapter,
    make_synthetic_benchmark_dataset,
)
from rnd_reports.datasets.contracts import BenchmarkDataset
from rnd_reports.feature_safety.contracts import FeatureClass
from rnd_reports.synthetic.schemas import SYNTHETIC_FEATURE_CLASSES
from rnd_reports.synthetic.scenarios import make_synthetic_scenario


def test_scenario_reproducible_by_seed() -> None:
    a = make_synthetic_scenario(n=500, seed=7)
    b = make_synthetic_scenario(n=500, seed=7)
    pd.testing.assert_frame_equal(a.data, b.data)
    assert a.true_ate == b.true_ate


def test_scenario_columns_match_registry() -> None:
    sc = make_synthetic_scenario(n=300, seed=1)
    # все размеченные признаки присутствуют в данных
    for name in SYNTHETIC_FEATURE_CLASSES:
        assert name in sc.data.columns
    # ключевые колонки на месте
    for col in (sc.id_col, sc.treatment_col, sc.target_col):
        assert col in sc.data.columns
    # все классы A–F + unsafe_demo представлены
    present = set(sc.feature_registry.values())
    assert present == set(FeatureClass)


def test_adapter_builds_benchmark_dataset() -> None:
    sc = make_synthetic_scenario(n=400, seed=2)
    ds = SyntheticAdapter().to_benchmark_dataset(sc)
    assert isinstance(ds, BenchmarkDataset)
    assert ds.n == 400
    assert set(ds.feature_columns()) == set(SYNTHETIC_FEATURE_CLASSES)
    # A/B/C допустимы в estimator; D/E/F/unsafe_demo — нет
    usable = set(ds.feature_registry.usable_in_estimator())
    assert "x_inflation" in usable  # B
    assert "x_context_1" in usable  # C
    assert "x_mediator" not in usable  # E
    assert "x_future_target_sum" not in usable  # F
    assert "x_unsafe_demo" not in usable  # unsafe_demo


def test_convenience_helper() -> None:
    ds = make_synthetic_benchmark_dataset(n=200, seed=3)
    assert isinstance(ds, BenchmarkDataset)
    assert ds.features_by_class(FeatureClass.A_PRE_TREATMENT)


def test_raw_diff_in_means_recovers_true_ate() -> None:
    # рандомизированный A/B: разность средних — несмещённая оценка истинного ATE
    sc = make_synthetic_scenario(n=8000, seed=11)
    df, t = sc.data, sc.treatment_col
    diff = df.loc[df[t] == 1, "target"].mean() - df.loc[df[t] == 0, "target"].mean()
    assert abs(diff - sc.true_ate) < 0.25
