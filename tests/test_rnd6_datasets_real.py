"""Step 8: адаптеры реальных датасетов (Hillstrom) + loader/каталог.

Тесты не требуют скачивания: используется крошечная синтетическая фикстура,
повторяющая схему Hillstrom. Реальные данные не коммитятся.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rnd_reports.benchmark.protocol import run_benchmark
from rnd_reports.datasets.adapters import (
    HillstromAdapter,
    get_adapter,
    load_benchmark_dataset,
)
from rnd_reports.datasets.catalog import get_dataset_spec
from rnd_reports.datasets.contracts import BenchmarkDataset
from rnd_reports.datasets.loaders import get_loader


def _fake_hillstrom(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seg = rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], size=n)
    treated = seg != "No E-Mail"
    spend = rng.gamma(2.0, 2.0, n) + treated * 1.5
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 12, n),
            "history_segment": rng.choice(["1) $0-$100", "2) $100-$200"], size=n),
            "history": rng.gamma(2.0, 50.0, n),
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Surburban", "Urban", "Rural"], size=n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Phone", "Web", "Multichannel"], size=n),
            "segment": seg,
            "visit": rng.integers(0, 2, n),
            "conversion": rng.integers(0, 2, n),
            "spend": spend,
        }
    )


def test_hillstrom_adapter_builds_benchmark_dataset() -> None:
    bds = HillstromAdapter().to_benchmark_dataset(_fake_hillstrom())
    assert isinstance(bds, BenchmarkDataset)
    assert set(bds.data["treatment"].unique()) <= {0, 1}
    assert bds.target_col == "target"
    # числовые A + one-hot категориальные → класс A
    a = bds.features_by_class("A_pre_treatment")
    assert {"recency", "history", "mens", "womens", "newbie"} <= set(a)
    assert any(c.startswith("channel_") for c in a)
    # поведенческие исходы помечены E и не идут в estimator
    assert set(bds.features_by_class("E_mediator_risk")) == {"visit", "conversion"}
    assert "visit" not in bds.feature_registry.usable_in_estimator()


def test_hillstrom_adapter_missing_columns_raises() -> None:
    bad = _fake_hillstrom().drop(columns=["spend"])
    with pytest.raises(ValueError):
        HillstromAdapter().to_benchmark_dataset(bad)


def test_benchmark_runs_on_hillstrom_like_data() -> None:
    bds = HillstromAdapter().to_benchmark_dataset(_fake_hillstrom(n=120, seed=1))
    res = run_benchmark(
        bds,
        methods=["ab_hypex", "sklearn_cupac_A"],
        dataset_name="hillstrom",
        dataset_type="real",
        random_state=1,
    )
    by = {r.method: r for r in res}
    assert by["sklearn_cupac_A"].dataset_type == "real"
    assert by["sklearn_cupac_A"].n == 120
    assert by["sklearn_cupac_A"].feature_groups_used == ["A"]


def test_loader_missing_file_raises_with_hint() -> None:
    loader = get_loader("hillstrom")
    with pytest.raises(FileNotFoundError):
        loader.load(path="/nonexistent/hillstrom.csv")


def test_load_benchmark_dataset_wires_loader_and_adapter(tmp_path) -> None:
    csv = tmp_path / "hillstrom.csv"
    _fake_hillstrom(n=60, seed=2).to_csv(csv, index=False)
    bds = load_benchmark_dataset("hillstrom", path=str(csv))
    assert isinstance(bds, BenchmarkDataset)
    assert bds.n == 60


def test_catalog_hillstrom_is_randomized() -> None:
    assert get_dataset_spec("hillstrom").is_randomized is True


def test_unimplemented_real_adapter_still_stub() -> None:
    with pytest.raises(NotImplementedError):
        get_adapter("criteo").to_benchmark_dataset(pd.DataFrame())
