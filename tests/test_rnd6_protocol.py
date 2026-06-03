"""Step 3: A/B протокол, метрики (§7) и сборка таблицы результатов."""

from __future__ import annotations

import math

import pytest

from rnd_reports.benchmark.contracts import RESULT_COLUMNS
from rnd_reports.benchmark.protocol import (
    estimate_ate,
    finalize_metrics,
    reduction_pct,
    run_benchmark,
    run_method,
)
from rnd_reports.benchmark.reporting import results_to_frame
from rnd_reports.datasets.adapters import make_synthetic_benchmark_dataset


def _bds(n=6000, seed=11):
    return make_synthetic_benchmark_dataset(n=n, seed=seed)


def test_estimate_ate_recovers_true_ate() -> None:
    bds = _bds()
    est = estimate_ate(bds.data, bds.target_col, bds.treatment_col)
    # истинный ATE синтетики = 2.8
    assert abs(est["ate"] - 2.8) < 0.25
    assert est["ci_low"] < est["ate"] < est["ci_high"]
    assert est["variance"] > 0


def test_reduction_pct_formula() -> None:
    # §7: 100*(1 - var_method/var_baseline)
    assert reduction_pct(5.0, 10.0) == pytest.approx(50.0)
    assert reduction_pct(10.0, 10.0) == pytest.approx(0.0)
    assert reduction_pct(5.0, None) is None
    assert reduction_pct(5.0, 0.0) is None


def test_run_benchmark_columns_and_methods() -> None:
    res = run_benchmark(_bds(), methods=["raw", "ab_hypex"], dataset_name="syn")
    assert [r.method for r in res] == ["raw", "ab_hypex"]
    frame = results_to_frame(res)
    assert list(frame.columns) == RESULT_COLUMNS
    # vs A/B для baseline ≈ 0 (тот же сырой таргет)
    for r in res:
        assert abs(r.variance_reduction_vs_ab_pct) < 1e-6
        assert r.sample_size_reduction_vs_ab_pct == r.variance_reduction_vs_ab_pct
        # CUPAC-базы нет на этом шаге
        assert r.variance_reduction_vs_sklearn_cupac_pct is None
        assert r.n_features_used == 0


def test_finalize_incremental_vs_predecessor() -> None:
    # синтетический ручной кейс: предшественник имеет большую дисперсию
    res = run_benchmark(_bds(), methods=["raw", "ab_hypex"])
    # ab_hypex predecessor = None → инкремент None
    ab = next(r for r in res if r.method == "ab_hypex")
    assert ab.predecessor_method is None
    assert ab.incremental_variance_reduction_vs_predecessor_pct is None


def test_unimplemented_method_raises() -> None:
    with pytest.raises(NotImplementedError):
        run_method(_bds(n=500), "sklearn_cupac_A")


def test_ab_hypex_has_parity_note() -> None:
    res = run_method(_bds(n=2000), "ab_hypex")
    # либо parity-сверка с HypEx, либо явная пометка о fallback
    assert "hypex" in res.diagnostic_notes.lower()
