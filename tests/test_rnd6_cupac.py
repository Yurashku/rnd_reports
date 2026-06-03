"""Step 4: sklearn_cupac_A (основной baseline) + hypex_cupac (reference/parity)."""

from __future__ import annotations

from rnd_reports.benchmark.contracts import SAFETY_REFERENCE, SAFETY_UNAVAILABLE
from rnd_reports.benchmark.protocol import run_benchmark, run_method
from rnd_reports.datasets.adapters import make_synthetic_benchmark_dataset
from rnd_reports.variance_reduction.hypex_cupac_adapter import hypex_available
from rnd_reports.variance_reduction.local_cupac import local_cupac_adjust


def _bds(n=6000, seed=11):
    return make_synthetic_benchmark_dataset(n=n, seed=seed)


def test_local_cupac_adjust_reduces_variance() -> None:
    bds = _bds()
    a_features = bds.features_by_class("A_pre_treatment")
    adjusted, info = local_cupac_adjust(
        bds.data, bds.target_col, a_features, random_state=11
    )
    assert adjusted.var() < bds.data[bds.target_col].var()
    assert info["variance_reduction"] > 0


def test_sklearn_cupac_A_reduces_variance_and_unbiased() -> None:
    res = run_method(_bds(), "sklearn_cupac_A", random_state=11)
    assert res.method == "sklearn_cupac_A"
    # относительные метрики заполняет run_benchmark/finalize_metrics (см. ниже);
    # одиночный run_method даёт сырую строку — проверяем ATE (несмещён) и набор признаков
    assert abs(res.ate - 2.8) < 0.3
    assert res.feature_groups_used == ["A"]
    assert res.n_features_used == 6
    assert res.safety_status == "ok"


def test_chain_variance_reduction_ordering() -> None:
    res = run_benchmark(
        _bds(),
        methods=["ab_hypex", "hypex_cupac", "sklearn_cupac_A"],
        random_state=11,
    )
    by = {r.method: r for r in res}
    # sklearn_cupac_A снижает дисперсию относительно A/B
    assert by["sklearn_cupac_A"].variance_reduction_vs_ab_pct > 0
    # инкремент vs предшественник (ab_hypex) положителен
    assert by["sklearn_cupac_A"].incremental_variance_reduction_vs_predecessor_pct > 0
    # vs самого себя как cupac-базы — ноль
    assert abs(by["sklearn_cupac_A"].variance_reduction_vs_sklearn_cupac_pct) < 1e-6


def test_hypex_cupac_is_reference() -> None:
    res = run_method(_bds(), "hypex_cupac", random_state=11)
    assert res.predecessor_method is None
    if hypex_available():
        assert res.safety_status == SAFETY_REFERENCE
        assert res.ate is not None and abs(res.ate - 2.8) < 0.3
        assert res.adjusted_target_variance is not None
    else:
        assert res.safety_status in (SAFETY_UNAVAILABLE, SAFETY_REFERENCE)
