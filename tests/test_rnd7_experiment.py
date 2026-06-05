"""Тесты causal-слоя R&D-7 (numpy/sklearn, без pyspark).

Проверяют синтетический сценарий и то, что поправка на эмбеддинги снижает смещение
наивной оценки ATE и восстанавливает баланс ковариат.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from rnd_reports.embeddings import (
    covariate_balance_after_adjustment,
    estimate_ate_with_adjustment,
    evaluate_adjustment_set_quality,
    fit_propensity,
    make_embedding_observational_scenario,
    overlap_diagnostics,
)


def _scenario_arrays(seed: int = 11):
    sc = make_embedding_observational_scenario(n=4000, k=8, true_ate=3.0, seed=seed)
    df = sc.data
    emb = df[sc.embedding_columns].to_numpy()
    reduced = PCA(n_components=5, random_state=0).fit_transform(emb)
    return sc, reduced, df["treatment"].to_numpy(), df["outcome"].to_numpy()


def test_scenario_schema_and_known_truth() -> None:
    sc = make_embedding_observational_scenario(n=500, k=6, true_ate=2.5, n_months=4, seed=1)
    df = sc.data
    assert {"epk_id", "report_dt", "treatment", "outcome"} <= set(df.columns)
    assert sc.embedding_columns == [f"col_{i:03d}" for i in range(6)]
    assert len(df) == 500 and sc.true_ate == 2.5
    assert 0.0 < df["treatment"].mean() < 1.0           # есть обе группы
    assert df["report_dt"].nunique() <= 4


def test_adjustment_reduces_bias_vs_naive() -> None:
    sc, reduced, t, y = _scenario_arrays()
    g = sc.true_ate
    naive = estimate_ate_with_adjustment(reduced, t, y, method="naive")
    ipw = estimate_ate_with_adjustment(reduced, t, y, method="propensity_weighting")
    dr = estimate_ate_with_adjustment(reduced, t, y, method="doubly_robust")
    # поправка на эмбеддинги должна заметно снизить |смещение|
    assert abs(ipw - g) < abs(naive - g)
    assert abs(dr - g) < abs(naive - g)
    assert abs(ipw - g) < 0.3                            # близко к истине


def test_balance_improves_after_weighting() -> None:
    _, reduced, t, _ = _scenario_arrays()
    bal = covariate_balance_after_adjustment(reduced, t)
    assert {"max_abs_smd_before", "max_abs_smd_after"} <= set(bal)
    assert bal["max_abs_smd_after"] < bal["max_abs_smd_before"]


def test_overlap_diagnostics_ranges() -> None:
    _, reduced, t, _ = _scenario_arrays()
    p = fit_propensity(reduced, t)
    ov = overlap_diagnostics(p)
    assert 0.0 <= ov["frac_in_0.1_0.9"] <= 1.0
    assert 0.0 <= ov["min"] <= ov["max"] <= 1.0


def test_evaluate_quality_keys_and_bias_reduction() -> None:
    sc, reduced, t, y = _scenario_arrays()
    q = evaluate_adjustment_set_quality(reduced, t, y, ground_truth_effect=sc.true_ate)
    for key in ("ate_naive", "ate_ipw", "ate_dr", "bias_naive", "bias_ipw",
                "abs_bias_reduction_ipw_pct", "max_abs_smd_before", "max_abs_smd_after"):
        assert key in q
    assert q["abs_bias_reduction_ipw_pct"] > 0           # IPW снижает смещение


def test_unknown_method_raises() -> None:
    _, reduced, t, y = _scenario_arrays()
    with pytest.raises(ValueError):
        estimate_ate_with_adjustment(reduced, t, y, method="nope")
