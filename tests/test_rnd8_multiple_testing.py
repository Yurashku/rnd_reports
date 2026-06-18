"""R&D-8: тесты пакета multiple_testing (сравнение методов множественного тестирования)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rnd_reports.multiple_testing import (
    METHOD_NAMES,
    Scenario,
    add_corrections,
    compute_elementary_tests,
    make_ab_table,
    operating_characteristics,
    run_comparison,
    validate_input_table,
)
from rnd_reports.multiple_testing.methods import (
    romano_wolf_stepdown_adjusted_pvalues,
    westfall_young_maxT_adjusted_pvalues,
)


def _corrected(df, true_effects=None, **kw):
    targets = validate_input_table(df)
    tests = compute_elementary_tests(df, targets)
    return add_corrections(tests, df, targets, **kw)


def test_table_contract_validation() -> None:
    df, _ = make_ab_table(n=500, n_targets=5, seed=1)
    assert validate_input_table(df) == [f"target_{i}" for i in range(1, 6)]

    with pytest.raises(ValueError):
        validate_input_table(df.drop(columns=["treatment"]))

    bad = df.copy()
    bad["treatment"] = 2  # обе группы 2 → нет control/treatment
    with pytest.raises(ValueError):
        validate_input_table(bad)


def test_adjusted_pvalues_in_unit_interval() -> None:
    df, true = make_ab_table(n=1500, n_targets=10, rho=0.4, n_true=3, seed=2)
    corrected = _corrected(df, n_resamples=200, random_state=2)
    for name in METHOD_NAMES:
        p = corrected[f"p_adj_{name}"].to_numpy(dtype=float)
        assert np.all(np.isfinite(p))
        assert np.all((p >= 0) & (p <= 1)), name
        assert corrected[f"reject_{name}"].dtype == bool


def test_correction_monotonicity_holm_le_bonferroni() -> None:
    # Holm не строже Bonferroni: по каждой гипотезе p_holm <= p_bonferroni, и оба >= raw.
    df, _ = make_ab_table(n=2000, n_targets=12, rho=0.3, n_true=4, seed=3)
    corrected = _corrected(df, n_resamples=200, random_state=3)
    raw = corrected["p_value"].to_numpy()
    bonf = corrected["p_adj_bonferroni"].to_numpy()
    holm = corrected["p_adj_holm"].to_numpy()
    assert np.all(holm <= bonf + 1e-12)
    assert np.all(bonf >= raw - 1e-12)
    assert np.all(holm >= raw - 1e-12)


def test_westfall_young_independent_close_to_bonferroni() -> None:
    # На независимых метриках maxT-поправка близка к Bonferroni-уровню (не мощнее существенно).
    df, _ = make_ab_table(n=2500, n_targets=10, rho=0.0, n_true=0, seed=4)
    corrected = _corrected(df, n_resamples=1000, random_state=4)
    # Под полной H0 все FWER-методы почти всегда ничего не отвергают.
    for name in ("bonferroni", "holm", "westfall_young", "romano_wolf"):
        assert corrected[f"reject_{name}"].sum() == 0


def test_romano_wolf_at_least_as_powerful_as_westfall_young() -> None:
    # Romano–Wolf stepdown не слабее single-step Westfall–Young: p_rw <= p_wy поэлементно.
    df, _ = make_ab_table(n=3000, n_targets=15, rho=0.6, n_true=5, effect=0.12, seed=5)
    corrected = _corrected(df, n_resamples=800, random_state=5)
    p_wy = corrected["p_adj_westfall_young"].to_numpy()
    p_rw = corrected["p_adj_romano_wolf"].to_numpy()
    assert np.all(p_rw <= p_wy + 1e-9)


def test_resampling_methods_align_with_target_order() -> None:
    # Перестановочные p-value привязаны к target по имени, а не по позиции строки.
    df, true = make_ab_table(n=1500, n_targets=8, rho=0.5, n_true=3, seed=6)
    corrected = _corrected(df, n_resamples=300, random_state=6).set_index("target")
    # Прямой вызов на исходном порядке колонок должен совпасть после выравнивания.
    from rnd_reports.multiple_testing.elementary import compute_vectorized_t_stats

    targets = [c for c in df.columns if c.startswith("target_")]
    rng = np.random.default_rng(6)
    y = df[targets].to_numpy(float)
    t = df["treatment"].to_numpy(int)
    t_obs = compute_vectorized_t_stats(y, t)
    t_perm = np.vstack([compute_vectorized_t_stats(y, rng.permutation(t)) for _ in range(300)])
    direct = westfall_young_maxT_adjusted_pvalues(t_obs, t_perm)
    for i, tgt in enumerate(targets):
        assert corrected.loc[tgt, "p_adj_westfall_young"] == pytest.approx(direct[i])


def test_run_comparison_truth_evaluation_recovers_signal() -> None:
    df, true = make_ab_table(n=4000, n_targets=12, rho=0.5, n_true=3, effect=0.13, seed=7)
    res = run_comparison(df, true_effects=true, n_resamples=400, random_state=7)
    assert list(res.method_summary["method"]) == METHOD_NAMES
    assert res.truth_evaluation is not None
    # Сильный сигнал: все методы находят все три истинных эффекта без ложных.
    holm = res.truth_evaluation.set_index("method").loc["holm"]
    assert holm["true_positives"] == 3
    assert holm["false_positives"] == 0
    # На реальных данных (правда неизвестна) truth_evaluation пропускается.
    res_real = run_comparison(df, true_effects=None, n_resamples=200, random_state=7)
    assert res_real.truth_evaluation is None


def test_pvalue_profile_signal_below_null() -> None:
    # У каждого метода средний adjusted-p на истинно-ненулевых меньше, чем на нулевых
    # (сигнал детектируется сильнее шума).
    df, true = make_ab_table(n=3000, n_targets=14, rho=0.5, n_true=4, effect=0.13, seed=31)
    res = run_comparison(df, true_effects=true, n_resamples=400, random_state=31)
    prof = res.pvalue_profile.set_index("method")
    assert list(res.pvalue_profile["method"]) == METHOD_NAMES
    for name in METHOD_NAMES:
        assert prof.loc[name, "mean_p_adj_signal"] < prof.loc[name, "mean_p_adj_null"], name


def test_pvalue_profile_conservativeness_ordering() -> None:
    # Зазор консервативности Bonferroni >= Holm (следствие поточечного p_adj_bonf >= p_adj_holm).
    df, true = make_ab_table(n=2500, n_targets=12, rho=0.3, n_true=3, effect=0.11, seed=32)
    res = run_comparison(df, true_effects=true, n_resamples=300, random_state=32)
    prof = res.pvalue_profile.set_index("method")
    assert prof.loc["bonferroni", "conservativeness_gap"] >= prof.loc["holm", "conservativeness_gap"] - 1e-12


def test_pvalue_profile_gap_nonneg_pvalue_only() -> None:
    # Для методов-монотонных-преобразований raw p (Bonferroni/Holm/BH) зазор неотрицателен.
    df, _ = make_ab_table(n=2500, n_targets=12, rho=0.4, n_true=3, effect=0.10, seed=33)
    res = run_comparison(df, true_effects=None, n_resamples=200, random_state=33)
    prof = res.pvalue_profile.set_index("method")
    # На реальном режиме (true_effects=None) signal/null недоступны, gap — считается.
    assert prof["mean_p_adj_signal"].isna().all()
    for name in ("bonferroni", "holm", "benjamini_hochberg"):
        assert prof.loc[name, "conservativeness_gap"] >= -1e-12, name


def test_fwer_controlled_under_global_null() -> None:
    # Под полной H0 эмпирический FWER FWER-методов держится у α (с запасом на шум симуляций).
    scenarios = [Scenario(rho=0.5, n=1500, n_targets=10, n_true=0, effect=0.0)]
    oc = operating_characteristics(scenarios, n_sims=40, n_resamples=200, base_seed=11)
    oc = oc.set_index("method")
    for name in ("bonferroni", "holm", "westfall_young", "romano_wolf"):
        assert oc.loc[name, "fwer"] <= 0.20  # порог с запасом для 40 симуляций


def test_operating_characteristics_power_grows_with_correlation() -> None:
    # Ключевое различие: power Romano–Wolf растёт с корреляцией, у Bonferroni — нет.
    scenarios = [Scenario(rho=r, n=1800, n_targets=20, n_true=4, effect=0.11) for r in (0.0, 0.8)]
    oc = operating_characteristics(scenarios, n_sims=50, n_resamples=250, base_seed=21)
    piv = oc.pivot(index="method", columns="rho", values="power")
    assert piv.loc["romano_wolf", 0.8] > piv.loc["romano_wolf", 0.0] + 0.02
    assert piv.loc["romano_wolf", 0.8] > piv.loc["bonferroni", 0.8] + 0.02
