"""Steps 5–7: цепочка +B/+C, balance-gate, unsafe_demo и график ATE±CI."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # без дисплея

from rnd_reports.benchmark.protocol import run_benchmark, run_method
from rnd_reports.benchmark.reporting import plot_ate_ci
from rnd_reports.datasets.adapters import make_synthetic_benchmark_dataset
from rnd_reports.feature_safety.diagnostics import balance_gate, diagnose
from rnd_reports.synthetic.scenarios import make_synthetic_scenario

TRUE_ATE = 2.8


def _bds(n=6000, seed=11):
    return make_synthetic_benchmark_dataset(n=n, seed=seed)


def test_chain_variance_reduction_is_monotonic() -> None:
    res = run_benchmark(_bds(), random_state=11)
    by = {r.method: r for r in res}
    a = by["sklearn_cupac_A"].variance_reduction_vs_ab_pct
    ab = by["sklearn_cupac_A_plus_B_linear"].variance_reduction_vs_ab_pct
    abc = by["sklearn_cupac_A_plus_B_plus_C_linear"].variance_reduction_vs_ab_pct
    assert 0 < a < ab < abc
    # инкременты B и C положительны
    assert by["sklearn_cupac_A_plus_B_linear"].incremental_variance_reduction_vs_predecessor_pct > 0
    assert by["sklearn_cupac_A_plus_B_plus_C_linear"].incremental_variance_reduction_vs_predecessor_pct > 0


def test_chain_methods_unbiased() -> None:
    res = run_benchmark(_bds(), random_state=11)
    for name in (
        "sklearn_cupac_A",
        "sklearn_cupac_A_plus_B_linear",
        "sklearn_cupac_A_plus_B_plus_C_linear",
    ):
        r = next(r for r in res if r.method == name)
        assert abs(r.ate - TRUE_ATE) < 0.3


def test_unsafe_demo_is_flagged_and_biased() -> None:
    r = run_method(_bds(), "unsafe_demo_optional", random_state=11)
    assert r.safety_status == "unsafe_demo"
    assert "unsafe_demo" in r.feature_groups_used
    # огромное «снижение дисперсии», но смещённый ATE (демонстрация ловушки)
    r2 = run_benchmark(_bds(), random_state=11)
    demo = next(x for x in r2 if x.method == "unsafe_demo_optional")
    assert demo.variance_reduction_vs_ab_pct > 60
    assert abs(demo.ate - TRUE_ATE) > 0.5


def test_balance_gate_passes_safe_rejects_unsafe() -> None:
    bds = _bds()
    # класс C проходит gate
    passed, _ = balance_gate(
        bds.data, bds.treatment_col, ["x_context_1", "x_context_2"]
    )
    assert set(passed) == {"x_context_1", "x_context_2"}
    # treatment-зависимые признаки gate не проходят
    passed2, _ = balance_gate(
        bds.data, bds.treatment_col, ["x_mediator", "x_session_signal", "x_future_target_sum"]
    )
    assert passed2 == []


def test_diagnose_columns_and_flags() -> None:
    bds = _bds(n=3000)
    rep = diagnose(bds.data, bds.treatment_col, bds.feature_registry)
    assert {"feature", "feature_class", "smd", "gate_pass", "role_consistent"} <= set(rep.columns)
    row = rep.set_index("feature").loc["x_mediator"]
    assert not bool(row["gate_pass"])  # медиатор разбалансирован


def test_plot_ate_ci_returns_axes() -> None:
    res = run_benchmark(_bds(n=2000), random_state=1)
    sc = make_synthetic_scenario(n=10, seed=1)
    ax = plot_ate_ci(res, true_ate=sc.true_ate)
    assert ax is not None
    assert len(ax.get_yticklabels()) == len(res)
