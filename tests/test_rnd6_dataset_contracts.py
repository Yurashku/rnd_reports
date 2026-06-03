"""Step 1: контракты данных/методов R&D-6 (без алгоритмов)."""

from __future__ import annotations

import pandas as pd
import pytest

from rnd_reports.benchmark import method_registry as mr
from rnd_reports.benchmark.contracts import RESULT_COLUMNS, MethodResult
from rnd_reports.datasets.contracts import BenchmarkDataset
from rnd_reports.feature_safety.contracts import (
    FORBIDDEN,
    FeatureClass,
)
from rnd_reports.feature_safety.registry import build_feature_registry


# --- пример реестра из §8 контекста ---
EXAMPLE_REGISTRY = {
    "x_pre_1": "A_pre_treatment",
    "x_inflation": "B_expert_safe_intime",
    "x_context_1": "C_balance_gated_intime",
    "x_session_action": "D_dag_required",
    "x_click_after_treatment": "E_mediator_risk",
    "x_future_target_sum": "F_leakage",
}


def _tiny_df() -> pd.DataFrame:
    cols = {"id": [1, 2, 3, 4], "treatment": [0, 1, 0, 1], "target": [1.0, 2.0, 1.5, 2.5]}
    for name in EXAMPLE_REGISTRY:
        cols[name] = [0.1, 0.2, 0.3, 0.4]
    return pd.DataFrame(cols)


def test_build_feature_registry_assigns_classes() -> None:
    reg = build_feature_registry(EXAMPLE_REGISTRY)
    assert len(reg) == 6
    assert reg.by_class(FeatureClass.A_PRE_TREATMENT) == ["x_pre_1"]
    assert reg.by_class("F_leakage") == ["x_future_target_sum"]
    # A/B/C допустимы в estimator; D/E/F — нет
    assert set(reg.usable_in_estimator()) == {"x_pre_1", "x_inflation", "x_context_1"}


def test_benchmark_dataset_valid() -> None:
    ds = BenchmarkDataset(
        data=_tiny_df(),
        id_col="id",
        treatment_col="treatment",
        target_col="target",
        feature_registry=EXAMPLE_REGISTRY,  # dict принимается и коэрсится
    )
    assert ds.n == 4
    assert set(ds.feature_columns()) == set(EXAMPLE_REGISTRY)
    assert ds.features_by_class(FeatureClass.B_EXPERT_SAFE_INTIME) == ["x_inflation"]
    assert list(ds.frame_for("A_pre_treatment").columns) == ["x_pre_1"]


def test_benchmark_dataset_missing_key_column_raises() -> None:
    df = _tiny_df().drop(columns=["treatment"])
    with pytest.raises(ValueError):
        BenchmarkDataset(df, "id", "treatment", "target", {})


def test_benchmark_dataset_missing_feature_raises() -> None:
    with pytest.raises(ValueError):
        BenchmarkDataset(
            _tiny_df(), "id", "treatment", "target", {"x_absent": "A_pre_treatment"}
        )


def test_forbidden_classes() -> None:
    assert FeatureClass.E_MEDIATOR_RISK in FORBIDDEN
    assert FeatureClass.F_LEAKAGE in FORBIDDEN


def test_method_registry_chain_and_roles() -> None:
    names = [m.name for m in mr.METHODS]
    assert set(names) == {
        "ab_hypex",
        "hypex_cupac",
        "sklearn_cupac_A",
        "sklearn_cupac_A_plus_B_linear",
        "sklearn_cupac_A_plus_B_plus_C_linear",
        "unsafe_demo_optional",
    }
    # hypex_cupac — reference, вне цепочки
    assert mr.by_name("hypex_cupac").kind is mr.MethodKind.REFERENCE
    assert mr.by_name("hypex_cupac").predecessor is None
    # unsafe_demo — demo, не кандидат
    assert mr.by_name("unsafe_demo_optional").kind is mr.MethodKind.DEMO
    # predecessor-chain корректна и заканчивается на A+B+C
    chain_names = [m.name for m in mr.chain()]
    assert chain_names == [
        "ab_hypex",
        "sklearn_cupac_A",
        "sklearn_cupac_A_plus_B_linear",
        "sklearn_cupac_A_plus_B_plus_C_linear",
    ]


def test_method_result_row_matches_columns() -> None:
    row = MethodResult(
        method="ab_hypex", dataset_name="synthetic_demo", dataset_type="synthetic"
    ).as_row()
    assert list(row.keys()) == RESULT_COLUMNS
