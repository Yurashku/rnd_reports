"""Тесты портированного из VarWar локального CUPAC (CUPACTransformer)."""

from __future__ import annotations

import importlib.util

import pytest
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from rnd_reports.synthetic.generators import DataGenerator
from rnd_reports.variance_reduction.local_cupac import CUPACTransformer

_SKLEARN_MODELS = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.01, max_iter=10000),
}


def _fitted_transformer():
    df = DataGenerator(n_samples=1500, seed=11).generate()
    t = CUPACTransformer(
        target_col="y", n_folds=3, random_state=11, models=dict(_SKLEARN_MODELS)
    )
    return t, t.fit_transform(df)


def test_fit_transform_adds_cupac_column() -> None:
    t, out = _fitted_transformer()
    assert t.is_fitted
    assert "y_cupac" in out.columns


def test_variance_reduction_is_non_negative() -> None:
    t, _ = _fitted_transformer()
    assert t.variance_reduction is not None
    assert t.variance_reduction >= 0.0
    assert t.best_model_name in _SKLEARN_MODELS


def test_report_renders_after_fit() -> None:
    t, _ = _fitted_transformer()
    report = t.get_report()
    assert "CUPAC Report" in report
    assert "Снижение дисперсии" in report


@pytest.mark.skipif(
    importlib.util.find_spec("catboost") is None,
    reason="CatBoost не установлен (опциональная зависимость)",
)
def test_default_models_include_catboost_when_available() -> None:
    t = CUPACTransformer(target_col="y")
    assert "CatBoost" in t.models
