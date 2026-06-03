"""Проверка, что пакет rnd_reports и его модули-заглушки импортируются."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "rnd_reports",
    "rnd_reports.variance_reduction",
    "rnd_reports.variance_reduction.cuped",
    "rnd_reports.variance_reduction.metrics",
    "rnd_reports.variance_reduction.local_cupac",
    "rnd_reports.variance_reduction.hypex_cupac_adapter",
    "rnd_reports.variance_reduction.safe_intime_adjustment",
    "rnd_reports.feature_safety",
    "rnd_reports.feature_safety.contracts",
    "rnd_reports.feature_safety.registry",
    "rnd_reports.feature_safety.rules",
    "rnd_reports.feature_safety.diagnostics",
    "rnd_reports.synthetic",
    "rnd_reports.synthetic.generators",
    "rnd_reports.synthetic.scenarios",
    "rnd_reports.synthetic.schemas",
    "rnd_reports.benchmark",
    "rnd_reports.benchmark.protocol",
    "rnd_reports.benchmark.runners",
    "rnd_reports.benchmark.reporting",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
