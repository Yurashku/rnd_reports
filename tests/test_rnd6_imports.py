"""Step 1: импортируемость фундамент-модулей R&D-6."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "rnd_reports.datasets",
    "rnd_reports.datasets.contracts",
    "rnd_reports.datasets.catalog",
    "rnd_reports.datasets.loaders",
    "rnd_reports.datasets.adapters",
    "rnd_reports.feature_safety.contracts",
    "rnd_reports.feature_safety.registry",
    "rnd_reports.benchmark.contracts",
    "rnd_reports.benchmark.method_registry",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
