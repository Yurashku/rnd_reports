"""Adapters: «сырой» DataFrame датасета → ``BenchmarkDataset`` с реестром классов.

Заглушки Step 1: контракт + per-dataset болванки (``NotImplementedError`` с TODO).
Каждый реальный датасет требует своей предобработки и разметки признаков по классам
A–F (см. контекст §3) — это реализуется отдельными шагами, по одному датасету.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from .contracts import BenchmarkDataset


@runtime_checkable
class DatasetAdapter(Protocol):
    """Контракт адаптера: привести сырой DataFrame к ``BenchmarkDataset``."""

    name: str

    def to_benchmark_dataset(self, df: pd.DataFrame) -> BenchmarkDataset: ...


class _NotImplementedAdapter:
    """Базовая заглушка адаптера для реального датасета (реализация — позже)."""

    def __init__(self, name: str):
        self.name = name

    def to_benchmark_dataset(self, df: pd.DataFrame) -> BenchmarkDataset:
        raise NotImplementedError(
            f"Адаптер для '{self.name}' ещё не реализован (Step 1 — только контракт). "
            "Нужна предобработка и разметка признаков по классам A–F; "
            "см. docs/06_safe_intime_cupac_implementation_plan.md."
        )


class SyntheticAdapter:
    """Тривиальный адаптер для синтетики: ``SyntheticScenario`` → ``BenchmarkDataset``.

    В отличие от реальных датасетов, классы признаков на синтетике известны по
    построению, поэтому адаптер просто переносит данные и готовую разметку в
    единый формат ``BenchmarkDataset``.
    """

    name = "synthetic"

    def to_benchmark_dataset(self, scenario) -> BenchmarkDataset:
        return BenchmarkDataset(
            data=scenario.data,
            id_col=scenario.id_col,
            treatment_col=scenario.treatment_col,
            target_col=scenario.target_col,
            feature_registry=scenario.feature_registry,
        )


def make_synthetic_benchmark_dataset(**kwargs) -> BenchmarkDataset:
    """Сгенерировать синтетику и сразу обернуть в ``BenchmarkDataset``.

    Все ``kwargs`` пробрасываются в
    :func:`rnd_reports.synthetic.scenarios.make_synthetic_scenario`.
    """
    # Локальный импорт, чтобы избежать жёсткой связи datasets→synthetic на уровне модуля.
    from ..synthetic.scenarios import make_synthetic_scenario

    scenario = make_synthetic_scenario(**kwargs)
    return SyntheticAdapter().to_benchmark_dataset(scenario)


# Реестр заглушек-адаптеров по кандидатам каталога.
ADAPTERS: dict[str, _NotImplementedAdapter] = {
    name: _NotImplementedAdapter(name)
    for name in ("hillstrom", "lenta", "criteo", "x5_retailhero", "megafon")
}


def get_adapter(name: str) -> _NotImplementedAdapter:
    return ADAPTERS[name]
