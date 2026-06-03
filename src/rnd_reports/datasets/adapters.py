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


# Реестр заглушек-адаптеров по кандидатам каталога.
ADAPTERS: dict[str, _NotImplementedAdapter] = {
    name: _NotImplementedAdapter(name)
    for name in ("hillstrom", "lenta", "criteo", "x5_retailhero", "megafon")
}


def get_adapter(name: str) -> _NotImplementedAdapter:
    return ADAPTERS[name]
