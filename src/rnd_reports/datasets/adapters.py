"""Adapters: «сырой» DataFrame датасета → ``BenchmarkDataset`` с реестром классов.

Синтетика подключается тривиально (классы известны по построению). Реальные датасеты
требуют предобработки и ручной разметки признаков по классам A–F (см. контекст §3);
данные читаются локально из gitignored-директории и **не коммитятся** (см. loaders).

Реализовано: ``SyntheticAdapter`` и ``HillstromAdapter`` (рандомизированный
email-датасет MineThatData). Остальные реальные датасеты — заглушки до своих шагов.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
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


class HillstromAdapter:
    """Адаптер Hillstrom (MineThatData E-Mail) → ``BenchmarkDataset``.

    Рандомизированный email-эксперимент: treatment = было отправлено письмо
    (``segment != 'No E-Mail'``), target = ``spend``. Признаки клиента известны до
    рассылки → класс **A** (числовые + one-hot категориальные). Поведенческие исходы
    ``visit``/``conversion`` — пост-treatment, помечаются классом **E** (mediator-risk) и
    в основной estimator не идут. У датасета нет in-time ковариат (B/C), поэтому здесь
    работает в первую очередь CUPAC по A.
    """

    name = "hillstrom"
    _REQUIRED = (
        "recency", "history", "mens", "womens", "newbie",
        "history_segment", "zip_code", "channel",
        "segment", "spend", "visit", "conversion",
    )
    _NUMERIC_A = ["recency", "history", "mens", "womens", "newbie"]
    _CATEGORICAL_A = ["history_segment", "zip_code", "channel"]

    def to_benchmark_dataset(self, df: pd.DataFrame) -> BenchmarkDataset:
        missing = [c for c in self._REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Hillstrom: отсутствуют ожидаемые колонки: {missing}")

        out = pd.DataFrame()
        out["id"] = np.arange(len(df))
        out["treatment"] = (df["segment"].astype(str) != "No E-Mail").astype(int)
        out["target"] = pd.to_numeric(df["spend"], errors="coerce").astype(float)
        for c in self._NUMERIC_A:
            out[c] = pd.to_numeric(df[c], errors="coerce")

        dummies = pd.get_dummies(
            df[self._CATEGORICAL_A].astype(str), prefix=self._CATEGORICAL_A
        ).astype(int)
        out = pd.concat([out, dummies], axis=1)

        out["visit"] = pd.to_numeric(df["visit"], errors="coerce")
        out["conversion"] = pd.to_numeric(df["conversion"], errors="coerce")

        registry = {c: "A_pre_treatment" for c in self._NUMERIC_A + list(dummies.columns)}
        registry["visit"] = "E_mediator_risk"
        registry["conversion"] = "E_mediator_risk"

        return BenchmarkDataset(
            data=out,
            id_col="id",
            treatment_col="treatment",
            target_col="target",
            feature_registry=registry,
        )


# Реестр адаптеров: реализованные + заглушки по кандидатам каталога.
ADAPTERS: dict[str, object] = {
    "hillstrom": HillstromAdapter(),
    **{
        name: _NotImplementedAdapter(name)
        for name in ("lenta", "criteo", "x5_retailhero", "megafon")
    },
}


def get_adapter(name: str):
    return ADAPTERS[name]


def load_benchmark_dataset(name: str, path=None) -> BenchmarkDataset:
    """Загрузить локальный датасет и привести к ``BenchmarkDataset``.

    Связка loader → adapter. Данные читаются из gitignored-директории
    (``loaders.DEFAULT_DATA_DIR``); при отсутствии файла loader бросает понятную ошибку.
    """
    from .loaders import get_loader

    df = get_loader(name).load(path)
    return get_adapter(name).to_benchmark_dataset(df)
