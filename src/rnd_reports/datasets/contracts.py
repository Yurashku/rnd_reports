"""Единая обёртка данных для бенчмарка R&D-6.

``BenchmarkDataset`` — формат, близкий к будущей интеграции в HypEx (§8 контекста):
id / treatment / target / признаки + реестр классов безопасности. Один и тот же
интерфейс используют и синтетика, и реальные open-source датасеты.

Фундамент Step 1: контракт + валидация структуры. **Без алгоритмов оценки.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union

import pandas as pd

from ..feature_safety.contracts import FeatureClass, coerce_feature_class
from ..feature_safety.registry import FeatureRegistry, build_feature_registry

RegistryLike = Union[FeatureRegistry, Mapping[str, object]]


@dataclass
class BenchmarkDataset:
    """Обёртка данных эксперимента для бенчмарка снижения дисперсии.

    Параметры:
        data: таблица наблюдений;
        id_col / treatment_col / target_col: ключевые колонки;
        feature_registry: ``FeatureRegistry`` или простой dict ``{name: FeatureClass}``.

    Валидация (в ``__post_init__``): ключевые колонки присутствуют; все признаки
    реестра есть в ``data``.
    """

    data: pd.DataFrame
    id_col: str
    treatment_col: str
    target_col: str
    feature_registry: RegistryLike = field(default_factory=FeatureRegistry)

    def __post_init__(self) -> None:
        if not isinstance(self.feature_registry, FeatureRegistry):
            self.feature_registry = build_feature_registry(self.feature_registry)

        missing_keys = [
            c
            for c in (self.id_col, self.treatment_col, self.target_col)
            if c not in self.data.columns
        ]
        if missing_keys:
            raise ValueError(f"В data отсутствуют ключевые колонки: {missing_keys}")

        missing_feats = [
            n for n in self.feature_registry.names() if n not in self.data.columns
        ]
        if missing_feats:
            raise ValueError(
                f"Признаки реестра отсутствуют в data: {missing_feats}"
            )

    @property
    def n(self) -> int:
        return len(self.data)

    def feature_columns(self) -> list[str]:
        """Все зарегистрированные признаки (в порядке реестра)."""
        return self.feature_registry.names()

    def features_by_class(self, feature_class) -> list[str]:
        """Имена признаков заданного класса безопасности."""
        return self.feature_registry.by_class(coerce_feature_class(feature_class))

    def frame_for(self, feature_classes) -> pd.DataFrame:
        """Подтаблица только из признаков указанных классов (в объявленном порядке)."""
        if isinstance(feature_classes, (str, FeatureClass)):
            feature_classes = [feature_classes]
        wanted = [coerce_feature_class(fc) for fc in feature_classes]
        cols = [
            n
            for n in self.feature_registry.names()
            if self.feature_registry.get(n).feature_class in wanted
        ]
        return self.data[cols]
