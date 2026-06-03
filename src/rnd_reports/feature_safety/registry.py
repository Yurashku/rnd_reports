"""Реестр признаков и их классов безопасности для R&D-6.

Фундамент Step 1: минимальная структура-контракт (хранит ``name → FeatureClass``)
и удобный конструктор из простого словаря (§8 контекста) или списка ``FeatureSpec``.
**Без алгоритмов** — гейты/диагностики появятся в последующих шагах.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Union

from .contracts import (
    FeatureClass,
    FeatureSpec,
    coerce_feature_class,
)

RegistryInput = Union[Mapping[str, object], Iterable[FeatureSpec]]


class FeatureRegistry:
    """Реестр ``FeatureSpec`` с доступом по имени и по классу безопасности."""

    def __init__(self, specs: Iterable[FeatureSpec] | None = None):
        self._specs: dict[str, FeatureSpec] = {}
        for spec in specs or []:
            self.add(spec)

    def add(self, spec: FeatureSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[FeatureSpec]:
        return list(self._specs.values())

    def by_class(self, feature_class) -> list[str]:
        """Имена признаков заданного класса (в порядке добавления)."""
        fc = coerce_feature_class(feature_class)
        return [n for n, s in self._specs.items() if s.feature_class == fc]

    def classes(self) -> dict[str, FeatureClass]:
        """Отображение имя → класс."""
        return {n: s.feature_class for n, s in self._specs.items()}

    def usable_in_estimator(self) -> list[str]:
        """Признаки классов, допустимых в основном estimator (A/B/C)."""
        return [n for n, s in self._specs.items() if s.usable_in_estimator]


def build_feature_registry(spec: RegistryInput) -> FeatureRegistry:
    """Собрать реестр из словаря ``{name: feature_class}`` или списка ``FeatureSpec``.

    Пример (см. §8 контекста)::

        build_feature_registry({
            "x_pre_1": "A_pre_treatment",
            "x_inflation": "B_expert_safe_intime",
            "x_future_target_sum": "F_leakage",
        })
    """
    registry = FeatureRegistry()
    if isinstance(spec, Mapping):
        for name, feature_class in spec.items():
            registry.add(FeatureSpec(name=name, feature_class=feature_class))
    else:
        for item in spec:
            if not isinstance(item, FeatureSpec):
                raise TypeError(
                    "Ожидался FeatureSpec или Mapping[str, FeatureClass], "
                    f"получено: {type(item)!r}"
                )
            registry.add(item)
    return registry
