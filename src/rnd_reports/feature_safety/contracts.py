"""Контракты безопасности признаков для R&D-6 (Safe in-time CUPAC).

Фундамент Step 1: декларативные контракты (классы признаков A–F + unsafe_demo,
``FeatureSpec``, политики использования). **Без алгоритмов** — правила-гейты и
диагностики появятся в последующих шагах (см.
``docs/06_safe_intime_cupac_implementation_plan.md``).

Классы признаков (см. docs/06_safe_intime_cupac_context.md §3–4):
- ``A_pre_treatment``       — до назначения; используется в CUPAC;
- ``B_expert_safe_intime``  — in-time, экспертно безопасные; linear second-stage;
- ``C_balance_gated_intime``— in-time, допускаются после balance/missingness gate; linear second-stage;
- ``D_dag_required``        — требуют causal-проверки; в estimator не входят;
- ``E_mediator_risk``       — возможный медиатор; запрещены для основного ATE;
- ``F_leakage``             — производные от исхода/leakage; запрещены;
- ``unsafe_demo``           — только для демонстрации риска; никогда не кандидат.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FeatureContract:
    """Заглушка-контракт безопасной ковариаты (исторический placeholder).

    Сохранён для обратной совместимости; новый контракт — :class:`FeatureSpec`.
    """

    ...


class FeatureClass(str, Enum):
    """Класс признака по его допустимости в снижении дисперсии."""

    A_PRE_TREATMENT = "A_pre_treatment"
    B_EXPERT_SAFE_INTIME = "B_expert_safe_intime"
    C_BALANCE_GATED_INTIME = "C_balance_gated_intime"
    D_DAG_REQUIRED = "D_dag_required"
    E_MEDIATOR_RISK = "E_mediator_risk"
    F_LEAKAGE = "F_leakage"
    UNSAFE_DEMO = "unsafe_demo"

    def __str__(self) -> str:
        return self.value


# Стадии использования признака в цепочке методов.
STAGE_NONE = "none"
STAGE_CUPAC = "cupac"  # признак идёт в CUPAC-предсказание (класс A)
STAGE_LINEAR_SECOND_STAGE = "linear_second_stage"  # линейная коррекция (классы B, C)

# Декларативная политика (без логики; гейты/проверки — последующие шаги).
# Где класс допустимо использовать в основном ATE-estimator.
USAGE_STAGE: dict[FeatureClass, str] = {
    FeatureClass.A_PRE_TREATMENT: STAGE_CUPAC,
    FeatureClass.B_EXPERT_SAFE_INTIME: STAGE_LINEAR_SECOND_STAGE,
    FeatureClass.C_BALANCE_GATED_INTIME: STAGE_LINEAR_SECOND_STAGE,
    FeatureClass.D_DAG_REQUIRED: STAGE_NONE,
    FeatureClass.E_MEDIATOR_RISK: STAGE_NONE,
    FeatureClass.F_LEAKAGE: STAGE_NONE,
    FeatureClass.UNSAFE_DEMO: STAGE_NONE,
}

# Классы, допустимые в основном estimator (цепочка A→A+B→A+B+C).
USABLE_IN_ESTIMATOR: tuple[FeatureClass, ...] = (
    FeatureClass.A_PRE_TREATMENT,
    FeatureClass.B_EXPERT_SAFE_INTIME,
    FeatureClass.C_BALANCE_GATED_INTIME,
)
# Исключены из текущего estimator, но не запрещены концептуально (нужен DAG).
EXCLUDED_FROM_ESTIMATOR: tuple[FeatureClass, ...] = (FeatureClass.D_DAG_REQUIRED,)
# Жёстко запрещены для основного ATE-adjustment.
FORBIDDEN: tuple[FeatureClass, ...] = (
    FeatureClass.E_MEDIATOR_RISK,
    FeatureClass.F_LEAKAGE,
)
# Только демонстрация риска; никогда не кандидат к использованию.
DEMO_ONLY: tuple[FeatureClass, ...] = (FeatureClass.UNSAFE_DEMO,)


def coerce_feature_class(value) -> FeatureClass:
    """Привести строку/Enum к ``FeatureClass`` (принимает значения вида ``A_pre_treatment``)."""
    if isinstance(value, FeatureClass):
        return value
    return FeatureClass(str(value))


@dataclass
class FeatureSpec:
    """Контракт одного признака: имя, класс безопасности и заметка."""

    name: str
    feature_class: FeatureClass
    notes: str = ""

    def __post_init__(self) -> None:
        self.feature_class = coerce_feature_class(self.feature_class)

    @property
    def usage_stage(self) -> str:
        return USAGE_STAGE[self.feature_class]

    @property
    def usable_in_estimator(self) -> bool:
        return self.feature_class in USABLE_IN_ESTIMATOR

    @property
    def forbidden(self) -> bool:
        return self.feature_class in FORBIDDEN

    @property
    def demo_only(self) -> bool:
        return self.feature_class in DEMO_ONLY
