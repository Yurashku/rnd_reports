"""Реестр методов бенчмарка R&D-6 и их predecessor-chain (декларативно).

Фундамент Step 1: описание шести методов (§5 контекста), их ролей, цепочки
предшественников и ожидаемого статуса безопасности. **Без алгоритмов.**

Цепочка прироста (predecessor-chain):
    ab_hypex → sklearn_cupac_A → sklearn_cupac_A_plus_B_linear
             → sklearn_cupac_A_plus_B_plus_C_linear
Исключения:
    hypex_cupac        — reference/parity, вне цепочки (predecessor=None);
    unsafe_demo_optional — демонстрация риска, не кандидат к использованию.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..feature_safety.contracts import FeatureClass
from .contracts import SAFETY_OK, SAFETY_REFERENCE, SAFETY_UNSAFE_DEMO


class MethodKind(str, Enum):
    AB_BASELINE = "ab_baseline"  # A/B без снижения дисперсии
    REFERENCE = "reference"  # parity-сверка (вне цепочки)
    CHAIN = "chain"  # звено основной цепочки прироста
    DEMO = "demo"  # демонстрация риска, не кандидат

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class MethodSpec:
    """Описание одного метода бенчмарка."""

    name: str
    kind: MethodKind
    predecessor: str | None = None
    feature_classes: tuple[FeatureClass, ...] = ()
    expected_safety_status: str = SAFETY_OK
    notes: str = ""


A = FeatureClass.A_PRE_TREATMENT
B = FeatureClass.B_EXPERT_SAFE_INTIME
C = FeatureClass.C_BALANCE_GATED_INTIME
UNSAFE = FeatureClass.UNSAFE_DEMO

METHODS: list[MethodSpec] = [
    MethodSpec(
        name="ab_hypex",
        kind=MethodKind.AB_BASELINE,
        predecessor=None,
        feature_classes=(),
        notes="A/B baseline без variance reduction (HypEx).",
    ),
    MethodSpec(
        name="hypex_cupac",
        kind=MethodKind.REFERENCE,
        predecessor=None,  # reference_only — вне predecessor-chain
        feature_classes=(A,),
        expected_safety_status=SAFETY_REFERENCE,
        notes="Reference/parity CUPAC из HypEx; не входит в цепочку прироста.",
    ),
    MethodSpec(
        name="sklearn_cupac_A",
        kind=MethodKind.CHAIN,
        predecessor="ab_hypex",
        feature_classes=(A,),
        notes="Основной local/scikit CUPAC baseline (класс A).",
    ),
    MethodSpec(
        name="sklearn_cupac_A_plus_B_linear",
        kind=MethodKind.CHAIN,
        predecessor="sklearn_cupac_A",
        feature_classes=(A, B),
        notes="CUPAC A + линейная коррекция по expert-safe in-time (B).",
    ),
    MethodSpec(
        name="sklearn_cupac_A_plus_B_plus_C_linear",
        kind=MethodKind.CHAIN,
        predecessor="sklearn_cupac_A_plus_B_linear",
        feature_classes=(A, B, C),
        notes="CUPAC A+B + линейная коррекция по balance-gated in-time (C).",
    ),
    MethodSpec(
        name="unsafe_demo_optional",
        kind=MethodKind.DEMO,
        predecessor=None,
        feature_classes=(A, B, C, UNSAFE),
        expected_safety_status=SAFETY_UNSAFE_DEMO,
        notes="Демонстрация риска: добавляет unsafe-признаки; НЕ кандидат к использованию.",
    ),
]

_BY_NAME: dict[str, MethodSpec] = {m.name: m for m in METHODS}


def by_name(name: str) -> MethodSpec:
    return _BY_NAME[name]


def chain() -> list[MethodSpec]:
    """Методы основной predecessor-chain в порядке от A/B baseline до A+B+C."""
    ordered: list[MethodSpec] = []
    # стартуем с ab-baseline, затем идём по предшественникам
    current = next(m for m in METHODS if m.kind is MethodKind.AB_BASELINE)
    ordered.append(current)
    changed = True
    while changed:
        changed = False
        for m in METHODS:
            if m.kind is MethodKind.CHAIN and m.predecessor == ordered[-1].name:
                ordered.append(m)
                changed = True
                break
    return ordered
