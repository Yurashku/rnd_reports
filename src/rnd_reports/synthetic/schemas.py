"""Схемы синтетических сценариев R&D-6 (Step 2).

Определяет:
- ``SYNTHETIC_FEATURE_CLASSES`` — отображение колонок канонического синтетического
  сценария на классы безопасности A–F + unsafe_demo (единый источник для генератора
  и адаптера);
- ``SyntheticScenario`` — результат генерации: данные, разметка признаков, известный
  истинный ATE и имена ключевых колонок.

Только синтетика; реальные данные не используются. Алгоритмы оценки — последующие шаги.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..feature_safety.contracts import FeatureClass

ID_COL = "id"
TREATMENT_COL = "treatment"
TARGET_COL = "target"

# Канонический синтетический сценарий: колонка -> класс безопасности.
# (см. docs/06_safe_intime_cupac_context.md §3–4)
SYNTHETIC_FEATURE_CLASSES: dict[str, FeatureClass] = {
    # A — pre-treatment (идут в CUPAC)
    "y_lag1": FeatureClass.A_PRE_TREATMENT,
    "y_lag2": FeatureClass.A_PRE_TREATMENT,
    "x_pre_1_lag1": FeatureClass.A_PRE_TREATMENT,
    "x_pre_1_lag2": FeatureClass.A_PRE_TREATMENT,
    "x_pre_2_lag1": FeatureClass.A_PRE_TREATMENT,
    "x_pre_2_lag2": FeatureClass.A_PRE_TREATMENT,
    # B — expert-safe in-time (внешний контекст, ⟂ treatment)
    "x_inflation": FeatureClass.B_EXPERT_SAFE_INTIME,
    "x_weather": FeatureClass.B_EXPERT_SAFE_INTIME,
    # C — balance-gated in-time (неочевидные, но сбалансированные/безопасные)
    "x_context_1": FeatureClass.C_BALANCE_GATED_INTIME,
    "x_context_2": FeatureClass.C_BALANCE_GATED_INTIME,
    # D — DAG-required (неоднозначные; в estimator не входят)
    "x_session_signal": FeatureClass.D_DAG_REQUIRED,
    # E — mediator-risk (следствие treatment, влияет на исход)
    "x_mediator": FeatureClass.E_MEDIATOR_RISK,
    # F — leakage (производное от исхода)
    "x_future_target_sum": FeatureClass.F_LEAKAGE,
    # unsafe_demo (только демонстрация риска)
    "x_unsafe_demo": FeatureClass.UNSAFE_DEMO,
}


@dataclass
class SyntheticScenario:
    """Результат генерации синтетического сценария с известной истиной."""

    data: pd.DataFrame
    feature_registry: dict[str, FeatureClass]
    true_ate: float
    id_col: str = ID_COL
    treatment_col: str = TREATMENT_COL
    target_col: str = TARGET_COL
    params: dict = field(default_factory=dict)
