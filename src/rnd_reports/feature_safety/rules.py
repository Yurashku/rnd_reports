"""Практический safety-gate для balance-gated in-time признаков (класс C).

R&D-6, Step 6. Декларативные пороги + решение «прошёл ли признак gate». Gate —
практический фильтр (после исключения очевидных E/F), а не доказательство causal-
безопасности (см. docs/feature_safety_policy.md §3C).

Минимальный gate: признак считается прошедшим, если он **сбалансирован** между
treatment/control (малый |SMD|) и не имеет treatment-зависимой доли пропусков.
"""

from __future__ import annotations

# Пороги gate (консервативные значения по умолчанию).
MAX_ABS_SMD = 0.1  # |стандартизованная разность средних|
MAX_MISSINGNESS_DIFF = 0.02  # |разность долей пропусков между группами|

# Историческая заглушка (оставлена для обратной совместимости импорта).
SAFETY_RULES: list = []


def passes_balance_gate(
    smd: float,
    missingness_diff: float = 0.0,
    *,
    max_abs_smd: float = MAX_ABS_SMD,
    max_missingness_diff: float = MAX_MISSINGNESS_DIFF,
) -> bool:
    """Проходит ли признак balance/missingness gate.

    Безопасным считаем сбалансированный по среднему и по пропускам признак.
    """
    if smd is None:
        return False
    return abs(smd) <= max_abs_smd and abs(missingness_diff) <= max_missingness_diff
