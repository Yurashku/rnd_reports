"""Проверка контракта-заглушки подпакета feature_safety.

Алгоритмы не реализованы: символы должны существовать, а функции-заглушки —
явно поднимать NotImplementedError. Это фиксирует «ещё не реализовано».
"""

from __future__ import annotations

import pytest

from rnd_reports.feature_safety import contracts, diagnostics, registry, rules


def test_placeholder_symbols_exist() -> None:
    assert isinstance(contracts.FeatureContract, type)
    assert isinstance(registry.FeatureRegistry, type)
    assert rules.SAFETY_RULES == []


def test_rules_check_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        rules.check()


def test_diagnostics_diagnose_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        diagnostics.diagnose()
