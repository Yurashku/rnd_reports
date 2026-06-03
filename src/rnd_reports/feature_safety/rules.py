"""TODO: правила проверки безопасности ковариат (отсутствие утечки назначения и т.п.).

Заглушка для R&D-6 «Safe in-time covariates для CUPAC».
Алгоритмы не реализованы — см. docs/feature_safety_policy.md.
"""

from __future__ import annotations

# TODO: наполнить набором правил безопасности; пока пусто.
SAFETY_RULES: list = []


def check(*args, **kwargs):
    """TODO: применить правила безопасности к ковариате. Пока не реализовано."""
    raise NotImplementedError(
        "feature_safety.rules.check ещё не реализован (R&D-6, заглушка)."
    )
