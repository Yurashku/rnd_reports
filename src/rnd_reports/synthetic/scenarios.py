"""Генератор синтетических сценариев R&D-6 (Step 2).

Минимальный, чистый и воспроизводимый DGP с **известным истинным ATE** и
признаками всех классов безопасности A–F + unsafe_demo. На синтетике мы заранее
знаем роль каждого признака — это нужно, чтобы проверять, что B/C дают выигрыш,
а E/F/unsafe_demo не должны попадать в основной estimator (последующие шаги).

DGP (схематично):
    treatment ~ Bernoulli(0.5)                                   # рандомизация
    base_A = a·(pre-treatment лаги)                              # предсказуемо из A
    y = base_A + sB·b_latent + sC·c_latent
          + tau·treatment + c·x_mediator + eps                   # истинный исход
    x_mediator = b·treatment + ε                                 # E (медиатор)
    x_inflation/x_weather ≈ b_latent  (⟂ treatment)              # B (expert-safe in-time)
    x_context_1/2        ≈ c_latent   (⟂ treatment)              # C (balance-gated in-time)
    x_session_signal      = d·treatment + corr + ε               # D (неоднозначный)
    x_future_target_sum   = y + ε                                # F (leakage)
    x_unsafe_demo         = y + tiny ε                            # unsafe_demo
    ИСТИННЫЙ ATE = tau + c·b

Только синтетика; алгоритмы оценки — последующие шаги.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import (
    SYNTHETIC_FEATURE_CLASSES,
    SyntheticScenario,
)


def _ar_lags(rng: np.random.Generator, n: int, rho: float, sigma: float = 1.0):
    """Два лага AR(1)-процесса (lag2, lag1) с корреляцией rho."""
    lag2 = rng.normal(0.0, sigma, n)
    lag1 = rho * lag2 + np.sqrt(max(0.0, 1.0 - rho**2)) * rng.normal(0.0, sigma, n)
    return lag2, lag1


def make_synthetic_scenario(
    n: int = 4000,
    seed: int = 42,
    *,
    effect_size: float = 2.0,
    pre_strength: float = 1.0,
    b_strength: float = 1.2,
    c_strength: float = 1.0,
    mediator_b: float = 1.0,
    mediator_c: float = 0.8,
    noise_sd: float = 1.0,
    rho: float = 0.7,
) -> SyntheticScenario:
    """Сгенерировать канонический синтетический сценарий R&D-6.

    Возвращает :class:`SyntheticScenario` с данными, разметкой признаков по классам
    A–F + unsafe_demo и известным истинным ATE (``effect_size + mediator_c*mediator_b``).
    """
    rng = np.random.default_rng(seed)

    # --- A: pre-treatment лаги ---
    x1_lag2, x1_lag1 = _ar_lags(rng, n, rho)
    x2_lag2, x2_lag1 = _ar_lags(rng, n, rho)
    y_lag2, y_lag1 = _ar_lags(rng, n, rho, sigma=1.5)
    base_a = pre_strength * (0.8 * x1_lag1 + 0.6 * x2_lag1 + 0.7 * y_lag1)

    # --- рандомизированное назначение ---
    treatment = rng.binomial(1, 0.5, n)

    # --- латенты, объясняемые in-time признаками (⟂ treatment) ---
    b_latent = rng.normal(0.0, 1.0, n)  # объясняется B
    c_latent = rng.normal(0.0, 1.0, n)  # объясняется C

    # B — expert-safe in-time (внешний контекст), не зависит от treatment
    x_inflation = b_latent + rng.normal(0.0, 0.5, n)
    x_weather = b_latent + rng.normal(0.0, 0.5, n)

    # C — balance-gated in-time, не зависит от treatment, но «неочевидный»
    x_context_1 = c_latent + rng.normal(0.0, 0.5, n)
    x_context_2 = c_latent + rng.normal(0.0, 0.5, n)

    # E — медиатор: следствие treatment, влияет на исход
    x_mediator = mediator_b * treatment + rng.normal(0.0, 1.0, n)

    # идиосинкратический шум исхода
    eps = rng.normal(0.0, noise_sd, n)

    target = (
        base_a
        + b_strength * b_latent
        + c_strength * c_latent
        + effect_size * treatment
        + mediator_c * x_mediator
        + eps
    )
    true_ate = float(effect_size + mediator_c * mediator_b)

    # D — DAG-required: неоднозначный in-time сигнал (слабо зависит от treatment)
    x_session_signal = (
        0.5 * treatment + 0.5 * c_latent + rng.normal(0.0, 1.0, n)
    )

    # F — leakage: производное от реализованного исхода
    x_future_target_sum = target + rng.normal(0.0, 1.0, n)

    # unsafe_demo: почти копия исхода (демонстрация искусственного «выигрыша»)
    x_unsafe_demo = target + rng.normal(0.0, 0.1, n)

    data = pd.DataFrame(
        {
            "id": np.arange(n),
            "treatment": treatment,
            "target": target,
            "y_lag1": y_lag1,
            "y_lag2": y_lag2,
            "x_pre_1_lag1": x1_lag1,
            "x_pre_1_lag2": x1_lag2,
            "x_pre_2_lag1": x2_lag1,
            "x_pre_2_lag2": x2_lag2,
            "x_inflation": x_inflation,
            "x_weather": x_weather,
            "x_context_1": x_context_1,
            "x_context_2": x_context_2,
            "x_session_signal": x_session_signal,
            "x_mediator": x_mediator,
            "x_future_target_sum": x_future_target_sum,
            "x_unsafe_demo": x_unsafe_demo,
        }
    )

    return SyntheticScenario(
        data=data,
        feature_registry=dict(SYNTHETIC_FEATURE_CLASSES),
        true_ate=true_ate,
        params={
            "n": n,
            "seed": seed,
            "effect_size": effect_size,
            "pre_strength": pre_strength,
            "b_strength": b_strength,
            "c_strength": c_strength,
            "mediator_b": mediator_b,
            "mediator_c": mediator_c,
            "noise_sd": noise_sd,
            "rho": rho,
        },
    )
