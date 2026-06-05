"""Синтетический наблюдательный сценарий R&D-7 (numpy/pandas, без pyspark).

Генерирует данные со схемой эмбеддингов (``epk_id, report_dt, col_*``) + ``treatment`` и
``outcome`` с **известным** ATE. Назначение трита НЕ рандомизировано: зависит от латентных
конфаундеров через эмбеддинги (селекция). Эмбеддинги покрывают конфаундеры, поэтому поправка
на них должна убирать смещение наивной оценки.

Используется ноутбуком rnd/07 и тестами; ground-truth ATE известен по построению.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EmbeddingScenario:
    data: pd.DataFrame              # epk_id, report_dt, col_000.., treatment, outcome
    true_ate: float
    embedding_columns: list[str]
    params: dict = field(default_factory=dict)


def make_embedding_observational_scenario(
    n: int = 4000,
    k: int = 8,
    *,
    n_latent: int = 3,
    true_ate: float = 3.0,
    n_months: int = 6,
    confounding: float = 1.5,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> EmbeddingScenario:
    """Сгенерировать наблюдательный сценарий «эмбеддинги как adjustment set».

    - латентные конфаундеры ``z`` (n × n_latent) ~ N(0, 1);
    - эмбеддинги ``col_0..col_{k-1}`` = ``z @ W + шум`` (несут информацию о конфаундерах);
    - ``treatment ~ Bernoulli(sigmoid(confounding · z·a))`` — селекция, не рандом;
    - ``outcome = true_ate·T + z·b + ε`` (конфаундеры влияют и на трит, и на исход);
    - ``report_dt`` — равномерно по ``n_months`` месячным срезам (для in-time демо).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_latent))

    # Эмбеддинги: зашумлённая линейная проекция конфаундеров (k >= n_latent желательно).
    w = rng.standard_normal((n_latent, k))
    emb = z @ w + rng.standard_normal((n, k)) * 0.5

    # Назначение трита: логистическая селекция по конфаундерам.
    a = rng.standard_normal(n_latent)
    logits = confounding * (z @ a)
    p_treat = 1.0 / (1.0 + np.exp(-logits))
    treatment = (rng.uniform(size=n) < p_treat).astype(int)

    # Исход: эффект трита + влияние конфаундеров + шум.
    b = rng.standard_normal(n_latent)
    outcome = true_ate * treatment + z @ b + rng.standard_normal(n) * noise_sd

    emb_cols = [f"col_{i:03d}" for i in range(k)]
    df = pd.DataFrame(emb, columns=emb_cols)
    df.insert(0, "epk_id", np.arange(n))
    # Месячные срезы report_dt (строки YYYY-MM-01).
    months = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    df["report_dt"] = months[rng.integers(0, n_months, size=n)].strftime("%Y-%m-%d")
    df["treatment"] = treatment
    df["outcome"] = outcome

    return EmbeddingScenario(
        data=df,
        true_ate=float(true_ate),
        embedding_columns=emb_cols,
        params=dict(n=n, k=k, n_latent=n_latent, n_months=n_months,
                    confounding=confounding, noise_sd=noise_sd, seed=seed),
    )
