"""Генератор синтетической A/B-таблицы с известными истинными эффектами.

Метрики коррелированы (equicorrelation-ковариация, параметр ``rho``) — именно это даёт
различие методов: Westfall–Young / Romano–Wolf эксплуатируют корреляцию и выигрывают в
мощности у Bonferroni/Holm, которые её игнорируют. Сэмплирование коррелированного шума —
через готовый :func:`scipy.stats.multivariate_normal`.

Контракт выхода совпадает с реальной таблицей: ``id, treatment, target_1, ..., target_n`` и
словарь ``true_effects`` (target → истинный аддитивный эффект; 0 — нулевая гипотеза).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def make_equicorrelation_cov(n_targets: int, rho: float, variances: Optional[np.ndarray] = None) -> np.ndarray:
    """Equicorrelation-ковариация с опциональными гетерогенными дисперсиями."""
    if not (-1 / max(n_targets - 1, 1) < rho < 1):
        raise ValueError("rho вне допустимого диапазона положительной определённости матрицы")
    corr = np.full((n_targets, n_targets), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    if variances is None:
        variances = np.ones(n_targets)
    std = np.sqrt(np.asarray(variances, dtype=float))
    return corr * np.outer(std, std)


def make_ab_table(
    n: int = 5_000,
    n_targets: int = 15,
    rho: float = 0.45,
    n_true: int = 3,
    effect: float = 0.10,
    treatment_share: float = 0.5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Синтетическая A/B-таблица ``id, treatment, target_1..n`` с известной правдой.

    Параметры:
        n: число юнитов;
        n_targets: число target-метрик (гипотез);
        rho: попарная корреляция метрик (ключ к различию методов);
        n_true: сколько метрик имеют ненулевой истинный эффект (первые ``n_true``);
        effect: величина аддитивного эффекта у истинно-ненулевых метрик;
        treatment_share: доля treatment-группы;
        seed: фиксированный seed.

    Возвращает ``(df, true_effects)``, где ``true_effects[target]`` — истинный эффект
    (0 для нулевых гипотез).
    """
    if not 0 <= n_true <= n_targets:
        raise ValueError("n_true должен быть в диапазоне [0, n_targets]")
    rng = np.random.default_rng(seed)
    target_cols = [f"target_{i}" for i in range(1, n_targets + 1)]

    true_effects = {col: 0.0 for col in target_cols}
    for col in target_cols[:n_true]:
        true_effects[col] = float(effect)

    treatment = rng.binomial(1, treatment_share, size=n)
    # Небольшая неоднородность дисперсий приближает пример к реальным метрикам.
    variances = np.linspace(0.8, 1.4, n_targets)
    cov = make_equicorrelation_cov(n_targets=n_targets, rho=rho, variances=variances)

    base = stats.multivariate_normal.rvs(mean=np.zeros(n_targets), cov=cov, size=n, random_state=rng)
    base = np.atleast_2d(base).reshape(n, n_targets)
    effects = np.array([true_effects[col] for col in target_cols])
    y = base + treatment[:, None] * effects[None, :]

    df = pd.DataFrame(y, columns=target_cols)
    df.insert(0, "treatment", treatment.astype(int))
    df.insert(0, "id", np.arange(1, n + 1))
    return df, true_effects
