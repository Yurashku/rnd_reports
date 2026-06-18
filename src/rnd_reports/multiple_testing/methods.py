"""Реестр методов множественного тестирования (OR-регион + один FDR-ориентир).

Все методы отвечают на OR-вопрос «есть ли в семействе target-метрик хотя бы один
надёжный сигнал и какие именно метрики его дали»:

- **Bonferroni**, **Holm** — контроль FWER, только по вектору raw p-value
  (готовый :func:`statsmodels.stats.multitest.multipletests`);
- **Westfall–Young maxT**, **Romano–Wolf stepdown** — контроль FWER c учётом
  корреляции метрик через перестановки treatment-лейблов (resampling);
- **Benjamini–Hochberg** — единственный FDR-ориентир (готовый
  :func:`scipy.stats.false_discovery_control`), как exploratory-компаратор.

Перестановочные методы свои только там, где готового нет (Romano–Wolf stepdown);
Westfall–Young и BH опираются на scipy/statsmodels.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .elementary import compute_vectorized_t_stats

# Порядок методов в таблицах/графиках: FWER-семейство, затем FDR-ориентир.
METHOD_NAMES: List[str] = [
    "bonferroni",
    "holm",
    "westfall_young",
    "romano_wolf",
    "benjamini_hochberg",
]

# Какой метод какой ошибкой управляет — для отчётности и группировки.
METHOD_ERROR: Dict[str, str] = {
    "bonferroni": "FWER",
    "holm": "FWER",
    "westfall_young": "FWER",
    "romano_wolf": "FWER",
    "benjamini_hochberg": "FDR",
}


def _build_permutation_t_statistics(
    df: pd.DataFrame,
    target_cols: List[str],
    treatment_col: str,
    n_resamples: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Наблюдаемые и перестановочные t-статистики семейства.

    Возвращает ``(t_obs[n_targets], t_perm[n_resamples, n_targets])``. Перестановка
    treatment-лейблов — естественная нулевая модель для рандомизированного эксперимента
    (совместное распределение статистик семейства сохраняет корреляцию метрик).
    """
    rng = np.random.default_rng(random_state)
    y = df[target_cols].to_numpy(dtype=float)
    treatment = df[treatment_col].to_numpy(dtype=int)

    t_obs = compute_vectorized_t_stats(y, treatment)
    t_perm = np.empty((n_resamples, len(target_cols)), dtype=float)
    for b in range(n_resamples):
        t_perm[b] = compute_vectorized_t_stats(y, rng.permutation(treatment))
    return t_obs, t_perm


def westfall_young_maxT_adjusted_pvalues(t_obs: np.ndarray, t_perm: np.ndarray) -> np.ndarray:
    """Single-step Westfall–Young maxT adjusted p-values (двусторонние тесты).

    Калибруется по распределению максимума |t| по всему семейству — отсюда контроль
    FWER с учётом зависимости метрик. Сглаживание ``+1`` исключает нулевые p-value при
    конечном числе перестановок.
    """
    abs_obs = np.abs(np.asarray(t_obs, dtype=float))
    abs_perm = np.abs(np.asarray(t_perm, dtype=float))
    max_perm = np.max(abs_perm, axis=1)
    p_adj = (1 + np.sum(max_perm[None, :] >= abs_obs[:, None], axis=1)) / (len(max_perm) + 1)
    return np.minimum(p_adj, 1.0)


def romano_wolf_stepdown_adjusted_pvalues(t_obs: np.ndarray, t_perm: np.ndarray) -> np.ndarray:
    """Romano–Wolf stepdown adjusted p-values (двусторонние тесты).

    Гипотезы сортируются по убыванию |t|. На шаге k референс — максимум |T| по ещё
    активному (сужающемуся) множеству гипотез, поэтому процедура мощнее single-step
    Westfall–Young при сохранении контроля FWER. Готового решения в scipy нет.
    """
    abs_obs = np.abs(np.asarray(t_obs, dtype=float))
    abs_perm = np.abs(np.asarray(t_perm, dtype=float))
    n_hyp = abs_obs.size
    n_perm = abs_perm.shape[0]

    order = np.argsort(-abs_obs)  # самый сильный сигнал — первым
    raw_step_p = np.empty(n_hyp, dtype=float)
    for k, hyp_idx in enumerate(order):
        active = order[k:]
        max_active_perm = np.max(abs_perm[:, active], axis=1)
        raw_step_p[k] = (1 + np.sum(max_active_perm >= abs_obs[hyp_idx])) / (n_perm + 1)

    # Монотонность вдоль stepdown-порядка.
    adj_ordered = np.maximum.accumulate(raw_step_p)
    p_adj = np.empty(n_hyp, dtype=float)
    p_adj[order] = np.minimum(adj_ordered, 1.0)
    return p_adj


def add_corrections(
    tests_df: pd.DataFrame,
    df: pd.DataFrame,
    target_cols: List[str],
    treatment_col: str = "treatment",
    alpha: float = 0.05,
    n_resamples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Добавить adjusted p-value и reject-флаги всех методов к таблице элементарных тестов.

    Возвращает копию ``tests_df`` с колонками ``p_adj_<method>`` и ``reject_<method>``
    для каждого имени из :data:`METHOD_NAMES`.
    """
    out = tests_df.copy()
    p = out["p_value"].to_numpy(dtype=float)

    # p-value-only: Bonferroni / Holm — готовый statsmodels.multipletests.
    for name, sm_method in (("bonferroni", "bonferroni"), ("holm", "holm")):
        _, p_adj, _, _ = multipletests(p, alpha=alpha, method=sm_method)
        out[f"p_adj_{name}"] = p_adj

    # FDR-ориентир: Benjamini–Hochberg — готовый scipy.stats.false_discovery_control.
    out["p_adj_benjamini_hochberg"] = stats.false_discovery_control(p, method="bh")

    # Resampling-методы на общей перестановочной матрице t-статистик.
    t_obs, t_perm = _build_permutation_t_statistics(
        df, target_cols, treatment_col=treatment_col, n_resamples=n_resamples, random_state=random_state
    )
    # Выровнять порядок t_obs (как target_cols) с порядком строк tests_df.
    pos = {t: i for i, t in enumerate(target_cols)}
    idx = out["target"].map(pos).to_numpy()
    out["p_adj_westfall_young"] = westfall_young_maxT_adjusted_pvalues(t_obs, t_perm)[idx]
    out["p_adj_romano_wolf"] = romano_wolf_stepdown_adjusted_pvalues(t_obs, t_perm)[idx]

    for name in METHOD_NAMES:
        out[f"reject_{name}"] = out[f"p_adj_{name}"].to_numpy(dtype=float) <= alpha
    return out


def reject_columns() -> Dict[str, str]:
    """{имя метода → колонка reject_<method>} в каноническом порядке."""
    return {name: f"reject_{name}" for name in METHOD_NAMES}


def p_adj_columns() -> Dict[str, str]:
    """{имя метода → колонка p_adj_<method>} в каноническом порядке."""
    return {name: f"p_adj_{name}" for name in METHOD_NAMES}
