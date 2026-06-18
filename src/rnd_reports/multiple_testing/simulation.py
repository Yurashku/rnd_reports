"""Monte-Carlo operating characteristics — ядро честного сравнения методов.

Одна реализация таблицы не позволяет сравнивать методы: нужно усреднить их поведение по
многим симуляциям с известной правдой. Здесь для сетки сценариев (варьируем корреляцию
``rho``) считаем по каждому методу:

- **эмпирический FWER** — доля симуляций, где есть хотя бы одно ложное отвержение
  (ложный «успех» OR-claim);
- **average power** — средняя доля найденных истинно-ненулевых метрик.

Так видно главное различие: все FWER-методы держат FWER ≤ α, но Westfall–Young /
Romano–Wolf наращивают power с ростом корреляции, тогда как Bonferroni/Holm — нет.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .elementary import compute_elementary_tests, infer_target_columns
from .methods import METHOD_ERROR, METHOD_NAMES, add_corrections, reject_columns
from .synthetic import make_ab_table


@dataclass
class Scenario:
    """Один сценарий синтетики для симуляции."""

    rho: float
    n: int = 4_000
    n_targets: int = 15
    n_true: int = 3
    effect: float = 0.12
    treatment_share: float = 0.5
    extra: Dict[str, float] = field(default_factory=dict)

    def label(self) -> str:
        return f"rho={self.rho:g}"


def _single_run_rejections(
    df: pd.DataFrame,
    true_effects: Dict[str, float],
    alpha: float,
    n_resamples: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Один прогон: по каждому методу вернуть fwer_event и долю найденных истинных."""
    targets = infer_target_columns(df)
    tests = compute_elementary_tests(df, targets, alpha=alpha)
    corrected = add_corrections(tests, df, targets, alpha=alpha, n_resamples=n_resamples, random_state=seed)

    truth = corrected["target"].map(lambda t: abs(float(true_effects.get(t, 0.0))) > 1e-12).to_numpy()
    n_non_null = int(truth.sum())

    out: Dict[str, Dict[str, float]] = {}
    for name, col in reject_columns().items():
        reject = corrected[col].to_numpy(dtype=bool)
        fp = int(np.sum(reject & ~truth))
        tp = int(np.sum(reject & truth))
        out[name] = {
            "fwer_event": float(fp >= 1),
            "power": tp / n_non_null if n_non_null > 0 else np.nan,
        }
    return out


def operating_characteristics(
    scenarios: Sequence[Scenario],
    n_sims: int = 200,
    alpha: float = 0.05,
    n_resamples: int = 500,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Эмпирические FWER и power по методам для каждого сценария (усреднение по симуляциям).

    Возвращает tidy-таблицу со строками ``(scenario, rho, method, error_control, fwer, power,
    n_sims)``. Перестановки внутри каждого метода используют ``n_resamples`` (умеренно — это
    самый дорогой шаг), число симуляций ``n_sims`` задаёт точность оценок FWER/power.
    """
    rows = []
    for s_idx, scenario in enumerate(scenarios):
        agg = {name: {"fwer": [], "power": []} for name in METHOD_NAMES}
        for sim in range(n_sims):
            seed = base_seed + 1_000 * s_idx + sim
            df, true_effects = make_ab_table(
                n=scenario.n,
                n_targets=scenario.n_targets,
                rho=scenario.rho,
                n_true=scenario.n_true,
                effect=scenario.effect,
                treatment_share=scenario.treatment_share,
                seed=seed,
            )
            per_method = _single_run_rejections(df, true_effects, alpha, n_resamples, seed)
            for name in METHOD_NAMES:
                agg[name]["fwer"].append(per_method[name]["fwer_event"])
                agg[name]["power"].append(per_method[name]["power"])

        for name in METHOD_NAMES:
            powers = np.asarray(agg[name]["power"], dtype=float)
            # power неопределён, если в сценарии нет истинных эффектов (n_true=0).
            mean_power = float(np.nanmean(powers)) if np.any(np.isfinite(powers)) else float("nan")
            rows.append(
                {
                    "scenario": scenario.label(),
                    "rho": scenario.rho,
                    "method": name,
                    "error_control": METHOD_ERROR[name],
                    "fwer": float(np.mean(agg[name]["fwer"])),
                    "power": mean_power,
                    "n_sims": n_sims,
                }
            )
    return pd.DataFrame(rows)


def rho_grid(rhos: Sequence[float], **scenario_kwargs) -> List[Scenario]:
    """Удобный конструктор сетки сценариев, варьирующих только корреляцию ``rho``."""
    return [Scenario(rho=float(r), **scenario_kwargs) for r in rhos]
