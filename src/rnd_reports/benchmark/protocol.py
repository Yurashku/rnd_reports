"""Протокол оценки и метрик бенчмарка R&D-6 (Step 3).

Реализует:
- ``estimate_ate`` — оценку ATE рандомизированного A/B (разность средних + Welch);
- A/B-методы ``raw`` и ``ab_hypex`` (последний дополнительно сверяется с HypEx-A/B);
- ``finalize_metrics`` — относительные и инкрементальные метрики (§6–7 контекста):
  снижение дисперсии и требуемой выборки vs A/B, vs sklearn CUPAC и vs предшественник.

На этом шаге реализованы только baseline-методы (``raw``, ``ab_hypex``); CUPAC и
safe-intime коррекции добавляются в последующих шагах. Алгоритмы коррекции тут нет.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from ..datasets.contracts import BenchmarkDataset
from . import method_registry as mr
from .contracts import (
    RESULT_COLUMNS,
    SAFETY_OK,
    MethodResult,
)

# Методы, реализованные на текущем шаге (Step 3).
IMPLEMENTED_METHODS = ("raw", "ab_hypex")


def estimate_ate(df, outcome_col: str, treatment_col: str) -> dict:
    """Оценка ATE рандомизированного A/B: разность средних + Welch t-test.

    Возвращает ``n, ate, se, p_value, ci_low, ci_high, variance`` (дисперсия исхода —
    база для расчёта снижения дисперсии/выборки).
    """
    y = df[outcome_col].to_numpy(dtype=float)
    t = df[treatment_col].to_numpy()
    yt, yc = y[t == 1], y[t == 0]
    if len(yt) < 2 or len(yc) < 2:
        raise ValueError("Недостаточно наблюдений в группах treatment/control")

    ate = float(yt.mean() - yc.mean())
    vt, vc = yt.var(ddof=1), yc.var(ddof=1)
    se = float(np.sqrt(vt / len(yt) + vc / len(yc)))
    _, p_value = stats.ttest_ind(yt, yc, equal_var=False, nan_policy="omit")
    dof = (vt / len(yt) + vc / len(yc)) ** 2 / (
        (vt / len(yt)) ** 2 / (len(yt) - 1) + (vc / len(yc)) ** 2 / (len(yc) - 1)
    )
    tcrit = float(stats.t.ppf(0.975, dof))
    return {
        "n": int(len(y)),
        "ate": ate,
        "se": se,
        "p_value": float(p_value),
        "ci_low": ate - tcrit * se,
        "ci_high": ate + tcrit * se,
        "variance": float(np.var(y, ddof=1)),
    }


def reduction_pct(variance_method: Optional[float], variance_baseline: Optional[float]):
    """Снижение дисперсии/требуемой выборки vs baseline, %, по формуле §7.

    ``100 * (1 - variance_method / variance_baseline)``. ``None`` при отсутствии базы.
    """
    if (
        variance_method is None
        or variance_baseline is None
        or not np.isfinite(variance_method)
        or not np.isfinite(variance_baseline)
        or variance_baseline == 0
    ):
        return None
    return 100.0 * (1.0 - variance_method / variance_baseline)


def _hypex_ab_parity(bds: BenchmarkDataset) -> Optional[dict]:
    """A/B через HypEx (parity-сверка). ``None``, если HypEx недоступен/ошибка.

    Library-first: используется как дополнительная проверка сопоставимости; основные
    числа таблицы считает внутренний ``estimate_ate`` (внутренняя согласованность).
    """
    try:
        from hypex import ABTest
        from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole

        roles = {
            bds.treatment_col: TreatmentRole(),
            bds.target_col: TargetRole(),
        }
        data = Dataset(
            roles=roles,
            data=bds.data[[bds.treatment_col, bds.target_col]].copy(),
            default_role=InfoRole(),
        )
        res = ABTest().execute(data)
        row = res.resume.data
        r = row[row["feature"] == bds.target_col].iloc[0]
        return {"ate": float(r["difference"]), "p_value": float(r["TTest p-value"])}
    except Exception:
        return None


def _baseline_result(
    bds: BenchmarkDataset,
    method: str,
    dataset_name: str,
    dataset_type: str,
    hypothesis_name: str,
) -> MethodResult:
    """MethodResult для A/B-метода (``raw`` или ``ab_hypex``) — без коррекции исхода.

    ``raw`` — внутренний sanity-baseline (нет в реестре методов), ``ab_hypex`` —
    официальный A/B baseline из контекста.
    """
    spec = next((m for m in mr.METHODS if m.name == method), None)
    predecessor = spec.predecessor if spec is not None else None
    est = estimate_ate(bds.data, bds.target_col, bds.treatment_col)

    notes = ""
    if method == "ab_hypex":
        parity = _hypex_ab_parity(bds)
        if parity is not None:
            notes = (
                f"hypex A/B parity: ate={parity['ate']:.4f}, p={parity['p_value']:.3g}"
            )
        else:
            notes = "hypex недоступен — A/B посчитан внутренним оценщиком"

    return MethodResult(
        method=method,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        target=bds.target_col,
        hypothesis_name=hypothesis_name,
        predecessor_method=predecessor,
        n=est["n"],
        ate=est["ate"],
        se=est["se"],
        p_value=est["p_value"],
        ci_low=est["ci_low"],
        ci_high=est["ci_high"],
        adjusted_target_variance=est["variance"],
        feature_groups_used=[],
        n_features_used=0,
        safety_status=SAFETY_OK,
        diagnostic_notes=notes,
    )


def run_method(
    bds: BenchmarkDataset,
    method: str,
    dataset_name: str = "synthetic",
    dataset_type: str = "synthetic",
    hypothesis_name: str = "",
) -> MethodResult:
    """Посчитать один метод. На Step 3 поддержаны только ``raw`` и ``ab_hypex``."""
    if method not in IMPLEMENTED_METHODS:
        raise NotImplementedError(
            f"Метод '{method}' ещё не реализован (Step 3 — только {IMPLEMENTED_METHODS}). "
            "См. docs/06_safe_intime_cupac_implementation_plan.md."
        )
    return _baseline_result(bds, method, dataset_name, dataset_type, hypothesis_name)


def run_benchmark(
    bds: BenchmarkDataset,
    methods: Optional[list[str]] = None,
    dataset_name: str = "synthetic",
    dataset_type: str = "synthetic",
    hypothesis_name: str = "",
) -> list[MethodResult]:
    """Прогнать набор методов и заполнить относительные/инкрементальные метрики."""
    methods = methods or list(IMPLEMENTED_METHODS)
    results = [
        run_method(bds, m, dataset_name, dataset_type, hypothesis_name) for m in methods
    ]
    finalize_metrics(results)
    return results


def finalize_metrics(results: list[MethodResult]) -> list[MethodResult]:
    """Заполнить колонки снижения дисперсии/выборки (vs A/B, vs sklearn CUPAC, vs предшественник).

    По §6–7: для дисперсии и требуемой выборки величина в первом приближении совпадает.
    """
    by_name = {r.method: r for r in results}

    def var_of(name: Optional[str]) -> Optional[float]:
        r = by_name.get(name) if name else None
        return r.adjusted_target_variance if r else None

    # База A/B — метод-AB_BASELINE (ab_hypex), при отсутствии — 'raw'.
    ab_method = next(
        (m.name for m in mr.METHODS if m.kind is mr.MethodKind.AB_BASELINE), None
    )
    ab_var = var_of(ab_method) if ab_method in by_name else var_of("raw")
    cupac_var = var_of("sklearn_cupac_A")

    for r in results:
        v = r.adjusted_target_variance
        vr_ab = reduction_pct(v, ab_var)
        r.variance_reduction_vs_ab_pct = vr_ab
        r.sample_size_reduction_vs_ab_pct = vr_ab

        vr_cupac = reduction_pct(v, cupac_var)
        r.variance_reduction_vs_sklearn_cupac_pct = vr_cupac
        r.sample_size_reduction_vs_sklearn_cupac_pct = vr_cupac

        pred_var = var_of(r.predecessor_method)
        incr = reduction_pct(v, pred_var)
        r.incremental_variance_reduction_vs_predecessor_pct = incr
        r.incremental_sample_size_reduction_vs_predecessor_pct = incr

    return results


__all__ = [
    "RESULT_COLUMNS",
    "estimate_ate",
    "reduction_pct",
    "run_method",
    "run_benchmark",
    "finalize_metrics",
    "IMPLEMENTED_METHODS",
]
