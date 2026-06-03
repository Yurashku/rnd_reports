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
from ..feature_safety.contracts import FeatureClass
from ..variance_reduction.hypex_cupac_adapter import run_hypex_cupac
from ..variance_reduction.local_cupac import local_cupac_adjust
from . import method_registry as mr
from .contracts import (
    RESULT_COLUMNS,
    SAFETY_OK,
    SAFETY_REFERENCE,
    SAFETY_UNAVAILABLE,
    MethodResult,
)

# Методы, реализованные на текущем шаге (Step 4).
IMPLEMENTED_METHODS = ("raw", "ab_hypex", "sklearn_cupac_A", "hypex_cupac")


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


def _ate_from_ate_p(ate: float, p_value: float) -> dict:
    """Реконструкция SE/CI из ATE и p-value (для метрик, пришедших из HypEx)."""
    p = min(max(p_value, 1e-300), 1.0)
    z = float(stats.norm.isf(p / 2.0))
    se = abs(ate) / z if z > 0 else float("nan")
    return {"se": se, "ci_low": ate - 1.96 * se, "ci_high": ate + 1.96 * se}


def _sklearn_cupac_result(
    bds, dataset_name, dataset_type, hypothesis_name, random_state
) -> MethodResult:
    """sklearn_cupac_A: основной CUPAC baseline по признакам класса A."""
    spec = mr.by_name("sklearn_cupac_A")
    a_features = bds.feature_registry.by_class(FeatureClass.A_PRE_TREATMENT)
    adjusted, info = local_cupac_adjust(
        bds.data, bds.target_col, a_features, random_state=random_state
    )
    tmp = bds.data.assign(_adj=adjusted.to_numpy())
    est = estimate_ate(tmp, "_adj", bds.treatment_col)
    return MethodResult(
        method="sklearn_cupac_A",
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        target=bds.target_col,
        hypothesis_name=hypothesis_name,
        predecessor_method=spec.predecessor,
        n=est["n"],
        ate=est["ate"],
        se=est["se"],
        p_value=est["p_value"],
        ci_low=est["ci_low"],
        ci_high=est["ci_high"],
        adjusted_target_variance=est["variance"],
        feature_groups_used=["A"],
        n_features_used=len(a_features),
        safety_status=SAFETY_OK,
        diagnostic_notes=f"cupac best_model={info['best_model']}",
    )


def _hypex_cupac_result(
    bds, dataset_name, dataset_type, hypothesis_name, cupac_models
) -> MethodResult:
    """hypex_cupac: reference/parity CUPAC из HypEx (вне predecessor-chain)."""
    import numpy as _np

    hres = run_hypex_cupac(bds, cupac_models=cupac_models)
    base = MethodResult(
        method="hypex_cupac",
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        target=bds.target_col,
        hypothesis_name=hypothesis_name,
        predecessor_method=None,  # reference_only
        feature_groups_used=["A"],
        safety_status=SAFETY_REFERENCE,
        diagnostic_notes=hres.get("notes", ""),
    )
    if hres.get("status") != "ok":
        base.safety_status = (
            SAFETY_UNAVAILABLE if hres.get("status") == "unavailable" else SAFETY_REFERENCE
        )
        base.diagnostic_notes = f"hypex_cupac {hres.get('status')}: {hres.get('notes','')}"
        return base

    raw_var = float(_np.var(bds.data[bds.target_col].to_numpy(dtype=float), ddof=1))
    vr = hres["variance_reduction"]
    base.n = int(bds.n)
    base.ate = hres["ate"]
    base.p_value = hres["p_value"]
    base.adjusted_target_variance = raw_var * (1 - vr / 100.0)
    base.n_features_used = len(hres["feature_set_used"])
    base.diagnostic_notes = f"{hres['notes']}; best_model={hres['best_model']}"
    se_ci = _ate_from_ate_p(hres["ate"], hres["p_value"])
    base.se = se_ci["se"]
    base.ci_low = se_ci["ci_low"]
    base.ci_high = se_ci["ci_high"]
    return base


def run_method(
    bds: BenchmarkDataset,
    method: str,
    dataset_name: str = "synthetic",
    dataset_type: str = "synthetic",
    hypothesis_name: str = "",
    random_state: Optional[int] = None,
    cupac_models: Optional[list[str]] = None,
) -> MethodResult:
    """Посчитать один метод. На Step 4 поддержаны raw, ab_hypex, sklearn_cupac_A, hypex_cupac."""
    if method in ("raw", "ab_hypex"):
        return _baseline_result(bds, method, dataset_name, dataset_type, hypothesis_name)
    if method == "sklearn_cupac_A":
        return _sklearn_cupac_result(
            bds, dataset_name, dataset_type, hypothesis_name, random_state
        )
    if method == "hypex_cupac":
        return _hypex_cupac_result(
            bds, dataset_name, dataset_type, hypothesis_name, cupac_models
        )
    raise NotImplementedError(
        f"Метод '{method}' ещё не реализован (Step 4 — {IMPLEMENTED_METHODS}). "
        "См. docs/06_safe_intime_cupac_implementation_plan.md."
    )


def run_benchmark(
    bds: BenchmarkDataset,
    methods: Optional[list[str]] = None,
    dataset_name: str = "synthetic",
    dataset_type: str = "synthetic",
    hypothesis_name: str = "",
    random_state: Optional[int] = None,
    cupac_models: Optional[list[str]] = None,
) -> list[MethodResult]:
    """Прогнать набор методов и заполнить относительные/инкрементальные метрики."""
    methods = methods or list(IMPLEMENTED_METHODS)
    results = [
        run_method(
            bds,
            m,
            dataset_name,
            dataset_type,
            hypothesis_name,
            random_state=random_state,
            cupac_models=cupac_models,
        )
        for m in methods
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
