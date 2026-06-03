"""Адаптер к CUPAC из HypEx (reference/parity baseline для R&D-6).

Гоняет production-like CUPAC из HypEx через высокоуровневый
``ABTest(enable_cupac=True)`` на признаках класса **A** (pre-treatment лаги),
извлечённых из ``BenchmarkDataset``. Library-first: если ``hypex`` недоступен или
запуск не удался — возвращает статус ``unavailable``/``error`` (бенчмарк не падает;
HypEx при этом следует установить — см. план реализации).

Роли HypEx строятся из лаговых имён класса A по соглашению ``<parent>_lag<k>``:
лаги цели (parent ∈ {target_col, "y"}) → ``PreTargetRole``; лаги ковариат →
``FeatureRole`` с их parent; cofounders цели = parents ковариат.
"""

from __future__ import annotations

import re
from typing import Optional

from ..feature_safety.contracts import FeatureClass

DEFAULT_CUPAC_MODELS = ["linear", "ridge", "lasso"]
_LAG_RE = re.compile(r"^(?P<parent>.+)_lag(?P<lag>\d+)$")


def hypex_available() -> bool:
    try:
        import hypex  # noqa: F401

        return True
    except Exception:
        return False


def _parse_a_features(a_features, target_col: str):
    """Разбить класс-A признаки на лаги цели и лаги ковариат по именам."""
    target_lag_parents = {target_col, "y"}
    target_lags: list[tuple[str, int]] = []  # (col, lag)
    cov_lags: list[tuple[str, str, int]] = []  # (col, parent, lag)
    for col in a_features:
        m = _LAG_RE.match(col)
        if not m:
            continue
        parent, lag = m.group("parent"), int(m.group("lag"))
        if parent in target_lag_parents:
            target_lags.append((col, lag))
        else:
            cov_lags.append((col, parent, lag))
    return target_lags, cov_lags


def run_hypex_cupac(bds, cupac_models: Optional[list[str]] = None) -> dict:
    """Запустить HypEx CUPAC на класс-A признаках ``BenchmarkDataset``.

    Возвращает dict со статусом и (при ``ok``) полями ``ate, p_value,
    variance_reduction, best_model, feature_set_used``.
    """
    if not hypex_available():
        return {"status": "unavailable", "notes": "hypex not installed"}

    cupac_models = cupac_models or DEFAULT_CUPAC_MODELS
    target_col = bds.target_col
    a_features = bds.feature_registry.by_class(FeatureClass.A_PRE_TREATMENT)
    target_lags, cov_lags = _parse_a_features(a_features, target_col)

    if len(target_lags) < 2 or not cov_lags:
        return {
            "status": "error",
            "notes": "недостаточно лагов класса A для HypEx CUPAC (нужны лаги цели и ковариат)",
        }

    try:
        from hypex import ABTest
        from hypex.dataset import (
            Dataset,
            FeatureRole,
            InfoRole,
            PreTargetRole,
            TargetRole,
            TreatmentRole,
        )

        cofounder_parents = sorted({p for _, p, _ in cov_lags})
        roles: dict = {
            bds.treatment_col: TreatmentRole(),
            target_col: TargetRole(cofounders=cofounder_parents),
        }
        for col, lag in target_lags:
            roles[col] = PreTargetRole(parent=target_col, lag=lag)
        for col, parent, lag in cov_lags:
            roles[col] = FeatureRole(parent=parent, lag=lag)

        cols = [bds.treatment_col, target_col] + [c for c, _ in target_lags] + [
            c for c, _, _ in cov_lags
        ]
        data = Dataset(
            roles=roles, data=bds.data[cols].copy(), default_role=InfoRole()
        )
        result = ABTest(enable_cupac=True, cupac_models=cupac_models).execute(data)

        resume = result.resume.data
        adj = resume[resume["feature"] == f"{target_col}_cupac"]
        if adj.empty:
            return {"status": "error", "notes": "no <target>_cupac row in resume"}
        row = adj.iloc[0]
        vr = result.cupac.variance_reductions.data.iloc[0]
        feature_set = [c for c, _ in target_lags] + [c for c, _, _ in cov_lags]
        return {
            "status": "ok",
            "ate": float(row["difference"]),
            "p_value": float(row["TTest p-value"]),
            "variance_reduction": float(vr["variance_reduction_real"]),
            "best_model": str(vr["best_model"]),
            "feature_set_used": feature_set,
            "notes": f"hypex cupac_models={cupac_models}",
        }
    except Exception as e:  # noqa: BLE001 - library-first: не роняем бенчмарк
        return {"status": "error", "notes": f"{type(e).__name__}: {e}"}
