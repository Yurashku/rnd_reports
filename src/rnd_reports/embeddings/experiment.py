"""R&D-7: эмбеддинги как adjustment set в наблюдательном (нерандомизированном) испытании.

Реализация causal-слоя поверх сокращённых эмбеддингов. Всё на ``numpy``/``scikit-learn``
(без pyspark) — запускается в base-окружении и переиспользуется ноутбуком/тестами.

Идея: в наблюдательных данных назначение трита коррелирует с признаками (селекция), поэтому
наивная разность средних смещена. Если сокращённые эмбеддинги покрывают конфаундеры, то
поправка на них (propensity-взвешивание / doubly-robust) убирает смещение. Здесь — оценка ATE
с поправкой, диагностика баланса и overlap, и сравнение с эталонным эффектом.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

_EPS = 1e-6


def _as_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def fit_propensity(reduced_embeddings: Any, treatment: Any, *,
                   max_iter: int = 1000, C: float = 1.0) -> np.ndarray:
    """Propensity ``P(T=1 | reduced_embeddings)`` через ``LogisticRegression``.

    Возвращает вектор вероятностей, клипнутый в ``[_EPS, 1-_EPS]`` (для устойчивости IPW).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = _as_2d(reduced_embeddings)
    t = _as_1d(treatment).astype(int)
    xs = StandardScaler().fit_transform(x)
    model = LogisticRegression(max_iter=max_iter, C=C)
    model.fit(xs, t)
    p = model.predict_proba(xs)[:, 1]
    return np.clip(p, _EPS, 1.0 - _EPS)


def _ipw_ate(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Стабилизированный IPW-ATE (Hajek): нормировка весов внутри каждой группы."""
    w1 = t / p
    w0 = (1.0 - t) / (1.0 - p)
    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)
    return float(mu1 - mu0)


def _dr_ate(x: np.ndarray, t: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Doubly-robust (AIPW): per-arm линейная регрессия исхода + IPW-коррекция остатков."""
    from sklearn.linear_model import LinearRegression

    mu1_model = LinearRegression().fit(x[t == 1], y[t == 1])
    mu0_model = LinearRegression().fit(x[t == 0], y[t == 0])
    mu1 = mu1_model.predict(x)
    mu0 = mu0_model.predict(x)
    psi = (mu1 - mu0
           + t * (y - mu1) / p
           - (1.0 - t) * (y - mu0) / (1.0 - p))
    return float(np.mean(psi))


def estimate_ate_with_adjustment(
    reduced_embeddings: Any,
    treatment: Any,
    outcome: Any,
    *,
    method: str = "propensity_weighting",
) -> float:
    """Оценить ATE в наблюдательных данных с поправкой на эмбеддинги.

    ``method``:
    - ``"naive"`` — разность средних без поправки (смещена при селекции);
    - ``"propensity_weighting"`` — стабилизированный IPW по propensity на эмбеддингах;
    - ``"doubly_robust"`` — AIPW (per-arm регрессия исхода + IPW-коррекция).
    """
    t = _as_1d(treatment)
    y = _as_1d(outcome)
    if method == "naive":
        return float(y[t == 1].mean() - y[t == 0].mean())
    x = _as_2d(reduced_embeddings)
    p = fit_propensity(x, t)
    if method == "propensity_weighting":
        return _ipw_ate(t, y, p)
    if method == "doubly_robust":
        return _dr_ate(x, t, y, p)
    raise ValueError(
        f"Неизвестный method={method!r}; ожидается naive/propensity_weighting/doubly_robust"
    )


def _smd(values: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Standardized mean difference одной ковариаты (опц. с весами)."""
    if w is None:
        m1, m0 = values[t == 1].mean(), values[t == 0].mean()
        v1, v0 = values[t == 1].var(), values[t == 0].var()
    else:
        def wm(mask):
            ww = w[mask]
            return np.sum(ww * values[mask]) / np.sum(ww)

        def wv(mask, mu):
            ww = w[mask]
            return np.sum(ww * (values[mask] - mu) ** 2) / np.sum(ww)

        m1, m0 = wm(t == 1), wm(t == 0)
        v1, v0 = wv(t == 1, m1), wv(t == 0, m0)
    pooled = np.sqrt((v1 + v0) / 2.0)
    return float(abs(m1 - m0) / pooled) if pooled > _EPS else 0.0


def covariate_balance_after_adjustment(
    reduced_embeddings: Any,
    treatment: Any,
    *,
    propensity_score: Any | None = None,
) -> Mapping[str, float]:
    """Баланс ковариат (|SMD|) по эмбеддингам до и после IPW-взвешивания.

    Возвращает сводку: max/mean ``|SMD|`` до и после. Хороший adjustment set уменьшает
    дисбаланс после взвешивания.
    """
    x = _as_2d(reduced_embeddings)
    t = _as_1d(treatment)
    p = (_as_1d(propensity_score) if propensity_score is not None
         else fit_propensity(x, t))
    w = t / p + (1.0 - t) / (1.0 - p)
    before = [_smd(x[:, j], t) for j in range(x.shape[1])]
    after = [_smd(x[:, j], t, w) for j in range(x.shape[1])]
    return {
        "n_covariates": float(x.shape[1]),
        "max_abs_smd_before": float(np.max(before)),
        "mean_abs_smd_before": float(np.mean(before)),
        "max_abs_smd_after": float(np.max(after)),
        "mean_abs_smd_after": float(np.mean(after)),
    }


def overlap_diagnostics(propensity_score: Any) -> Mapping[str, float]:
    """Overlap/positivity по распределению propensity score.

    Хороший overlap: масса в середине [0.1, 0.9], мало экстремальных значений.
    """
    p = _as_1d(propensity_score)
    return {
        "min": float(p.min()),
        "max": float(p.max()),
        "p01": float(np.percentile(p, 1)),
        "p99": float(np.percentile(p, 99)),
        "frac_in_0.1_0.9": float(np.mean((p >= 0.1) & (p <= 0.9))),
        "frac_below_0.05": float(np.mean(p < 0.05)),
        "frac_above_0.95": float(np.mean(p > 0.95)),
    }


def evaluate_adjustment_set_quality(
    reduced_embeddings: Any,
    treatment: Any,
    outcome: Any,
    *,
    ground_truth_effect: float | None = None,
) -> Mapping[str, float]:
    """Качество adjustment set из сокращённых эмбеддингов.

    Считает ATE тремя способами (naive / IPW / DR), баланс и overlap. При известном
    ``ground_truth_effect`` добавляет смещения и относительное снижение смещения IPW vs naive.
    """
    x = _as_2d(reduced_embeddings)
    t = _as_1d(treatment)
    y = _as_1d(outcome)
    p = fit_propensity(x, t)

    ate_naive = estimate_ate_with_adjustment(x, t, y, method="naive")
    ate_ipw = _ipw_ate(t, y, p)
    ate_dr = _dr_ate(x, t, y, p)

    out: dict[str, float] = {
        "ate_naive": ate_naive,
        "ate_ipw": ate_ipw,
        "ate_dr": ate_dr,
    }
    bal = covariate_balance_after_adjustment(x, t, propensity_score=p)
    out["max_abs_smd_before"] = bal["max_abs_smd_before"]
    out["max_abs_smd_after"] = bal["max_abs_smd_after"]
    ov = overlap_diagnostics(p)
    out["overlap_frac_in_0.1_0.9"] = ov["frac_in_0.1_0.9"]

    if ground_truth_effect is not None:
        g = float(ground_truth_effect)
        out["ground_truth"] = g
        out["bias_naive"] = ate_naive - g
        out["bias_ipw"] = ate_ipw - g
        out["bias_dr"] = ate_dr - g
        denom = abs(ate_naive - g)
        if denom > _EPS:
            out["abs_bias_reduction_ipw_pct"] = float(
                (1.0 - abs(ate_ipw - g) / denom) * 100.0
            )
            out["abs_bias_reduction_dr_pct"] = float(
                (1.0 - abs(ate_dr - g) / denom) * 100.0
            )
    return out
