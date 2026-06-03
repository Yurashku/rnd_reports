"""Локальная R&D-реализация CUPAC (AutoCUPAC).

Портировано из репозитория VarWar (`autocupac.py`, класс ``CUPACTransformer``) без
изменения логики: KFold-перебор моделей, автовыбор по снижению дисперсии,
θ-резидуализация целевой метрики и отчёт по важностям признаков.

Отличия от оригинала:
- общая CUPED-математика вынесена в :mod:`rnd_reports.variance_reduction.cuped`
  и :mod:`rnd_reports.variance_reduction.metrics`;
- CatBoost — опциональная зависимость: если пакет не установлен, модель
  исключается из набора, а не роняет пайплайн.

См. docs/migration_from_varwar.md и docs/variance_reduction_methodology.md.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from .cuped import cuped_adjust
from .metrics import variance_reduction_pct

try:  # CatBoost — опциональная (тяжёлая) зависимость
    from catboost import CatBoostRegressor

    _HAS_CATBOOST = True
except ImportError:  # pragma: no cover - зависит от окружения
    CatBoostRegressor = None  # type: ignore[assignment]
    _HAS_CATBOOST = False


class CUPACTransformer:
    """Улучшенный CUPAC-трансформер с расширенной отчётностью."""

    def __init__(
        self,
        target_col: str,
        lag_suffix: str = "_lag",
        target_counterfactual_suffix: str = "0",
        models: Optional[Dict] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        self.target_col = target_col
        self.target_counterfactual_suffix = target_counterfactual_suffix
        self.lag_suffix = lag_suffix
        self.n_folds = n_folds
        self.random_state = random_state

        # Инициализация моделей (CatBoost — только если установлен)
        if models is not None:
            self.models = models
        else:
            self.models = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(alpha=0.5),
                "Lasso": Lasso(alpha=0.01, max_iter=10000),
            }
            if _HAS_CATBOOST:
                self.models["CatBoost"] = CatBoostRegressor(
                    iterations=100,
                    depth=4,
                    learning_rate=0.1,
                    silent=True,
                    random_state=random_state,
                    allow_writing_files=False,
                )

        # Состояние модели
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.variance_reduction = None
        self.lag_features = None
        self.current_features = None
        self.is_fitted = False
        self.model_results_: dict = {}
        self.feature_importances_ = None

    def _prepare_train_data(self, df: pd.DataFrame) -> tuple:
        """Подготовка данных для обучения."""
        target_counterfactual_name = (
            f"{self.target_col}{self.target_counterfactual_suffix}{self.lag_suffix}"
        )

        self.lag_features = [
            col
            for col in df.columns
            if col.endswith(self.lag_suffix)
            and col != f"{self.target_col}{self.lag_suffix}"
        ]

        if not self.lag_features:
            raise ValueError("Не найдены лаговые признаки для обучения")

        self.current_features = [
            col.replace(self.lag_suffix, "") for col in self.lag_features
        ]

        self.lag_features.append(f"{target_counterfactual_name}_2")
        self.current_features.append(f"{target_counterfactual_name}_1")

        return df[self.lag_features], df[f"{target_counterfactual_name}_1"]

    def _prepare_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка данных для применения."""
        if not self.current_features:
            raise RuntimeError("Сначала обучите модель (fit())")

        missing = [col for col in self.current_features if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют признаки: {missing}")

        return df[self.current_features].rename(
            columns=dict(zip(self.current_features, self.lag_features))
        )

    def fit(self, df: pd.DataFrame) -> "CUPACTransformer":
        """Обучение модели на исторических данных."""
        X, y = self._prepare_train_data(df)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        results = {}

        for name, model in self.models.items():
            fold_scores = []
            fold_var_reductions = []
            status = "success"

            try:
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    if name == "CatBoost":
                        m = CatBoostRegressor(**model.get_params())
                        m.fit(X_train, y_train, verbose=False)
                    else:
                        m = model.__class__(**model.get_params())
                        m.fit(X_train, y_train)

                    pred = m.predict(X_val)
                    fold_scores.append(r2_score(y_val, pred))
                    fold_var_reductions.append(variance_reduction_pct(y_val, pred))

                results[name] = {
                    "r2": np.nanmean(fold_scores),
                    "var_reduction": np.nanmean(fold_var_reductions),
                    "status": status,
                }

            except Exception as e:  # noqa: BLE001 - сохраняем поведение оригинала
                error_msg = f"{type(e).__name__}: {str(e)}"
                results[name] = {
                    "r2": None,
                    "var_reduction": None,
                    "status": f"failed: {error_msg}",
                }
                print(f"Ошибка в {name}: {error_msg}")

        self.model_results_ = results

        # Выбор лучшей модели из успешных
        successful_models = {
            k: v for k, v in results.items() if v["status"] == "success"
        }
        if not successful_models:
            raise RuntimeError("Все модели завершились с ошибкой")

        self.best_model_name = max(
            successful_models, key=lambda x: successful_models[x]["var_reduction"]
        )
        self.best_score = successful_models[self.best_model_name]["r2"]
        self.variance_reduction = successful_models[self.best_model_name][
            "var_reduction"
        ]

        # Финальное обучение и feature importance
        X, y = self._prepare_train_data(df)
        best_model_params = self.models[self.best_model_name].get_params()

        if self.best_model_name == "CatBoost":
            self.best_model = CatBoostRegressor(**best_model_params)
            self.best_model.fit(X, y, verbose=False)
            self.feature_importances_ = dict(
                zip(X.columns, self.best_model.get_feature_importance())
            )
        else:
            self.best_model = self.models[self.best_model_name].__class__(
                **best_model_params
            )
            self.best_model.fit(X, y)
            if hasattr(self.best_model, "coef_"):
                self.feature_importances_ = dict(zip(X.columns, self.best_model.coef_))
            else:
                self.feature_importances_ = None

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Применение модели к новым данным (добавляет колонку ``<target>_cupac``)."""
        if not self.is_fitted:
            raise RuntimeError("Сначала вызовите fit()")

        X = self._prepare_inference_data(df)
        y = df[self.target_col]
        pred = self.best_model.predict(X)

        y_adj = cuped_adjust(y, pred)

        if inplace:
            df[f"{self.target_col}_cupac"] = y_adj
            return df
        return df.assign(**{f"{self.target_col}_cupac": y_adj})

    def get_report(self) -> str:
        """Генерация расширенного текстового отчёта."""
        if not self.is_fitted:
            return "Модель не обучена. Сначала вызовите fit()."

        sorted_features = (
            sorted(
                self.feature_importances_.items(), key=lambda x: abs(x[1]), reverse=True
            )[:10]
            if self.feature_importances_
            else []
        )

        model_comparison = []
        for name, data in self.model_results_.items():
            if data["status"] != "success":
                line = f"{name}: {data['status']}"
            else:
                line = (
                    f"{name}: R²={data['r2']:.3f}, "
                    f"Var.Red.={data['var_reduction']:.1f}%"
                )
            model_comparison.append(line)

        feature_analysis = []
        if sorted_features:
            max_coef = max(abs(v) for _, v in sorted_features)
            for feat, coef in sorted_features:
                rel_impact = abs(coef) / max_coef if max_coef != 0 else 0
                feature_analysis.append(
                    f"- {feat:<25} {coef:>7.3f} {'▇'*int(10*rel_impact)}"
                )

        report = [
            "Расширенный CUPAC Report",
            "=" * 40,
            "Сравнение моделей:",
            *model_comparison,
            "",
            f"Лучшая модель: {self.best_model_name}",
            f"Снижение дисперсии: {self.variance_reduction:.1f}%",
            f"Качество предсказания (R²): {self.best_score:.3f}",
            "",
            "Топ-10 значимых признаков:",
            *(
                feature_analysis
                if feature_analysis
                else ["Нет данных о важности признаков"]
            ),
            "",
            "Интерпретация:",
            "▇▇▇▇▇▇▇▇▇▇ - максимальное влияние",
            "Коэффициенты > 0: положительная связь с целевой переменной",
            "Коэффициенты < 0: отрицательная связь",
        ]
        return "\n".join(report)

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        df_apply: Optional[pd.DataFrame] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        self.fit(df_train)
        df_apply = df_train if df_apply is None else df_apply
        return self.transform(df_apply, inplace=inplace)

    def get_feature_mapping(self) -> Dict[str, str]:
        return dict(zip(self.lag_features, self.current_features))


def _default_sklearn_models() -> Dict[str, object]:
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=0.5),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
    }


def local_cupac_adjust(
    df: pd.DataFrame,
    target_col: str,
    features: list[str],
    models: Optional[Dict[str, object]] = None,
    n_folds: int = 5,
    random_state: Optional[int] = None,
):
    """Local CUPAC по явному списку признаков (feature-list API для R&D-6).

    Кросс-фитом предсказывает ``target_col`` по ``features`` (out-of-fold), выбирает
    лучшую модель по снижению дисперсии и делает CUPED θ-residualize на OOF-прогнозе
    (чтобы не переобучаться при подгонке). Не использует колонку treatment.

    Возвращает ``(adjusted: pd.Series, info: dict)`` с ``best_model``,
    ``variance_reduction`` (%) и ``per_model`` (var.red. по моделям).
    """
    from sklearn.base import clone

    y = df[target_col].reset_index(drop=True)
    if not features:
        return y.copy(), {"best_model": None, "variance_reduction": 0.0, "per_model": {}}

    X = df[features].reset_index(drop=True)
    models = models or _default_sklearn_models()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    per_model: dict[str, float] = {}
    oof_by_model: dict[str, np.ndarray] = {}

    for name, proto in models.items():
        oof = np.full(len(y), np.nan)
        for train_idx, val_idx in kf.split(X):
            m = clone(proto)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[val_idx] = m.predict(X.iloc[val_idx])
        oof_by_model[name] = oof
        per_model[name] = variance_reduction_pct(y, oof)

    best_model = max(per_model, key=per_model.get)
    adjusted = cuped_adjust(y, oof_by_model[best_model])
    info = {
        "best_model": best_model,
        "variance_reduction": per_model[best_model],
        "per_model": per_model,
    }
    return adjusted, info
