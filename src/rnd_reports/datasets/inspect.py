"""Лёгкая схема-инспекция «сырых» DataFrame для аудита датасетов R&D-6.

Цель — быстрый честный осмотр любого реального датасета перед разметкой A–F:
типы, доля пропусков, кардинальность, примеры значений и эвристические флаги
(datetime/id/binary/treatment-target). Используется в ``tools/audit_datasets.py``
и в отчёте ``docs/06_real_dataset_raw_audit.md``.

Контракт ``schema_summary``:

- чистая функция: ничего не качает, ничего не пишет на диск;
- не зависит от scikit-uplift;
- один ряд на колонку входного DataFrame.
"""

from __future__ import annotations

import re

import pandas as pd

# Эвристики по имени колонки (регистронезависимо).
_DATETIME_NAME = re.compile(r"(date|datetime|timestamp|_ts$|^ts$|_time$|^time$)", re.I)
_ID_NAME = re.compile(r"(^id$|_id$|^id_|client|customer|user|account)", re.I)
_TARGET_TREATMENT_NAME = re.compile(
    r"(treatment|treat|target|response|conversion|convert|visit|group|exposure|label|outcome)",
    re.I,
)

# Порог доли успешно распарсенных дат для object-колонок и доли уникальных для id.
_DATETIME_PARSE_THRESHOLD = 0.9
_ID_UNIQUE_RATIO = 0.9
_EXAMPLE_VALUES_N = 5


def _looks_like_datetime(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if _DATETIME_NAME.search(str(name)):
        return True
    # Пытаемся распарсить только object/строковые колонки, чтобы не трактовать
    # обычные числа (например, recency в днях) как даты.
    if series.dtype == object:
        non_null = series.dropna()
        if non_null.empty:
            return False
        sample = non_null.head(1000)
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= _DATETIME_PARSE_THRESHOLD
    return False


def _looks_like_id(series: pd.Series, name: str, n_rows: int) -> bool:
    if not _ID_NAME.search(str(name)):
        return False
    if n_rows == 0:
        return False
    return series.nunique(dropna=True) / n_rows >= _ID_UNIQUE_RATIO


def _looks_like_binary(n_unique: int) -> bool:
    return n_unique <= 2


def _example_values(series: pd.Series) -> list:
    return list(series.dropna().unique()[:_EXAMPLE_VALUES_N])


def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Вернуть таблицу-сводку схемы: один ряд на колонку ``df``.

    Колонки результата:
        column, dtype, missing_pct, n_unique, example_values,
        looks_like_datetime, looks_like_id, looks_like_binary,
        looks_like_target_or_treatment.

    Чистая функция: без скачиваний, записи на диск и зависимости от sklift.
    """
    n_rows = len(df)
    rows: list[dict] = []
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        missing_pct = float(s.isna().mean() * 100.0) if n_rows else 0.0
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_pct": round(missing_pct, 4),
                "n_unique": n_unique,
                "example_values": _example_values(s),
                "looks_like_datetime": _looks_like_datetime(s, col),
                "looks_like_id": _looks_like_id(s, col, n_rows),
                "looks_like_binary": _looks_like_binary(n_unique),
                "looks_like_target_or_treatment": bool(
                    _TARGET_TREATMENT_NAME.search(str(col))
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "dtype",
            "missing_pct",
            "n_unique",
            "example_values",
            "looks_like_datetime",
            "looks_like_id",
            "looks_like_binary",
            "looks_like_target_or_treatment",
        ],
    )
