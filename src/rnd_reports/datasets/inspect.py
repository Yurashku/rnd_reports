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
# Анонимизированные имена-признаки: f0/f12, X_1, PC1, FACTOR3, V17, feature_5, col_2.
_ANON_NAME = re.compile(
    r"^(f\d+|x_?\d+|pc\d+|factor\d+|v\d+|feature_?\d+|col_?\d+|var_?\d+)$", re.I
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


def _feature_family(series: pd.Series, name: str, *, is_dt: bool, is_id: bool) -> str:
    """Грубая «семья» колонки для аудита: id / datetime / anonymized / named.

    Помогает быстро отличить интерпретируемые именованные признаки (потенциально
    A/B/C при наличии timing-evidence) от анонимных снимков (``f*``/``PC*``/``X_*``),
    которые НЕ могут быть размечены как safe B/C.
    """
    if is_id:
        return "id"
    if is_dt:
        return "datetime"
    if _ANON_NAME.search(str(name)):
        return "anonymized"
    if _TARGET_TREATMENT_NAME.search(str(name)):
        return "target_or_treatment"
    return "named"


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Сгруппировать колонки по ролям-эвристикам (без скачиваний и записи).

    Возвращает словарь со списками имён:
    ``id`` · ``datetime`` · ``target_or_treatment`` · ``anonymized`` · ``named``.
    Переиспользуется в ``tools/audit_datasets.py`` и при разметке адаптеров —
    это лишь подсказка, итоговая разметка A–F всегда ручная (см. контекст §3).
    """
    n_rows = len(df)
    out: dict[str, list[str]] = {
        "id": [],
        "datetime": [],
        "target_or_treatment": [],
        "anonymized": [],
        "named": [],
    }
    for col in df.columns:
        s = df[col]
        is_dt = _looks_like_datetime(s, col)
        is_id = _looks_like_id(s, col, n_rows)
        fam = _feature_family(s, col, is_dt=is_dt, is_id=is_id)
        out[fam].append(col)
    return out


def schema_summary(
    df: pd.DataFrame,
    *,
    include_feature_family: bool = False,
    include_notes: bool = False,
) -> pd.DataFrame:
    """Вернуть таблицу-сводку схемы: один ряд на колонку ``df``.

    Базовые колонки результата (всегда):
        column, dtype, missing_pct, n_unique, example_values,
        looks_like_datetime, looks_like_id, looks_like_binary,
        looks_like_target_or_treatment.

    Опционально (обратная совместимость — по умолчанию выключено):
        - ``include_feature_family=True`` → колонка ``feature_family``
          (id / datetime / target_or_treatment / anonymized / named);
        - ``include_notes=True`` → колонка ``notes`` (короткая эвристическая пометка,
          напр. «anonymized snapshot — не размечать как safe B/C»).

    Чистая функция: без скачиваний, записи на диск и зависимости от sklift.
    """
    n_rows = len(df)
    rows: list[dict] = []
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        missing_pct = float(s.isna().mean() * 100.0) if n_rows else 0.0
        is_dt = _looks_like_datetime(s, col)
        is_id = _looks_like_id(s, col, n_rows)
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_pct": round(missing_pct, 4),
            "n_unique": n_unique,
            "example_values": _example_values(s),
            "looks_like_datetime": is_dt,
            "looks_like_id": is_id,
            "looks_like_binary": _looks_like_binary(n_unique),
            "looks_like_target_or_treatment": bool(
                _TARGET_TREATMENT_NAME.search(str(col))
            ),
        }
        if include_feature_family or include_notes:
            fam = _feature_family(s, col, is_dt=is_dt, is_id=is_id)
            if include_feature_family:
                row["feature_family"] = fam
            if include_notes:
                row["notes"] = _column_note(fam, is_dt)
        rows.append(row)

    columns = [
        "column",
        "dtype",
        "missing_pct",
        "n_unique",
        "example_values",
        "looks_like_datetime",
        "looks_like_id",
        "looks_like_binary",
        "looks_like_target_or_treatment",
    ]
    if include_feature_family:
        columns.append("feature_family")
    if include_notes:
        columns.append("notes")
    return pd.DataFrame(rows, columns=columns)


def _column_note(family: str, is_dt: bool) -> str:
    """Короткая эвристическая пометка для аудиторской сводки (не строгая разметка)."""
    if family == "anonymized":
        return "anonymized snapshot — нельзя размечать как safe B/C"
    if family == "datetime" or is_dt:
        return "datetime — потенциальное timing-evidence для in-time C/D"
    if family == "id":
        return "id-like"
    if family == "target_or_treatment":
        return "имя похоже на target/treatment — проверить роль"
    return "named feature — A при pre-treatment; B/C только при защитимом timing"
