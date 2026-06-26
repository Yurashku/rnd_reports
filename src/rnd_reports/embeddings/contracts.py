"""Контракты входных/выходных схем адаптеров R&D-7 (без Spark-зависимостей).

Только проверки структуры и имена колонок — **без алгоритмов**. Сами функции
(PCA-снижение, propensity) живут в :mod:`reducer` / :mod:`propensity`.

Формат сырой таблицы эмбеддингов: ``epk_id, report_dt, emb_0_val, emb_1_val, ...``.
Трит (``treatment``) — опциональная колонка того же датафрейма (отдельной таблицы нет).
Поддерживается и легаси-формат ``col_000, col_001, ...`` (старая витрина R&D-7), чтобы
один контракт обслуживал оба ноутбука.

Хелперы принимают любой объект с атрибутом ``.columns`` (в т.ч. pyspark ``DataFrame``),
поэтому pyspark здесь не импортируется.
"""

from __future__ import annotations

import re

# --- имена ключевых колонок и префиксов ---
EPK_ID = "epk_id"
REPORT_DT = "report_dt"
TREATMENT = "treatment"
KEY_COLUMNS = (EPK_ID, REPORT_DT)

EMBEDDING_PREFIX = "emb_"  # входные эмбеддинги: emb_0_val, emb_1_val, ...
REDUCED_PREFIX = "red_"  # выход reducer'а: red_0, red_1, ...
PROPENSITY_SCORE = "prop_score"  # выход propensity-адаптера

# emb_{i}_val (основной формат) либо col_{i} (легаси-витрина); номер i — для сортировки.
_EMB_NEW = re.compile(r"^emb_(\d+)_val$")
_EMB_LEGACY = re.compile(r"^col_(\d+)$")


def _embedding_index(name: str) -> int | None:
    """Числовой индекс эмбеддинг-колонки или ``None``, если имя не подходит."""
    match = _EMB_NEW.match(name) or _EMB_LEGACY.match(name)
    return int(match.group(1)) if match else None


def embedding_feature_columns(df) -> list[str]:
    """Эмбеддинг-колонки (``emb_{i}_val`` или легаси ``col_{i}``) по возрастанию индекса."""
    indexed = [(idx, c) for c in df.columns if (idx := _embedding_index(c)) is not None]
    return [name for _, name in sorted(indexed)]


def reduced_column_names(red_size: int) -> list[str]:
    """Имена выходных колонок reducer'а: ``red_0 ... red_{red_size-1}``."""
    if red_size < 1:
        raise ValueError(f"red_size должен быть >= 1, получено {red_size}")
    return [f"{REDUCED_PREFIX}{i}" for i in range(red_size)]


def validate_embedding_schema(df) -> list[str]:
    """Проверить схему эмбеддинг-датасета; вернуть список эмбеддинг-колонок.

    Требуется ``epk_id``, ``report_dt`` и >=1 эмбеддинг-колонка (``emb_{i}_val``/``col_{i}``).
    """
    cols = set(df.columns)
    missing = [c for c in KEY_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"В эмбеддинг-датасете нет ключевых колонок: {missing}")
    embeddings = embedding_feature_columns(df)
    if not embeddings:
        raise ValueError(
            "Не найдено ни одной эмбеддинг-колонки формата 'emb_{i}_val' (или легаси 'col_{i}')"
        )
    return embeddings
