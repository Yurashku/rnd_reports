"""Контракты входных/выходных схем адаптеров R&D-7 (без Spark-зависимостей).

Step 1: только проверки структуры и имена колонок — **без алгоритмов**. Сами
адаптеры (PCA-снижение, propensity) живут в :mod:`reducer` / :mod:`propensity`.

Хелперы принимают любой объект с атрибутом ``.columns`` (в т.ч. pyspark
``DataFrame``), поэтому pyspark здесь не импортируется.
"""

from __future__ import annotations

# --- имена ключевых колонок и префиксов ---
EPK_ID = "epk_id"
REPORT_DT = "report_dt"
TREATMENT = "treatment"
KEY_COLUMNS = (EPK_ID, REPORT_DT)

EMBEDDING_PREFIX = "col_"  # входные эмбеддинги: col_000, col_001, ...
REDUCED_PREFIX = "emb_"  # выход reducer'а: emb_000, emb_001, ...
PROPENSITY_SCORE = "propensity_score"  # выход propensity-адаптера


def _embedding_sort_key(name: str) -> tuple[int, object]:
    """Натуральная сортировка ``col_*``: числовой суффикс — как число, иначе строка."""
    suffix = name[len(EMBEDDING_PREFIX) :]
    return (0, int(suffix)) if suffix.isdigit() else (1, suffix)


def embedding_feature_columns(df) -> list[str]:
    """Эмбеддинг-колонки (префикс ``col_``) в естественном порядке индексов."""
    cols = [c for c in df.columns if c.startswith(EMBEDDING_PREFIX)]
    return sorted(cols, key=_embedding_sort_key)


def reduced_column_names(reducted_shape: int) -> list[str]:
    """Имена выходных колонок reducer'а: ``emb_000 ... emb_{reducted_shape-1}``."""
    if reducted_shape < 1:
        raise ValueError(f"reducted_shape должен быть >= 1, получено {reducted_shape}")
    return [f"{REDUCED_PREFIX}{i:03d}" for i in range(reducted_shape)]


def validate_embedding_schema(df) -> list[str]:
    """Проверить схему эмбеддинг-датасета; вернуть список эмбеддинг-колонок.

    Требуется ``epk_id``, ``report_dt`` и >=1 колонка с префиксом ``col_``.
    """
    cols = set(df.columns)
    missing = [c for c in KEY_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"В эмбеддинг-датасете нет ключевых колонок: {missing}")
    embeddings = embedding_feature_columns(df)
    if not embeddings:
        raise ValueError(
            f"Не найдено ни одной эмбеддинг-колонки с префиксом {EMBEDDING_PREFIX!r}"
        )
    return embeddings


def validate_treatment_schema(df) -> None:
    """Проверить схему датасета трита: нужны ``epk_id``, ``report_dt``, ``treatment``."""
    cols = set(df.columns)
    missing = [c for c in (*KEY_COLUMNS, TREATMENT) if c not in cols]
    if missing:
        raise ValueError(f"В датасете трита нет колонок: {missing}")
