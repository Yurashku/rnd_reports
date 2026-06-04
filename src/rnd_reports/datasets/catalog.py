"""Каталог open-source uplift-датасетов — кандидатов для бенчмарка R&D-6.

Только метаданные (имя, тип, источник, лицензия, рандомизация, подсказка по
загрузке). **Никаких скачиваний и данных в git.** Реальные файлы кладутся в
gitignored-директорию ``loaders.DEFAULT_DATA_DIR`` вручную.

Ссылки на Criteo: https://arxiv.org/abs/2111.10106 , https://arxiv.org/abs/2604.06123
"""

from __future__ import annotations

from dataclasses import dataclass

# Тип датасета.
KIND_SYNTHETIC = "synthetic"
KIND_REAL = "real"
KIND_SEMI_SYNTHETIC = "semi_synthetic"


@dataclass(frozen=True)
class DatasetSpec:
    """Метаданные датасета-кандидата (без самих данных)."""

    name: str
    kind: str
    is_randomized: bool | None  # None = требует проверки
    source_url: str = ""
    license: str = ""
    download_hint: str = ""
    notes: str = ""


# Первичный список кандидатов (см. контекст §9.2). is_randomized=None — нужно
# подтвердить перед использованием как полноценной causal-проверки.
CANDIDATE_DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="hillstrom",
        kind=KIND_REAL,
        is_randomized=True,
        source_url="https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html",
        license="public (MineThatData)",
        download_hint="Скачать CSV вручную и положить в <DATA_DIR>/hillstrom/.",
        notes="Email marketing; treatment = тип письма; рандомизированный. A-only (нет B/C).",
    ),
    DatasetSpec(
        name="lenta",
        kind=KIND_REAL,
        is_randomized=None,  # RCT в docs явно не доказан — нужна проверка
        source_url="https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html",
        license="см. условия источника",
        download_hint="sklift.datasets.fetch_lenta(data_home=<DATA_DIR>) (вне git).",
        notes="Retail uplift; 687k×195 именованных ПРЕ-кампейн агрегатов (15d/1m/3m/6m/12m) → "
        "A-only, реальных B/C нет. См. docs/06_real_dataset_raw_audit.md.",
    ),
    DatasetSpec(
        name="criteo",
        kind=KIND_REAL,
        is_randomized=True,
        source_url="https://arxiv.org/abs/2111.10106",
        license="Criteo research license",
        download_hint="sklift.datasets.fetch_criteo(percent10=True, data_home=<DATA_DIR>) — "
        "на момент аудита S3 отдаёт HTTP 403 (локально не подтверждён).",
        notes="RCT; f0..f11 анонимны → слабый A-only; exposure = E/mediator (unsafe_demo), не B/C.",
    ),
    DatasetSpec(
        name="x5_retailhero",
        kind=KIND_REAL,
        is_randomized=None,  # treatment_flg≈50/50, но рандомизация не подтверждена
        source_url="https://ods.ai/competitions/x5-retailhero-uplift-modeling",
        license="competition terms",
        download_hint="sklift.datasets.fetch_x5(data_home=<DATA_DIR>) (multi-table).",
        notes="Multi-table; есть реальный transaction_datetime, но дата кампании скрыта → "
        "A-only через инженерию истории; защитимых B/C нет.",
    ),
    DatasetSpec(
        name="megafon",
        kind=KIND_SYNTHETIC,  # сгенерированные telecom-like данные — не реальная валидация
        is_randomized=True,
        source_url="https://www.uplift-modeling.com/en/latest/api/datasets/fetch_megafon.html",
        license="competition terms",
        download_hint="sklift.datasets.fetch_megafon(data_home=<DATA_DIR>).",
        notes="Синтетический/сгенерированный telecom uplift; X_1..X_50 анонимны. "
        "Годен только как sanity/scale-бенчмарк, НЕ реальная валидация.",
    ),
    DatasetSpec(
        name="orange_belgium",
        kind=KIND_REAL,
        is_randomized=True,
        source_url="https://arxiv.org/abs/2312.07206",
        license="см. репозиторий (Verhelst 2025)",
        download_hint="Прямой CSV (Dropbox из github TheoVerhelst/Churn-Uplift-Dataset-Paper) "
        "или OpenML id 45580 → <DATA_DIR>/orange_belgium/.",
        notes="Telecom churn uplift (11896×180); признаки полностью анонимны "
        "(PC1..PC160 — PCA, FACTOR1..18) → A-only, B/C/timing нет.",
    ),
    DatasetSpec(
        name="lazada_descn",
        kind=KIND_REAL,
        is_randomized=None,  # test рандомизирован; train biased by prior policy
        source_url="https://github.com/kailiang-zhong/DESCN",
        license="research/non-commercial (см. репозиторий)",
        download_hint="Прямой zip lzd_data_public.zip (Dropbox из репозитория DESCN) → "
        "<DATA_DIR>/lazada_descn/.",
        notes="E-commerce voucher uplift; f0..f82 анонимны. test=181669 (random), "
        "train=926669 (biased) → A-only на рандомизированном тесте; B/C нет.",
    ),
]

CATALOG: dict[str, DatasetSpec] = {d.name: d for d in CANDIDATE_DATASETS}


def get_dataset_spec(name: str) -> DatasetSpec:
    return CATALOG[name]
