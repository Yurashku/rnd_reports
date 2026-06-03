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
        notes="Email marketing; treatment = тип письма; рандомизированный.",
    ),
    DatasetSpec(
        name="lenta",
        kind=KIND_REAL,
        is_randomized=True,
        source_url="https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html",
        license="см. условия источника",
        download_hint="Скачать вручную; положить в <DATA_DIR>/lenta/.",
        notes="Lenta uplift; доступен через sklift.datasets (вне git).",
    ),
    DatasetSpec(
        name="criteo",
        kind=KIND_REAL,
        is_randomized=True,
        source_url="https://arxiv.org/abs/2111.10106",
        license="Criteo research license",
        download_hint="Criteo Uplift v2.1 (~13.98M строк); скачать вручную в <DATA_DIR>/criteo/.",
        notes="Крупный uplift benchmark, near-random assignment.",
    ),
    DatasetSpec(
        name="x5_retailhero",
        kind=KIND_REAL,
        is_randomized=None,
        source_url="https://ods.ai/competitions/x5-retailhero-uplift-modeling",
        license="competition terms",
        download_hint="Регистрация на платформе; данные в <DATA_DIR>/x5_retailhero/.",
        notes="RetailHero uplift; проверить условия рандомизации/лицензию.",
    ),
    DatasetSpec(
        name="megafon",
        kind=KIND_REAL,
        is_randomized=None,
        source_url="https://ods.ai/competitions/megafon-and-ods-mlcup",
        license="competition terms",
        download_hint="Скачать вручную в <DATA_DIR>/megafon/.",
        notes="MegaFon uplift; проверить рандомизацию и лицензию.",
    ),
]

CATALOG: dict[str, DatasetSpec] = {d.name: d for d in CANDIDATE_DATASETS}


def get_dataset_spec(name: str) -> DatasetSpec:
    return CATALOG[name]
