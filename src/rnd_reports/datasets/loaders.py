"""Loaders реальных датасетов — заглушки Step 1 (ничего не качают, ничего не коммитят).

Реальные файлы кладутся вручную в gitignored-директорию ``DEFAULT_DATA_DIR``
(см. ``.gitignore``). Loader читает локальный файл, если он есть; иначе бросает
информативную ошибку с подсказкой, откуда взять данные. Полные реализации —
в последующих шагах (см. ``docs/06_safe_intime_cupac_implementation_plan.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import pandas as pd

from .catalog import get_dataset_spec

# Директория для СКАЧАННЫХ вручную датасетов. Относительно корня репозитория.
# Должна быть в .gitignore — данные не попадают в git.
DEFAULT_DATA_DIR = Path("data") / "06_safe_intime_cupac"


@runtime_checkable
class DatasetLoader(Protocol):
    """Контракт loader-а: вернуть «сырой» DataFrame датасета."""

    name: str

    def load(self, path: Optional[Path] = None) -> pd.DataFrame: ...


class LocalFileLoader:
    """Базовый loader: читает локальный файл из ``DEFAULT_DATA_DIR/<name>/``.

    Если файла нет — бросает ``FileNotFoundError`` с подсказкой из каталога.
    Ничего не скачивает (загрузка данных вне git — задача пользователя/CI).
    """

    def __init__(self, name: str, filename: str = "data.csv"):
        self.name = name
        self.filename = filename

    def default_path(self) -> Path:
        return DEFAULT_DATA_DIR / self.name / self.filename

    def load(self, path: Optional[Path] = None) -> pd.DataFrame:
        path = Path(path) if path is not None else self.default_path()
        if not path.exists():
            spec = get_dataset_spec(self.name)
            raise FileNotFoundError(
                f"Датасет '{self.name}' не найден по пути {path}. "
                f"Данные не коммитятся в git. {spec.download_hint} "
                f"Источник: {spec.source_url}"
            )
        # Минимальное чтение; специфичный парсинг — в полной реализации шага загрузчиков.
        return pd.read_csv(path)


# Реестр заглушек-loader-ов по кандидатам каталога.
LOADERS: dict[str, LocalFileLoader] = {
    "hillstrom": LocalFileLoader("hillstrom", "Kevin_Hillstrom_MineThatData_E-MailAnalytics.csv"),
    "lenta": LocalFileLoader("lenta", "lenta.csv"),
    "criteo": LocalFileLoader("criteo", "criteo-uplift-v2.1.csv"),
    "x5_retailhero": LocalFileLoader("x5_retailhero", "data.csv"),
    "megafon": LocalFileLoader("megafon", "megafon.csv"),
}


def get_loader(name: str) -> LocalFileLoader:
    return LOADERS[name]
