"""Конфигурация pytest для репозитория.

Добавляет ``src/`` в ``sys.path``, чтобы пакет ``rnd_reports`` импортировался в
тестах и ноутбуках без обязательной установки (``pip install -e .``).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
