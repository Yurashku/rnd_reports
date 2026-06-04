# CLAUDE.md — карта репозитория и правила работы

> Этот файл синхронизирован с [`AGENTS.md`](AGENTS.md): общий блок «Карта репозитория»,
> «Канон RnD», «Тулинг», «Политика данных» и «Конвенции» в обоих файлах совпадают. При
> изменении одного — обнови второй.

R&D-репозиторий по экспериментам вокруг HypEx. Два слоя: **отчётные RnD** (`rnd/NN_<topic>/`)
и **переиспользуемый код** (`src/rnd_reports/`, src-layout). Ноутбук — тонкая витрина над пакетом.

## Карта репозитория

```
rnd/0N_<topic>/      # отчётные RnD — ТОЛЬКО report-oriented (см. «Канон RnD»)
src/rnd_reports/     # общий переиспользуемый код (устанавливаемый пакет)
tools/               # общий CLI: generate_pdf.py, execute_notebooks.py, audit_datasets.py
tests/               # тесты пакета (pytest)
docs/0N_*.md         # контекст/ТЗ/методология по RnD + общие политики
configs/0N_<topic>/  # конфиги экспериментов по RnD
data/0N_<topic>/     # gitignored локальные датасеты (НЕ коммитим)
results/0N_<topic>/  # компактные КОММИТИМЫЕ артефакты: CSV-сводки, figures/
archive/             # исторические/демо-ноутбуки вне основной линии
planned_rnd/         # шаблон для новых RnD
```

Параллельные per-RnD папки имеют единый префикс `0N_<topic>` в `docs/`, `configs/`, `data/`,
`results/` — так всё, что относится к одному RnD, легко найти.

### `src/rnd_reports/` (подпакеты)
- `datasets/` — `BenchmarkDataset`/`ResearchDataset` контракты, каталог, loaders, `inspect`
  (schema_summary), `adapters` (synthetic + Hillstrom) и `expanded_adapters` (реальные/sandbox).
- `feature_safety/` — таксономия классов A–F/unsafe, реестр, правила, диагностики (balance gate).
- `variance_reduction/` — CUPED, local CUPAC, метрики, safe in-time linear-коррекция, HypEx-адаптер.
- `benchmark/` — протокол A/B + цепочка методов, реестр методов, контракты результата, отчётность.
- `synthetic/` — генераторы и сценарии синтетических данных (известные A–F по построению).
- `embeddings/` — RnD-7: тулкит-адаптеры pyspark-эмбеддингов (`reducer.py` PCA-снижение,
  `propensity.py` propensity-score с in-time safety); pyspark — опциональный extra `[spark]`.

## Канон RnD (однородность)

Каждая `rnd/0N_<topic>/` содержит **ровно**: `notebook.ipynb`, `report.md`, `report.pdf`,
`README.md`. Никаких `*.csv`/`*.png`/`*.py`/`*.txt` внутри rnd-папок — генерируемые артефакты
идут в `results/0N_<topic>/`, контекст — в `docs/`, переиспользуемый код — в `src/rnd_reports/`.

- `report.md` — единственный источник содержания; `report.pdf` собирается из него автоматически.
- `report.md` ссылается на соответствующий код/ноутбук; графики встраиваются в PDF через
  `![alt](../../results/0N_<topic>/figures/...png)`.

## Тулинг

```bash
python tools/generate_pdf.py          # пересобрать ВСЕ rnd/*/report.pdf из report.md
python tools/execute_notebooks.py     # выполнить все rnd/*/notebook.ipynb
python tools/audit_datasets.py --delta # RnD-6: delta-эффекты + sandbox-диагностика → results/
```
- `generate_pdf.py` — matplotlib-рендер (шрифт DejaVu → читаемая кириллица, markdown-таблицы
  сеткой, встраивание картинок). Автодискавери `rnd/*/report.md`.
- `audit_datasets.py` — RnD-6-специфичный CLI; пишет артефакты в `results/06_safe_intime_cupac/`.

## Политика данных
- Реальные/корпоративные/скачанные данные **никогда не коммитятся** — только в gitignored
  `data/0N_<topic>/`; адаптеры/loaders читают оттуда и падают понятной ошибкой при отсутствии.
- В git идут только код, отчёты и **компактные** сводки/фигуры в `results/`.
- Опциональные тяжёлые пакеты (scikit-uplift, obp, datasets, pyreadr, …) ставятся только в
  локальный `.venv` и **не** добавляются в обязательные зависимости проекта.

## Конвенции
- Все отчёты и комментарии в коде — на русском языке.
- Ноутбуки `*.ipynb` перед коммитом исполнены и сохранены с актуальными output.
- Эксперименты — на синтетике (фиксированный seed) и на локальных open-source датасетах.
- Алгоритмическая логика живёт в `src/rnd_reports/` и покрыта тестами; ноутбук — витрина.
- Не трогать выводы/содержание уже закрытых RnD (01–05) без явного запроса.

## Проверки
```bash
.venv/bin/python -m pytest -q         # тесты пакета
python tools/generate_pdf.py          # PDF собираются без ошибок
```
