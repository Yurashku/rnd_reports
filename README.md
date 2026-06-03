# R&D-репозиторий по экспериментам HypEx

Репозиторий разделён на два слоя:

- **Отчётные RnD** — каждая директория `rnd/NN_<topic>/` остаётся **report-oriented** и содержит:
  - `notebook.ipynb` — воспроизводимые эксперименты на синтетике;
  - `report.md` — единый источник правды отчёта;
  - `report.pdf` — автогенерация из `report.md`;
  - `README.md` — короткое описание (опционально).
- **Общий код** — переиспользуемая логика вынесена из ноутбуков в устанавливаемый пакет `src/rnd_reports/` (src-layout). Ноутбуки становятся тонкой витриной поверх этого пакета.

## Структура репозитория

```
rnd/                 # отчётные RnD-директории
src/rnd_reports/     # общий переиспользуемый код
configs/             # конфиги экспериментов (по RnD)
docs/                # контекст, ТЗ, методология, политики
tests/               # тесты пакета
tools/               # генерация PDF и пакетный прогон ноутбуков
archive/             # исторические материалы (вне основной линии)
```

### RnD и их статусы

| # | Директория | Статус | Кратко |
|---|------------|--------|--------|
| 01 | `rnd/01_bonferroni_aa_matching/` | ✅ done | Множественные проверки и матчинг в A/A. |
| 02 | `rnd/02_pyspark_fast_aa/` | ✅ done | Быстрый A/A на PySpark. |
| 03 | `rnd/03_autoconfig_homogeneity_split/` | ✅ done | Автоконфиг и контроль однородности сплита. |
| 04 | `rnd/04_faiss_matcher_tradeoff/` | 📄 report-only | FAISS-матчинг и trade-off; разработка ведётся вне этого репозитория. |
| 05 | `rnd/05_iv_cupac_policy/` | ⏸ paused | Пилот с one-sided noncompliance: baseline A/B, Wald LATE, 2SLS, CUPAC, policy-based OPE. |
| 06 | `rnd/06_safe_intime_cupac/` | 🚧 active | «Safe in-time covariates для CUPAC-style снижения дисперсии». |

### Общий код — `src/rnd_reports/`

Устанавливаемый пакет (src-layout) с подпакетами:

- `variance_reduction/` — методы снижения дисперсии:
  - `cuped.py` — классическая CUPED θ-корректировка;
  - `local_cupac.py` — локальная R&D-реализация CUPAC (портирована из VarWar);
  - `metrics.py` — расчёт снижения дисперсии и сводки;
  - `hypex_cupac_adapter.py`, `safe_intime_adjustment.py` — заглушки RnD-6 (см. роадмап);
- `feature_safety/` — контракты, реестр, правила и диагностики «безопасности» in-time ковариат (заглушки RnD-6);
- `synthetic/` — генераторы синтетических данных и сценарии (`generators.py` портирован из VarWar);
- `benchmark/` — протокол, раннеры и отчётность сравнений (заглушки RnD-6).

Статус кода: миграция базовых блоков из VarWar выполнена (`synthetic.generators`, `variance_reduction.{cuped,metrics,local_cupac}`); модули самого RnD-6 пока заглушки. См. [`docs/migration_from_varwar.md`](docs/migration_from_varwar.md).

### Сопутствующее
- `configs/06_safe_intime_cupac/` — сценарии, наборы ковариат, методы.
- `docs/` — `CONTEXT_TZ.md`, `06_safe_intime_cupac_context.md`, `feature_safety_policy.md`, `variance_reduction_methodology.md`, `migration_from_varwar.md`.
- `tests/` — тесты пакета (импорт, генераторы, CUPED, local CUPAC, контракт заглушек).
- `pyproject.toml` — конфигурация пакета; `conftest.py` добавляет `src/` в `sys.path`, поэтому импорт работает и без установки.
- `archive/` — демо-ноутбуки RnD-5 (`rnd5_late_iv_offpolicy_demo*.ipynb`), вынесенные из корня.

## Быстрый старт

```bash
# 1) Зависимости
pip install -r requirements.txt
# или установка пакета для разработки:
pip install -e .[ml]

# 2) Пересобрать PDF-отчёты из Markdown (по всем rnd/*/report.md)
python tools/generate_pdf.py

# 3) (опционально) Пакетно выполнить ноутбуки — перезаписывает их output!
python tools/execute_notebooks.py
```

`tools/generate_pdf.py` и `tools/execute_notebooks.py` автоматически находят цели через `rnd/*/report.md` и `rnd/*/notebook.ipynb` — новые RnD подхватываются без правки скриптов.

## Принципы качества
- Все отчёты и текстовые комментарии в коде — на русском языке.
- Реальные/корпоративные данные **не коммитятся**; эксперименты — на синтетике с фиксированным seed.
- Графики и артефакты остаются внутри ноутбуков (без внешних `png/csv/txt` в RnD-директориях).
- `report.md` — единственный источник содержания; `report.pdf` пересобирается автоматически.
- Алгоритмическая логика живёт в `src/rnd_reports/` и покрывается тестами; ноутбук — витрина.

## Зависимости отдельных RnD
- RnD-5 (paused), library-first: `linearmodels` (2SLS/LATE), `hypex` (CUPAC, A/B-summary), `OffPolicyLab` (OPE):
  ```bash
  pip install linearmodels hypex
  pip install git+https://github.com/Yurashku/OffPolicyLab.git
  ```
- RnD-6 / общий код: `scikit-learn` (обязательно для тестов), `catboost` (опционально). Ставятся через `pip install -e .[ml]`.
