# R&D-репозиторий по экспериментам HypEx

Репозиторий разделён на два слоя:

- **Отчётные RnD** — каждая директория `rnd/NN_<topic>/` остаётся **report-oriented**:
  - `notebook.ipynb` — обязателен; воспроизводимые эксперименты на синтетике/внешних данных;
  - `notebook_internal.ipynb` — опционален; самодостаточный ноутбук для **внутреннего контура** (код вшит в ячейки, таблица читается из `INPUT_CSV`, отдельная ячейка-генератор синтетики);
  - `report.md` — обязателен; единый источник правды отчёта;
  - `report.pdf` — опционален; собирается только для RnD из `PDF_ENABLED`;
  - `README.md` — обязателен; короткое описание RnD.
- **Общий код** — переиспользуемая логика вынесена из ноутбуков в устанавливаемый пакет `src/rnd_reports/` (src-layout). Ноутбуки становятся тонкой витриной поверх этого пакета.

Рабочий процесс целиком — в [`docs/WORKFLOW.md`](docs/WORKFLOW.md). Карта репозитория и правила для агентов — в [`CLAUDE.md`](CLAUDE.md) и синхронизированном [`AGENTS.md`](AGENTS.md).

## Структура репозитория

```
rnd/0N_<topic>/      # отчётные RnD: notebook.ipynb, report.md, README.md (+ опц. notebook_internal.ipynb, report.pdf)
src/rnd_reports/     # общий переиспользуемый код (устанавливаемый пакет)
tools/               # общий CLI: generate_pdf, execute_notebooks, audit_datasets
tests/               # тесты пакета
docs/0N_*.md         # контекст, ТЗ, методология, политики (по RnD)
configs/0N_<topic>/  # конфиги экспериментов (по RnD)
data/0N_<topic>/     # gitignored локальные датасеты (НЕ коммитим)
results/0N_<topic>/  # компактные коммитимые артефакты: CSV-сводки, figures/
archive/             # исторические/демо-ноутбуки (вне основной линии)
```

Сопутствующее по каждому RnD лежит под единым префиксом `0N_<topic>` в `docs/`, `configs/`, `data/`, `results/` — так всё, что относится к одному RnD, легко найти.

### RnD и их статусы

| # | Директория | Статус | Кратко |
|---|------------|--------|--------|
| 01 | `rnd/01_bonferroni_aa_matching/` | ✅ done | Множественные проверки и матчинг в A/A. |
| 02 | `rnd/02_pyspark_fast_aa/` | ✅ done | Быстрый A/A на PySpark. |
| 03 | `rnd/03_autoconfig_homogeneity_split/` | ✅ done | Автоконфиг и контроль однородности сплита. |
| 04 | `rnd/04_faiss_matcher_tradeoff/` | 📄 report-only | FAISS-матчинг и trade-off; разработка ведётся вне этого репозитория. |
| 05 | `rnd/05_iv_cupac_policy/` | ⏸ paused | Пилот с one-sided noncompliance: baseline A/B, Wald LATE, 2SLS, CUPAC, policy-based OPE. |
| 06 | `rnd/06_safe_intime_cupac/` | 🚧 active | «Safe in-time covariates для CUPAC-style снижения дисперсии». |
| 07 | `rnd/07_embedding_adjustment_set/` | 🚧 active | Эмбеддинги как adjustment set в нерандомизированном испытании (готов тулкит-адаптеры; эксперимент — заглушки). |
| 08 | `rnd/08_multiple_testing/` | 🚧 active | Сравнение методов множественного тестирования OR-региона (Bonferroni/Holm/WY/RW + BH): single-table + Monte-Carlo. |

### Общий код — `src/rnd_reports/`

Устанавливаемый пакет (src-layout) с подпакетами:

- `variance_reduction/` — методы снижения дисперсии:
  - `cuped.py` — классическая CUPED θ-корректировка;
  - `local_cupac.py` — локальная R&D-реализация CUPAC (портирована из VarWar);
  - `metrics.py` — расчёт снижения дисперсии и сводки;
  - `hypex_cupac_adapter.py` (parity-референс), `safe_intime_adjustment.py` (линейная second-stage коррекция B/C);
- `feature_safety/` — контракты, реестр, правила и диагностики «безопасности» in-time ковариат (классы A–F, balance gate);
- `datasets/` — `BenchmarkDataset`/`ResearchDataset`, каталог, loaders, `inspect`, `adapters` и `expanded_adapters` (реальные uplift-датасеты + research-песочницы);
- `synthetic/` — генераторы синтетических данных и сценарии (`generators.py` портирован из VarWar);
- `benchmark/` — протокол A/B, цепочка методов (`ab → CUPAC A → A+B → A+B+C`), реестр методов и отчётность.
- `embeddings/` — тулкит-адаптеры RnD-7 для pyspark-эмбеддингов (`epk_id, report_dt, emb_{i}_val`; легаси `col_*` распознаётся): `reducer.py` (`EmbeddingReducer`, StandardScaler+PCA-снижение до `red_size` → `red_*`), `propensity.py` (`PropensityScorer`, джойн с тритом + лучший по ROC-AUC из LogisticRegression/GBTClassifier → `prop_score`); оба с in-time safety (обучение на `report_dt <= cutoff`), `experiment.py` — реализованный causal-слой (numpy/sklearn): ATE с поправкой + диагностика баланса/overlap. pyspark — **опциональный extra** (`pip install -e .[spark]`); импорт пакета его не требует.

Статус кода: базовые блоки портированы из VarWar; ядро RnD-6 (feature_safety, datasets, benchmark, safe in-time CUPAC) реализовано и покрыто тестами. См. [`docs/migration_from_varwar.md`](docs/migration_from_varwar.md).

### Сопутствующее
- `configs/06_safe_intime_cupac/` — сценарии, наборы ковариат, методы.
- `configs/07_embedding_adjustment_set/` — дефолты адаптеров RnD-7 (`red_size`, propensity, in-time cutoff).
- `docs/` — `CONTEXT_TZ.md`, `06_safe_intime_cupac_context.md`, [`06_safe_intime_cupac_implementation_plan.md`](docs/06_safe_intime_cupac_implementation_plan.md) (план реализации RnD-6 по шагам/PR), `feature_safety_policy.md`, `variance_reduction_methodology.md`, `migration_from_varwar.md`.
- `tests/` — тесты пакета (импорт, генераторы, CUPED, local CUPAC, контракт заглушек).
- `pyproject.toml` — конфигурация пакета; `conftest.py` добавляет `src/` в `sys.path`, поэтому импорт работает и без установки.
- `results/06_safe_intime_cupac/` — коммитимые артефакты RnD-6 (delta-CSV, sandbox-диагностика, figures).
- `archive/` — исторические/демо-ноутбуки, вынесенные из rnd-папок (`rnd1_*`, `rnd3_*`, `rnd5_*`).

## Быстрый старт

```bash
# 1) Зависимости
pip install -r requirements.txt
# или установка пакета для разработки:
pip install -e .[ml]

# 2) Пересобрать PDF-отчёты из Markdown (только RnD из PDF_ENABLED)
python tools/generate_pdf.py

# 3) (опционально) Пакетно выполнить ноутбуки — перезаписывает их output!
python tools/execute_notebooks.py
```

`tools/execute_notebooks.py` находит цели через `rnd/*/notebook*.ipynb` (и исследовательский, и
internal-ноутбук) — новые RnD подхватываются без правки скриптов. `tools/generate_pdf.py`
собирает PDF только для RnD из списка `PDF_ENABLED` (PDF опционален); явный путь к `report.md`
в аргументах перекрывает фильтр.

## Принципы качества
- Все отчёты и текстовые комментарии в коде — на русском языке.
- Эксперименты идут на синтетике (фиксированный seed) и на локально скачанных open-source датасетах — последние подключаются через адаптеры/loaders из gitignored-директории `data/` (см. `src/rnd_reports/datasets/`). Реальные/корпоративные/скачанные данные **никогда не коммитятся**.
- Внутри `rnd/0N_<topic>/` нет внешних `png/csv/txt` (исключение — `notebook_internal.ipynb`): компактные коммитимые артефакты (CSV-сводки, figures) живут в `results/0N_<topic>/`.
- `report.md` — единственный источник содержания; `report.pdf` опционален (см. `PDF_ENABLED`).
- Передача во внутренний контур — отдельным самодостаточным `notebook_internal.ipynb` внутри папки RnD (директории `exports/` нет); подробности — в [`docs/WORKFLOW.md`](docs/WORKFLOW.md).
- Перед push все изменённые ноутбуки перезапускаются и проходят до конца.
- Алгоритмическая логика живёт в `src/rnd_reports/` и покрывается тестами; ноутбук — витрина.

## Зависимости отдельных RnD
- RnD-5 (paused), library-first: `linearmodels` (2SLS/LATE), `hypex` (CUPAC, A/B-summary), `OffPolicyLab` (OPE):
  ```bash
  pip install linearmodels hypex
  pip install git+https://github.com/Yurashku/OffPolicyLab.git
  ```
- RnD-6 / общий код: `scikit-learn` (обязательно для тестов), `catboost` (опционально). Ставятся через `pip install -e .[ml]`.
