# R&D-6 «Safe in-time CUPAC» — план реализации (итеративный, по PR)

- **Версия:** v1.0
- **Статус:** активный
- **Связанный контекст:** [`06_safe_intime_cupac_context.md`](06_safe_intime_cupac_context.md) (v0.2)
- **Связанный R&D:** `rnd/06_safe_intime_cupac/`

Документ фиксирует разбиение R&D-6 на крупные шаги (по одному PR), их цели,
ожидаемые файлы, критерии приёмки, риски и non-goals. Это «большой» R&D — он
**не реализуется одним PR**.

## Принципы
- Каждый шаг — отдельный PR в `main`, небольшой и проверяемый.
- Алгоритмы добавляются постепенно; общий код живёт в `src/rnd_reports/`,
  ноутбук RnD-6 — тонкая витрина (конвенция репозитория).
- Синтетика и реальные датасеты используют **один** интерфейс `BenchmarkDataset`.
- Реальные/корпоративные данные **не коммитятся**.

## Целевая архитектура (куда идём)
```
datasets/        BenchmarkDataset (data, id, treatment, target, feature_registry)
                 catalog (метаданные) → loaders (локальные файлы) → adapters (→ BenchmarkDataset)
feature_safety/  FeatureClass A–F + unsafe_demo, FeatureRegistry, gate/diagnostics (позже)
variance_reduction/ sklearn CUPAC (A), linear second-stage (B, C), hypex parity-адаптер
benchmark/       method_registry (цепочка), protocol (оценка), reporting (таблица §10 + график)
```

Цепочка прироста: `ab_hypex → sklearn_cupac_A → +B_linear → +C_linear`;
`hypex_cupac` — reference/parity (вне цепочки); `unsafe_demo_optional` — только демонстрация.

## Классы признаков (из контекста §3–4)
`A` pre-treatment → CUPAC · `B` expert-safe in-time → linear second-stage ·
`C` balance-gated in-time → linear second-stage (после gate) · `D` DAG-required → вне estimator ·
`E` mediator-risk → запрещён · `F` leakage → запрещён · `unsafe_demo` → demonstration-only.

## Последовательность шагов (PR)

### Step 1 — Фундамент + этот план (текущий PR)
- **Цель:** зафиксировать план; добавить контракты/скаффолдинг без алгоритмов.
- **Файлы:** `docs/06_safe_intime_cupac_implementation_plan.md`;
  `feature_safety/contracts.py` (FeatureClass, FeatureSpec, политики), `feature_safety/registry.py`
  (FeatureRegistry + build_feature_registry); `datasets/{__init__,contracts,catalog,loaders,adapters}.py`;
  `benchmark/{contracts,method_registry}.py`; `.gitignore` (`/data/`); тесты `tests/test_rnd6_imports.py`,
  `tests/test_rnd6_dataset_contracts.py`; ссылка из `README.md`.
- **Приёмка:** пакет импортируется; `BenchmarkDataset` валидирует структуру; `METHODS` описывает 6 методов
  с корректной цепочкой; существующие тесты зелёные; данных в git нет; алгоритмов нет.

### Step 2 — BenchmarkDataset в деле + первый синтетический генератор
- **Цель:** минимальный генератор синтетики с размеченными A–F (+unsafe_demo) и известным истинным ATE;
  синтетический adapter → `BenchmarkDataset`.
- **Файлы:** `synthetic/schemas.py`, `synthetic/scenarios.py` (генератор + 1–2 сценария),
  `datasets/adapters.py` (синтетический adapter), `configs/06_safe_intime_cupac/synthetic_scenarios.yaml`.
- **Приёмка:** воспроизводимость по seed; колонки соответствуют реестру; raw diff-in-means ≈ истинный ATE.

### Step 3 — Метрики и таблица результатов + A/B протокол
- **Цель:** оценка ATE (рандомизированный A/B), сборка строки результата (§10), относительные/
  инкрементальные метрики дисперсии и выборки.
- **Файлы:** `benchmark/protocol.py` (estimate_ate, сборка `MethodResult`), `benchmark/reporting.py`
  (таблица), `metrics` при необходимости.
- **Приёмка:** `ab_hypex`/raw считается; таблица содержит все колонки §10; формулы §7 покрыты тестами.

### Step 4 — sklearn_cupac_A + hypex_cupac (parity)
- **Цель:** основной CUPAC baseline на признаках A; reference-сверка с HypEx CUPAC.
- **Файлы:** `variance_reduction/local_cupac.py` (feature-list CUPAC), `variance_reduction/hypex_cupac_adapter.py`.
- **Приёмка:** `sklearn_cupac_A` снижает дисперсию vs A/B; `hypex_cupac` сопоставим (parity) и помечен `reference_only`.

### Step 5 — +B_linear (expert-safe in-time)
- **Цель:** линейная second-stage коррекция поверх CUPAC A по классу B.
- **Файлы:** `variance_reduction/safe_intime_adjustment.py`; расширение протокола.
- **Приёмка:** инкрементальный выигрыш B vs `sklearn_cupac_A` считается; B не смещает ATE на синтетике.

### Step 6 — balance/missingness gate + +C_linear
- **Цель:** практический safety-gate (§3C) и коррекция по прошедшим gate признакам C.
- **Файлы:** `feature_safety/rules.py`, `feature_safety/diagnostics.py` (реализовать вместо заглушек),
  расширение протокола.
- **Приёмка:** gate отклоняет несбалансированные/leakage-подобные; `+C_linear` даёт инкрементальный выигрыш;
  E/F никогда не проходят в estimator.

### Step 7 — unsafe_demo + визуализация + витрина-отчёт
- **Цель:** `unsafe_demo_optional` (явно `safety_status=unsafe_demo`); график ATE±CI по всем методам;
  заполнить `rnd/06_safe_intime_cupac/{notebook.ipynb,report.md}` и собрать `report.pdf`.
- **Приёмка:** график показывает сужение CI и вклад B/C; unsafe-demo визуально отделён; отчёт собран.

### Step 8 — адаптеры реальных датасетов (по одному)
- **Цель:** довести loaders/adapters до рабочего состояния для 1+ датасета (Hillstrom → Lenta → Criteo → …).
- **Файлы:** `datasets/loaders.py`, `datasets/adapters.py`, доки по ручной загрузке.
- **Приёмка:** при наличии локального файла датасет приводится к `BenchmarkDataset`; бенчмарк гоняется;
  если датасет не настоящий RCT — это явно помечается в отчёте; **данные не коммитятся**.

## Как вводятся адаптеры open-source датасетов
`catalog.DatasetSpec` (метаданные: источник, лицензия, рандомизация, подсказка загрузки) →
`loaders.LocalFileLoader` (читает локальный файл из gitignored `data/06_safe_intime_cupac/<name>/`,
ничего не качает) → `adapters.DatasetAdapter` (предобработка + разметка признаков по A–F →
`BenchmarkDataset`). Каждый датасет добавляется отдельным шагом и требует ручной разметки признаков.

## Локальные данные и git
- Скачанные датасеты кладутся в **`data/06_safe_intime_cupac/<name>/`** (корень репозитория).
- Директория `/data/` добавлена в `.gitignore` — **реальные данные не попадают в git**.
- Loader при отсутствии файла бросает информативную ошибку с источником/лицензией (см. `catalog`).

## Синтетика и реальные данные — общий интерфейс
Оба источника приводятся к `BenchmarkDataset(data, id_col, treatment_col, target_col, feature_registry)`.
Бенчмарк (protocol/method_registry/reporting) не зависит от происхождения данных — отличается только
adapter и разметка признаков. На синтетике классы A–F известны по построению; на реальных — задаются
экспертно/предобработкой.

## unsafe_demo: жёсткое отделение от кандидатов
`unsafe_demo` — отдельный `FeatureClass`; метод `unsafe_demo_optional` имеет `MethodKind.DEMO`,
`predecessor=None`, `expected_safety_status=unsafe_demo`. Он **не** входит в predecessor-chain и в
наборы кандидатных методов; в таблице/графике помечается `safety_status=unsafe_demo`. E/F-признаки
запрещены к использованию в любом кандидатном методе (политики в `feature_safety/contracts.py`).

## HypEx baselines и доступность
- `ab_hypex` и `hypex_cupac` строятся на HypEx (высокоуровневый `ABTest`, при необходимости
  `enable_cupac=True`).
- **Если HypEx недоступен в окружении — его нужно установить** (`pip install hypex`; объявлен как
  optional-зависимость в `pyproject.toml`). На момент написания HypEx 1.0.5 присутствует в `.venv`.
- Graceful-`unavailable` (метод помечается `safety_status=unavailable`, метрики NaN, пайплайн не падает)
  оставляем только как крайний fallback для офлайн-CI, где установка невозможна.

## Риски
- Несоответствие схемы реальных датасетов формату HypEx CUPAC (лаги/cofounders) — решается на уровне adapter.
- balance-gate не доказывает причинную безопасность (только практический фильтр после исключения E/F) — отражать в отчёте.
- Конфликты зависимостей при установке HypEx (например, пины numpy) — фиксировать совместимые версии.
- Реальные датасеты могут быть не настоящими RCT — помечать как demonstration/semi-synthetic.

## Non-goals (весь R&D, первый этап)
Полный DAG; CAIMAN как обязательный компонент; causal discovery; автоматическое разрешение D-признаков;
перенос кода в HypEx; production-ready API; использование E/F в основном estimator; смешивание всех
in-time признаков в одну ML-модель без классов безопасности.
