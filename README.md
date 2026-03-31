# R&D-репозиторий по экспериментам HypEx

Репозиторий приведён к унифицированному формату: в корне есть 5 тематических RnD-директорий, и в каждой лежат только три артефакта:
- `notebook.ipynb` — воспроизводимые эксперименты на синтетике;
- `report.md` — единый источник правды отчёта;
- `report.pdf` — автогенерация из `report.md`.

## Структура репозитория

### Актуальные RnD
1. `01_bonferroni_aa_matching/`
2. `02_pyspark_fast_aa/`
3. `03_autoconfig_homogeneity_split/`
4. `04_faiss_matcher_tradeoff/`
5. `05_rnd_iv_cupac_policy/` — финальный RnD-5 по сценарию пилота с one-sided noncompliance: baseline A/B, Wald LATE, 2SLS, CUPAC-комбинации и policy-based OPE.

### Служебные директории
- `tools/execute_notebooks.py` — пакетный запуск пяти целевых ноутбуков.
- `tools/generate_pdf.py` — генерация пяти `report.pdf` из `report.md`.
- `docs/CONTEXT_TZ.md` — контекст и целевое ТЗ, включая требования к layout.
- `legacy/` — архив исторических материалов, перенесённый без изменений из старой структуры.
- `planned_rnd/README.md` — памятка по формату для будущих исследований.

## Быстрый старт

### 1) Установить зависимости
```bash
pip install -r requirements.txt
```

### 2) Выполнить все ноутбуки (обязательно перед коммитом)
```bash
python tools/execute_notebooks.py
```

### 3) Пересобрать PDF-отчёты из Markdown
```bash
python tools/generate_pdf.py
```

## RnD-5: важные детали реализации
- Ноутбук: `05_rnd_iv_cupac_policy/notebook.ipynb`.
- Отчёт: `05_rnd_iv_cupac_policy/report.md` и `report.pdf`.
- В ноутбуке используется стратегия **library-first**:
  - `linearmodels.iv.model.IV2SLS` для 2SLS/LATE;
  - `hypex` для CUPAC и финального A/B-summary;
  - `OffPolicyLab` (<https://github.com/Yurashku/OffPolicyLab>) для OPE.
- Если конкретной библиотеки нет в окружении, соответствующий блок в ноутбуке **помечается как недоступный** (без самописной замены алгоритма).

## Принципы качества
- Все отчёты и текстовые комментарии в коде ведутся на русском языке.
- Все ноутбуки должны быть полностью выполнены и сохранены с актуальными output.
- Графики сохраняются только внутри ноутбуков (без внешних `png/csv/txt` в RnD-директориях).
- `report.md` — единственный источник содержания; `report.pdf` всегда пересобирается автоматически.

## Дополнение по зависимостям RnD-5
Рекомендуемая установка для полного режима:

```bash
pip install linearmodels hypex
pip install git+https://github.com/Yurashku/OffPolicyLab.git
```

## Legacy
Старый набор материалов (`rnd_01_ab_дизайн_экспериментов`) сохранён в `legacy/rnd_01_ab_дизайн_экспериментов` как архив истории, чтобы не потерять предыдущие артефакты и контекст миграции.
