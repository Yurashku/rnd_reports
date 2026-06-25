# R&D-7. Эмбеддинги как adjustment set в нерандомизированном испытании

> Статус: 🔶 базовый эксперимент готов. Реализованы тулкит-адаптеры (pyspark) **и** causal-слой
> (numpy/sklearn): оценка ATE с поправкой, диагностика баланса/overlap. Источник чисел —
> `notebook.ipynb` (синтетика, seed=11).

## Цель

Проверить, могут ли клиентские эмбеддинги (в сокращённом виде) выступать как хотя бы
сколько-то хороший **adjustment set** в НЕрандомизированном испытании — то есть
обеспечивать достаточный контроль смещения отбора при оценке причинного эффекта
без рандомизации назначения воздействия.

## Данные

Исходный pyspark-датасет эмбеддингов имеет схему:

```
epk_id, report_dt (месячная гранулярность), emb_0_val, emb_1_val, ..., emb_n_val
```

Поддерживается и легаси-формат `col_000, col_001, ...` (старая витрина) — детектор колонок
распознаёт оба. Назначение воздействия хранится отдельно — таблица трита с колонками
`epk_id, treatment` (бинарный); `report_dt` опционален: если он есть, джойн идёт по
`(epk_id, report_dt)`, иначе по `epk_id` (таблица псевдо-разбиения `epk_id × treatment`).

## Тулкит-адаптеры (готово)

Реализованы в пакете `src/rnd_reports/embeddings/` как «переходники» данных:

1. **`EmbeddingReducer`** (`reducer.py`) — снижает размерность эмбеддингов до
   `red_size` (по умолчанию 5) методом **StandardScaler + PCA** (`pyspark.ml`;
   стандартизация перед PCA обязательна — он чувствителен к масштабу признаков). Выход:
   `[epk_id, report_dt, red_0, ..., red_{red_size-1}]`.
2. **`PropensityScorer`** (`propensity.py`) — джойнит эмбеддинги с таблицей трита и сводит
   их к единственному `prop_score = P(treatment=1)`. Из **LogisticRegression** и
   **GBTClassifier** берётся модель с лучшим ROC-AUC на отложенной выборке. Выход:
   `[epk_id, report_dt, prop_score]`. **ROC-AUC — лишь selection-эвристика выбора модели**,
   а не мера качества propensity для causal-задачи: высокий AUC означает почти
   детерминированный трит и тянет за собой плохой overlap и экстремальные веса. Финальный
   causal-критерий — баланс ковариат (`|SMD|`) и overlap/positivity **после** matching/IPW
   (см. causal-слой ниже).

Оба адаптера поддерживают **in-time safety**: при заданном `cutoff` обучение идёт только
на `report_dt <= cutoff`, а применение — к более поздним срезам (без утечки из будущего;
ср. подход R&D-6 «safe in-time CUPAC»).

> **Миграция API (контракт R&D-7).** Имена приведены к финальной схеме:
> вход `col_* → emb_{i}_val` (легаси `col_*` всё ещё распознаётся), выход reducer'а
> `emb_000 → red_0`, propensity `propensity_score → prop_score`, параметр
> `reducted_shape → red_size`. Старое имя `reducted_shape` сохранено как deprecated-alias
> (с `DeprecationWarning`), чтобы не ломать ранее написанные ноутбуки/скрипты.

Контракты схем и валидация — `contracts.py`; конфиг — `configs/07_embedding_adjustment_set/adapters.yaml`.

## Causal-слой (реализован)

`src/rnd_reports/embeddings/experiment.py` (numpy/sklearn, без pyspark — запускается в base-окружении):

- `estimate_ate_with_adjustment(method=…)` — ATE наблюдательных данных: `naive` (разность средних),
  `propensity_weighting` (стабилизированный IPW, propensity = `LogisticRegression` на эмбеддингах),
  `doubly_robust` (AIPW: per-arm регрессия исхода + IPW-коррекция);
- `covariate_balance_after_adjustment` — `|SMD|` ковариат до/после IPW-взвешивания;
- `overlap_diagnostics` — overlap/positivity по распределению propensity;
- `evaluate_adjustment_set_quality` — сводка: ATE тремя методами + смещения vs эталон.

Синтетика — `embeddings/synthetic.py` (`make_embedding_observational_scenario`): латентные
конфаундеры → эмбеддинги `col_*`; `treatment ~ Bernoulli(sigmoid(confounding·z·a))` (селекция,
не рандом); `outcome = true_ate·T + z·b + ε`. ATE известен по построению.

## Результаты

Синтетика n=4000, k=8 эмбеддингов → 5 PCA-компонент как adjustment set, истинный ATE = 3.0, seed=11.

| метод | ATE | смещение vs true | снижение |SMD|/смещения |
| --- | --- | --- | --- |
| naive (без поправки) | 2.145 | −0.855 | — |
| IPW (propensity на эмбеддингах) | 2.930 | −0.070 | **−91.8%** смещения |
| doubly_robust (AIPW) | 2.948 | −0.052 | **−94.0%** смещения |

Баланс ковариат: max `|SMD|` **0.869 → 0.019** после IPW-взвешивания. Overlap: **97.6%** propensity
в [0.1, 0.9] (positivity выполняется).

![ATE по методам и overlap propensity](../../results/07_embedding_adjustment_set/figures/rnd7_ate_and_overlap.png)

![Баланс ковариат до/после поправки](../../results/07_embedding_adjustment_set/figures/rnd7_balance.png)

## Выводы

- Наивная разность средних **смещена** из-за селекции (трит коррелирует с конфаундерами через эмбеддинги).
- Сокращённые эмбеддинги как **adjustment set** + IPW/doubly-robust убирают **~92–94% смещения** и
  восстанавливают баланс ковариат (max `|SMD|` → ~0.02) при хорошем overlap.
- Вывод: клиентские эмбеддинги несут достаточно информации о конфаундерах, чтобы служить
  практичным adjustment set в наблюдательном испытании — **при условии overlap/positivity**.
- Ограничения: результат на синтетике с известным ATE; на реальных данных нужны проверка overlap и
  sensitivity к неучтённым конфаундерам (unconfoundedness напрямую не тестируема). Production-путь —
  pyspark-адаптеры `EmbeddingReducer`/`PropensityScorer` с in-time safety (обучение на `report_dt <= cutoff`).

## Воспроизведение
```bash
pip install -e .            # base (numpy/pandas/scipy/sklearn) — эксперимент запускается без pyspark
pip install -e .[spark]     # опц.: pyspark для production-адаптеров и их тестов
python tools/execute_notebooks.py   # выполнить notebook.ipynb (→ results/07.../figures/)
python tools/generate_pdf.py        # пересобрать report.pdf
```
