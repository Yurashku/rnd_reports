# RnD-6: Safe in-time covariates для CUPAC-style снижения дисперсии

> **Статус:** в работе. Реализованы синтетический бенчмарк и цепочка методов
> `A/B → CUPAC A → A+B → A+B+C`; подключены адаптеры реальных open-source датасетов и
> проведён расширенный скаутинг с delta-эффектами на A-only (раздел «Expanded public dataset
> validation»). Источник правды по B/C-числам — `notebook.ipynb` (синтетика, seed=11).

## Цель
Проверить, даёт ли **последовательное добавление безопасных in-time признаков** поверх
scikit-CUPAC дополнительное снижение дисперсии (и сокращение требуемой выборки) без
нарушения методологической корректности оценки ATE.

## Классы признаков (safety layer)
- **A** pre-treatment — идут в CUPAC;
- **B** expert-safe in-time — линейная second-stage коррекция;
- **C** balance-gated in-time — линейная коррекция после balance/missingness gate;
- **D** dag-required — вне estimator (нужна causal-проверка);
- **E** mediator-risk, **F** leakage — запрещены для основной оценки ATE;
- **unsafe_demo** — только демонстрация риска, не кандидат.

## Методы
`ab_hypex` (A/B baseline) · `hypex_cupac` (reference/parity, вне цепочки) ·
`sklearn_cupac_A` → `sklearn_cupac_A_plus_B_linear` → `sklearn_cupac_A_plus_B_plus_C_linear`
(основная цепочка прироста) · `unsafe_demo_optional` (демонстрация).

## Результаты на синтетике (n=8000, seed=11, истинный ATE = 2.8)

| method | ate | p_value | variance_reduction_vs_ab_% | incremental_vs_predecessor_% | safety_status |
| --- | --- | --- | --- | --- | --- |
| ab_hypex | 2.758 | 0.0 | 0.0 | — | ok |
| hypex_cupac | 2.785 | 0.0 | 13.56 | — | reference_only |
| sklearn_cupac_A | 2.801 | 0.0 | 25.63 | 25.63 | ok |
| sklearn_cupac_A_plus_B_linear | 2.775 | 0.0 | 42.38 | 22.52 | ok |
| sklearn_cupac_A_plus_B_plus_C_linear | 2.766 | 0.0 | 52.80 | 18.07 | ok |
| unsafe_demo_optional | 1.000 | 0.0 | 83.17 | — | unsafe_demo |

(Снижение требуемой выборки в первом приближении совпадает со снижением дисперсии.)

## Ключевые выводы
1. **Каждая безопасная группа даёт дополнительный выигрыш**: A (~26%) → +B (~42%) → +B+C (~53%)
   снижения дисперсии относительно A/B, при этом ATE остаётся несмещённым (≈ 2.8).
2. **CUPAC parity**: `hypex_cupac` (~14%) сопоставим по логике с `sklearn_cupac_A`;
   локальная реализация удобнее для R&D и используется как основная.
3. **Balance/missingness gate** пропускает A/B/C (|SMD|≈0) и отклоняет D/E/F и unsafe_demo
   (|SMD| ≈ 0.45–1.14, p≈0) — наглядно, почему их нельзя брать в estimator.
4. **unsafe_demo** даёт максимальное «снижение дисперсии» (~83%), но **смещает оценку ATE
   далеко от истинного 2.8 — примерно к 1.0** (см. таблицу) — демонстрация ловушки
   outcome-derived признаков; в кандидатные методы не входит.

## Безопасность и ограничения
- balance-test — практический gate (после исключения очевидных E/F), а **не** доказательство
  причинной безопасности; для D-признаков нужен будущий DAG/causal-этап.
- E/F запрещены; safe-intime используется **только** в линейном second-stage, не в произвольной ML-модели.

## Статус валидации на реальных данных
- **Синтетика** подтверждает полную цепочку **A+B+C** (таблица выше): только там классы
  A/B/C/D/E/F известны по построению.
- **Публичные реальные датасеты сейчас поддерживают в основном A-only** CUPAC-валидацию.
  Сырой аудит (download + `schema_summary`) покрыл Orange Belgium, Lenta, Criteo, X5 RetailHero,
  MegaFon, Lazada/DESCN, CUVET-policy, MT-LIFT, Dunnhumby и semi-synthetic causal-бенчмарки.
- **Hillstrom** проверяет pipeline/A-only, **не** B/C.
- Реальные safe in-time **B/C** отсутствуют: признаки либо анонимны (`f*`, `PC*`, `X_*`), либо
  это пре-кампейн агрегаты (A), либо пост-treatment поведение (E/F). Датасеты с настоящими
  таймстемпами либо скрывают дату кампании (X5), либо не рандомизированы (Dunnhumby).
- **Полная A+B+C-валидация на публичных реальных датасетах не подтверждена** и остаётся
  неподтверждённой, пока сырой аудит не найдёт защитимых B/C-колонок с timing-evidence.
- Детали и пятичастная классификация датасетов: **`docs/06_real_dataset_raw_audit.md`**.

## Expanded public dataset validation
Расширенный скаутинг (поиск шире классических uplift-наборов + реальная загрузка/инспекция +
прототипы адаптеров + фактические delta-эффект прогоны). Полный отчёт и таблица
представительности: **`docs/06_expanded_dataset_scouting.md`**.

**Представительность классов признаков (по факту схем, не по документации):**
- **A** (pre-treatment) — **9+** реальных кандидатов имеют A-признаки по схеме (+ синтетика), но
  **фактический delta-эффект прогнан и подтверждён на 5 из них** (hillstrom, orange_belgium,
  lazada_descn, lenta, x5 — таблица ниже). Остальные A/sandbox-кандидаты остаются на уровне
  схемы/документации (criteo — access_blocked; open_bandit/dunnhumby/criteo_private_ad — sandbox,
  не прогнаны как реальная A-валидация).
- **B** (expert-safe in-time) — **0**: ни один публичный набор не выкладывает именованные,
  семантически интерпретируемые in-time экспертно-безопасные колонки.
- **C-кандидаты** (реальные, защитимые) — **0**; sandbox-конструкция C теоретически возможна
  на **2** event-log датасетах (Open Bandit, dunnhumby), но это **не** реальная валидация.
- **D-кандидаты** — **5** (Criteo `exposure`; Lazada biased-train propensity; Open Bandit
  `action`/`position`/`pscore`; CriteoPrivateAd display/privacy-механика; dunnhumby campaign/coupon).
- **E/F/unsafe-кандидаты** — **7+** (Hillstrom `visit`/`conversion`; Criteo `exposure`;
  X5 пост-коммуникационные покупки; Open Bandit reward-производные; CriteoPrivateAd delayed
  click/sale; dunnhumby redemption и др.).

**Только-A валидация (real RCT/uplift):** Hillstrom, Lenta, X5 RetailHero, Orange Belgium,
Lazada/DESCN (рандомизированный test), Criteo (слабый A-only).
**C/D-«песочницы» (не A/B):** Open Bandit (logged bandit), dunnhumby Complete Journey
(observational), CriteoPrivateAd (advertising logs) — полезны для *исследования* C/D-механики,
не валидируют causal-корректность safe in-time.

> **Полная A+B+C-валидация на публичных реальных датасетах не подтверждена. Однако
> C/D-представительность можно исследовать на event-log / bandit / advertising / journey
> datasets, которые не являются прямыми A/B uplift benchmark.**

**Delta-эффекты на реальных данных (A-only цепочка `ab_hypex → sklearn_cupac_A`, seed=11).**
Прогоны и фигуры собираются `python tools/audit_datasets.py --delta`; компактные результаты —
`expanded_dataset_delta_results.csv`, графики — `figures/`.

| dataset | validation_level | n | ATE (A/B) | var.reduction CUPAC-A vs A/B, % | A-признаков | safety |
| --- | --- | --- | --- | --- | --- | --- |
| hillstrom | A_only_real | 64 000 | 0.597 | 0.06 | 18 | ok |
| orange_belgium | A_only_real | 11 896 | −0.0028 | 6.06 | 336 | ok |
| lazada_descn | A_only_real | 181 669 | 0.0037 | 5.41 | 83 | ok |
| lenta | A_only_real | 687 029 | 0.0075 | 14.36 | 196 | ok |
| x5_retailhero | A_only_real | 200 039 | 0.0332 | 16.10 | 10 | ok |

- **CUPAC-A даёт реальное снижение дисперсии 0–16%** на публичных данных; сильнее всего там,
  где есть содержательные pre-treatment признаки — **lenta** (именованные пре-кампейн агрегаты,
  ~14%) и **X5** (engineered история покупок, ~16%). ATE остаётся стабильным (несмещённым).
- **hillstrom** — pipeline/A-only smoke: целевой `spend` крайне шумный, A-вклад почти нулевой (~0.06%).
- **Цепочки `+B` и `+B+C` на реальных данных не запускались**: реальных B/C-колонок нет, поэтому
  относительные delta-эффекты B/C по-прежнему подтверждены **только на синтетике** (таблица выше).
- **Не вошли в delta-таблицу:** `criteo` — источник отдаёт HTTP 403 (access_blocked);
  `open_bandit` — пакет `obp` несовместим с pandas 2.x (`DataFrame.drop("col", 1)`),
  runtime-blocked sandbox; `dunnhumby` / `criteo_private_ad` — адаптеры реализованы
  (`research_sandbox`/`event_log_sandbox`, demo-only) и строят анкер-based ResearchDataset из
  локальных данных, но это observational/ad-логи (не RCT) → в реальную A-only delta-таблицу не
  входят и как safe B/C не используются.

## Рекомендация по переносу в HypEx
Подход показывает устойчивый прирост на синтетике и сопоставим с HypEx CUPAC; на реальных
публичных данных подтверждён **A-only** вклад CUPAC-A (0–16% снижения дисперсии, 5 датасетов).
Перед переносом в HypEx необходимо:
1. **Подтвердить вклад B/C на внутренних корпоративных данных, а не на публичных.** Расширенный
   скаутинг показал, что публичные датасеты **не валидируют B/C** (реальных именованных safe
   in-time колонок с timing-evidence нет; см. «Expanded public dataset validation»). Поэтому
   B/C-валидацию следует переносить на внутренние данные (VarWar-класса), где timing и семантика
   признаков известны и защитимы.
2. Формализовать balance/missingness gate как переиспользуемый компонент.
3. Зафиксировать запрет E/F и отдельный статус unsafe_demo.

Предыдущее свидетельство эффективности CUPAC на реальных данных — VarWar (снижение дисперсии
~34% и ~33%; данные приватны, не коммитятся) — это и есть пример внутреннего источника, на котором
можно проверять B/C.

## Воспроизведение
```bash
pip install -e .[ml]
# см. notebook.ipynb; график ATE±CI и таблица собираются из rnd_reports.benchmark
# delta-эффекты на реальных датасетах (читают gitignored data/, raw не коммитятся):
python tools/audit_datasets.py --delta   # → expanded_dataset_delta_results.csv + figures/
python tools/generate_pdf.py             # пересобрать report.pdf
```
