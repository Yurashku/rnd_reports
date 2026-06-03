# RnD-6: Safe in-time covariates для CUPAC-style снижения дисперсии

> **Статус:** в работе. Реализованы синтетический бенчмарк и цепочка методов
> `A/B → CUPAC A → A+B → A+B+C`; адаптеры реальных open-source датасетов — следующий шаг.
> Источник правды по числам — `notebook.ipynb` (синтетика, seed=11).

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
4. **unsafe_demo** даёт максимальное «снижение дисперсии» (~83%), но **смещает ATE к 0** —
   демонстрация ловушки outcome-derived признаков; в кандидатные методы не входит.

## Безопасность и ограничения
- balance-test — практический gate (после исключения очевидных E/F), а **не** доказательство
  причинной безопасности; для D-признаков нужен будущий DAG/causal-этап.
- E/F запрещены; safe-intime используется **только** в линейном second-stage, не в произвольной ML-модели.

## Рекомендация по переносу в HypEx
Подход показывает устойчивый прирост на синтетике и сопоставим с HypEx CUPAC. Перед переносом
в HypEx необходимо: (1) подтвердить вклад B/C на нескольких реальных open-source датасетах;
(2) формализовать balance/missingness gate как переиспользуемый компонент; (3) зафиксировать
запрет E/F и отдельный статус unsafe_demo. Предыдущее свидетельство эффективности CUPAC на
реальных данных — VarWar (снижение дисперсии ~34% и ~33%; данные приватны, не коммитятся).

## Воспроизведение
```bash
pip install -e .[ml]
# см. notebook.ipynb; график ATE±CI и таблица собираются из rnd_reports.benchmark
python tools/generate_pdf.py   # пересобрать report.pdf
```
