# Контекст R&D-6: Safe in-time CUPAC

- **Версия документа:** v0.2
- **Статус:** Draft / working context
- **Связанный R&D:** `06_safe_intime_cupac/`
- **Цель документа:** зафиксировать актуальную постановку R&D, классы признаков, цепочку методов, метрики эффективности, требования к данным и ожидаемый формат результатов.

## 1. Краткая формулировка

R&D-6 проверяет, можно ли безопасно повысить чувствительность A/B-тестов за счет последовательного добавления допустимых in-time признаков поверх локального scikit-learn-based CUPAC.

Основная цепочка улучшений:

```text
A/B -> Scikit CUPAC A -> Scikit CUPAC A+B -> Scikit CUPAC A+B+C
```

где:

```text
A = pre-treatment признаки;
B = экспертно очевидные safe in-time признаки;
C = неочевидные in-time признаки, прошедшие balance/missingness gate.
```

HypEx используется в двух местах:

1. `ab_hypex` — основной A/B baseline без снижения дисперсии;
2. `hypex_cupac` — reference/parity baseline для сверки с текущей библиотечной реализацией CUPAC.

Важно: `hypex_cupac` не входит в последовательную predecessor-chain. Он нужен для проверки сопоставимости с библиотечной реализацией, а основная цепочка прироста строится вокруг локального `sklearn_cupac`.

## 2. Почему нужен этот R&D

Текущий CUPAC использует признаки, доступные до воздействия / exposure / trigger. Это методологически безопасная зона, но ее предсказательной силы может быть недостаточно для максимального снижения дисперсии.

В компании могут существовать дополнительные признаки, доступные во время эксперимента или в период наблюдения. Часть из них может быть безопасной для линейного adjustment, если treatment на них не влияет или если они проходят практический safety-gate.

R&D должен ответить на вопрос:

> дает ли последовательное добавление safe in-time признаков дополнительное снижение дисперсии и сокращение требуемой выборки относительно A/B и Scikit CUPAC baseline?

## 3. Классы признаков

### A. Pre-treatment признаки

Признаки, доступные до assignment / exposure / trigger. Это текущая безопасная зона CUPAC.

Использование:

```text
A -> CUPAC / ML-CUPAC
```

Примеры:

```text
pre-period target;
исторические агрегаты;
предыдущая активность;
признаки, рассчитанные до пилота.
```

### B. Expert-safe in-time признаки

Признаки, которые могут быть доступны во время эксперимента, но экспертно очевидно, что treatment на них не влияет.

Основание допуска: продуктовая/экспертная невозможность влияния treatment на признак.

Использование:

```text
B -> linear second-stage adjustment поверх Scikit CUPAC A
```

Примеры:

```text
инфляция;
курс валют;
погода;
календарные признаки;
макроэкономический фон;
внешний рыночный контекст;
системный фон, если эксперимент не может его менять.
```

Balance/missingness checks для B все равно полезны как sanity-check: если внешний признак внезапно сильно различается между treatment/control, это может означать проблему с join, timestamp, trigger-логикой или сегментацией.

### C. Balance-gated in-time признаки

Неочевидные in-time признаки, которые не являются экспертно очевидными external/context features, но и не выглядят явно запрещенными.

Основание допуска: предварительный exclusion-filter + balance/missingness gate.

Минимальный gate:

```text
признак не является частью target/outcome;
признак не построен из outcome window;
признак не выглядит как очевидный медиатор;
признак не является post-treatment behavioral response;
признак не имеет treatment-dependent missingness;
признак сбалансирован между treatment/control;
желательно: баланс проверен не только по среднему, но и по missingness, распределению и ключевым сегментам.
```

Использование:

```text
C -> linear second-stage adjustment поверх Scikit CUPAC A+B
```

Важно: balance-test сам по себе не доказывает причинную безопасность. В этом R&D он используется как практический gate только после предварительного исключения очевидных mediator/leakage/outcome-derived признаков.

### D. DAG-required признаки

Потенциально полезные признаки, для которых balance-gate недостаточен или есть сомнение в причинной роли.

Использование на текущем этапе:

```text
D -> не входит в основной estimator;
D -> сохраняется как future DAG/causal-validation candidate.
```

Для D в будущем можно использовать DAG, CAIMAN, DoWhy, локальные causal certificates или экспертную causal-разметку.

### E. Mediator-risk признаки

Признаки, которые могут лежать на пути:

```text
Treatment -> Feature -> Outcome
```

Использование:

```text
E -> запрещены для основного ATE-adjustment.
```

Они могут быть полезны для анализа механизма эффекта, но не для снижения дисперсии основной оценки ATE.

### F. Leakage / outcome-derived признаки

Жестко запрещенные признаки:

```text
части target;
агрегаты из outcome window;
future information;
постфактум-статусы;
признаки, рассчитанные после исхода;
признаки, доступность которых зависит от treatment/exposure/trigger.
```

Использование:

```text
F -> forbidden.
```

## 4. Unsafe-demo группа

В benchmark можно добавить одну unsafe-группу признаков для демонстрации риска.

Эта группа не является кандидатом к использованию. Ее задача — показать, что запрещенные или сомнительные признаки могут искусственно давать сильное снижение дисперсии, но при этом быть методологически некорректными.

Пример метода:

```text
sklearn_cupac_A_plus_B_plus_C_plus_unsafe_demo
```

В отчетах и таблицах такой метод должен быть явно помечен как:

```text
safety_status = forbidden_demo / unsafe_demo
```

## 5. Цепочка методов benchmark

Обязательные методы:

```text
ab_hypex
hypex_cupac
sklearn_cupac_A
sklearn_cupac_A_plus_B_linear
sklearn_cupac_A_plus_B_plus_C_linear
unsafe_demo_optional
```

Назначение методов:

```text
ab_hypex                         -> A/B baseline без variance reduction;
hypex_cupac                       -> reference/parity baseline из HypEx;
sklearn_cupac_A                   -> основной local/scikit CUPAC baseline;
sklearn_cupac_A_plus_B_linear     -> CUPAC A + expert-safe in-time признаки;
sklearn_cupac_A_plus_B_plus_C_linear -> CUPAC A+B + balance-gated in-time признаки;
unsafe_demo_optional              -> демонстрация риска, не кандидат к использованию.
```

Predecessor-chain:

```text
ab_hypex -> sklearn_cupac_A -> sklearn_cupac_A_plus_B_linear -> sklearn_cupac_A_plus_B_plus_C_linear
```

Исключение:

```text
hypex_cupac -> reference_only, не входит в predecessor-chain.
```

## 6. Как считаем эффективность

Для каждого метода нужно считать три вида эффективности.

### 6.1. Абсолютный выигрыш относительно A/B

Показывает, насколько метод лучше базового A/B без снижения дисперсии.

```text
variance_reduction_vs_ab_%
sample_size_reduction_vs_ab_%
```

### 6.2. Абсолютный выигрыш относительно Scikit CUPAC

Показывает, насколько новый метод лучше основного local/scikit CUPAC baseline.

```text
variance_reduction_vs_sklearn_cupac_%
sample_size_reduction_vs_sklearn_cupac_%
```

Для самого `sklearn_cupac_A` эти значения равны нулю или помечаются как baseline.

### 6.3. Инкрементальный выигрыш относительно предыдущего метода

Показывает вклад конкретной новой итерации признаков.

```text
incremental_variance_reduction_vs_predecessor_%
incremental_sample_size_reduction_vs_predecessor_%
```

Пример:

```text
sklearn_cupac_A_plus_B_linear
сравнивается с sklearn_cupac_A

sklearn_cupac_A_plus_B_plus_C_linear
сравнивается с sklearn_cupac_A_plus_B_linear
```

Так мы видим не только общий выигрыш, но и вклад каждой следующей группы признаков.

## 7. Формулы

Для снижения дисперсии относительно baseline:

```text
variance_reduction_vs_baseline_% =
100 * (1 - variance_method / variance_baseline)
```

Для сокращения требуемой выборки:

```text
sample_size_reduction_vs_baseline_% =
100 * (1 - variance_method / variance_baseline)
```

В первом приближении эти величины совпадают, потому что при фиксированных alpha, power и MDE необходимая выборка пропорциональна дисперсии метрики.

Для инкрементального выигрыша:

```text
incremental_variance_reduction_vs_predecessor_% =
100 * (1 - variance_method / variance_predecessor)

incremental_sample_size_reduction_vs_predecessor_% =
100 * (1 - variance_method / variance_predecessor)
```

## 8. Формат данных

Код должен принимать данные в формате, близком к будущей интеграции в HypEx.

Минимальный формат:

```text
id
treatment
target
feature columns
feature registry / feature groups
```

Ожидаемый объект-обертка:

```python
BenchmarkDataset(
    data=df,
    id_col="id",
    treatment_col="treatment",
    target_col="target",
    feature_registry=registry,
)
```

Пример registry:

```python
{
    "x_pre_1": "A_pre_treatment",
    "x_inflation": "B_expert_safe_intime",
    "x_context_1": "C_balance_gated_intime",
    "x_session_action": "D_dag_required",
    "x_click_after_treatment": "E_mediator_risk",
    "x_future_target_sum": "F_leakage"
}
```

Важно: R&D-код не должен быть полностью игрушечным. Если метод докажет эффективность, его должно быть удобно перенести в HypEx без переписывания всей логики.

## 9. Данные для проверки

R&D должен тестироваться на двух типах данных.

### 9.1. Синтетические данные

Синтетика обязательна, потому что только в ней мы заранее знаем, какие признаки относятся к A, B, C, D, E, F и unsafe-demo.

На синтетике нужно проверить:

```text
стабильность A/B baseline;
сопоставимость HypEx CUPAC и Scikit CUPAC;
выигрыш от B;
выигрыш от C;
поведение unsafe-demo;
отсутствие попадания E/F в основной estimator.
```

### 9.2. Несколько реальных open-source датасетов

Нужно проверить подход на нескольких реальных open-source датасетах, которые можно скачать локально и привести к формату benchmark.

Первичный список кандидатов:

```text
Criteo Uplift Prediction Dataset;
Hillstrom Email Marketing Dataset;
Lenta Uplift Modeling Dataset;
X5 RetailHero Uplift Modeling Dataset;
MegaFon Uplift Competition Dataset.
```

Criteo особенно полезен как крупный uplift benchmark: исходная работа описывает публичную коллекцию из 13.9 млн наблюдений, собранных из randomized control trials: https://arxiv.org/abs/2111.10106

Для Criteo Uplift v2.1 также есть свежий benchmark, где указано 13.98 млн записей и near-random assignment: https://arxiv.org/abs/2604.06123

Другие датасеты нужно дополнительно проверить на доступность скачивания, лицензию, формат, наличие treatment/control, target и пригодность для классификации признаков.

Если датасет не является настоящим randomized experiment, это нужно явно указать в отчете. Такой датасет можно использовать как demonstration или semi-synthetic benchmark, но не как полноценную проверку causal-корректности.

## 10. Ожидаемая таблица результатов

Benchmark должен возвращать единую таблицу:

```text
hypothesis_name
method
predecessor_method
dataset_type
dataset_name
target
n
ate
se
p_value
ci_low
ci_high
adjusted_target_variance

variance_reduction_vs_ab_%
sample_size_reduction_vs_ab_%

variance_reduction_vs_sklearn_cupac_%
sample_size_reduction_vs_sklearn_cupac_%

incremental_variance_reduction_vs_predecessor_%
incremental_sample_size_reduction_vs_predecessor_%

feature_groups_used
n_features_used
safety_status
diagnostic_notes
```

Для `hypex_cupac` поле `predecessor_method` можно оставить пустым или указать `reference_only`.

## 11. Визуализация

Обязательный график:

```text
ATE + confidence interval по всем методам на одном графике
```

На графике должны быть:

```text
ab_hypex
hypex_cupac
sklearn_cupac_A
sklearn_cupac_A_plus_B_linear
sklearn_cupac_A_plus_B_plus_C_linear
unsafe_demo_optional
```

Цель графика:

```text
показать, сужается ли CI;
проверить, не сдвигается ли подозрительно ATE;
увидеть вклад B;
увидеть вклад C;
показать, почему unsafe-demo не должен быть кандидатом к использованию.
```

## 12. Критерии успеха R&D

R&D считается успешным, если:

1. есть воспроизводимый benchmark на синтетике;
2. есть benchmark на нескольких open-source датасетах или явный каталог датасетов-кандидатов с loader/adapters;
3. `ab_hypex` используется как A/B baseline;
4. `hypex_cupac` используется как reference/parity baseline;
5. `sklearn_cupac_A` используется как основной CUPAC baseline для цепочки улучшений;
6. `sklearn_cupac_A_plus_B_linear` показывает вклад expert-safe in-time признаков;
7. `sklearn_cupac_A_plus_B_plus_C_linear` показывает вклад balance-gated признаков;
8. для каждого метода считаются абсолютные и инкрементальные выигрыши по дисперсии и требуемой выборке;
9. risky/DAG-required признаки не попадают в основной estimator;
10. E/F признаки запрещены;
11. unsafe-demo явно помечен как demonstration-only;
12. есть график ATE + CI по всем методам;
13. по итогам можно принять решение, стоит ли переносить подход в HypEx.

## 13. Что не входит в первый этап

На первом этапе не нужно:

```text
строить полный DAG;
использовать CAIMAN как обязательный компонент;
реализовывать causal discovery;
автоматически разрешать D-признаки;
переносить код в HypEx;
делать production-ready API;
использовать E/F признаки в основном estimator;
смешивать все in-time признаки в одну ML-модель без safety-классов.
```

## 14. Итоговая формулировка

R&D-6 проверяет, можно ли безопасно повысить чувствительность A/B-тестов за счет последовательного добавления safe in-time признаков поверх local/scikit CUPAC.

Основная цепочка эффективности:

```text
A/B -> Scikit CUPAC A -> Scikit CUPAC A+B -> Scikit CUPAC A+B+C
```

Для каждого следующего метода считаем:

```text
1. абсолютный выигрыш относительно A/B;
2. абсолютный выигрыш относительно Scikit CUPAC;
3. инкрементальный выигрыш относительно предыдущего метода.
```

HypEx CUPAC используется как reference/parity baseline, но не входит в последовательную цепочку улучшений. Unsafe-группа может быть добавлена только для демонстрации риска и не является кандидатом к использованию.
