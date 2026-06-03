# Методология снижения дисперсии (CUPED / CUPAC / safe in-time)

- **Версия документа:** v0.1
- **Статус:** Draft (заглушка)
- **Связанный код:** `src/rnd_reports/variance_reduction/`

## Обзор подходов (черновик)
- **CUPED** — коррекция метрики по пред-периодной ковариате (`cuped.py`).
- **CUPAC** — коррекция по предсказанию модели на ковариатах; library-first через HypEx (`hypex_cupac_adapter.py`), локальный вариант — `local_cupac.py`.
- **Safe in-time adjustment** — коррекция только «безопасными» in-time ковариатами (`safe_intime_adjustment.py`), совместимыми с [политикой безопасности](feature_safety_policy.md).

## Метрики оценки (черновик)
- Процент снижения дисперсии относительно baseline.
- Несмещённость оценки эффекта (bias) на синтетике с известным эффектом.
- Покрытие доверительных интервалов.
Реализация — `metrics.py` (заглушка).

## Протокол сравнения
- Сценарии: `configs/06_safe_intime_cupac/synthetic_scenarios.yaml`.
- Наборы ковариат: `configs/06_safe_intime_cupac/feature_sets.yaml`.
- Методы: `configs/06_safe_intime_cupac/methods.yaml`.
- Прогон и отчётность: `src/rnd_reports/benchmark/` (`protocol.py`, `runners.py`, `reporting.py`).

## TODO
- [ ] Описать формальные оценки и их доверительные интервалы.
- [ ] Зафиксировать baseline и набор сравниваемых методов.
- [ ] Определить критерии «метод безопасен и полезен».
