# Миграция кода из VarWar

- **Версия документа:** v0.1
- **Статус:** актуально
- **Источник:** соседний репозиторий `VarWar` (CUPAC Variance Reduction Experiments)

Документ фиксирует, что и куда перенесено из `VarWar` в `rnd_reports`, и какие
материалы сознательно **не** мигрированы.

## Карта переноса

| VarWar | rnd_reports | Примечание |
|--------|-------------|------------|
| `data_gen.py::DataGenerator` | `src/rnd_reports/synthetic/generators.py` | Без изменения логики; добавлены типы и докстринги. Колонки сохранены: `X1,X1_lag,X2,X2_lag,y0,y0_lag_1,y0_lag_2,z,U,D,d,y1,y`. |
| `autocupac.py::CUPACTransformer` | `src/rnd_reports/variance_reduction/local_cupac.py` | Локальный CUPAC. Публичный API сохранён (`fit/transform/fit_transform/get_report/get_feature_mapping`). |
| θ-резидуализация (внутри `autocupac.py`) | `src/rnd_reports/variance_reduction/cuped.py` | Вынесена чистая CUPED-математика (`cuped_theta`, `cuped_adjust`). |
| расчёт `var_reduction` (внутри `autocupac.py`) | `src/rnd_reports/variance_reduction/metrics.py` | `variance_reduction_pct`. |
| теория CUPED/CUPAC (README VarWar) | `docs/variance_reduction_methodology.md` | Формулы и обоснование. |

## Отличия от оригинала (намеренные)
- Общая CUPED-математика вынесена из `CUPACTransformer` в модули `cuped`/`metrics`
  и переиспользуется (без изменения численного результата на нормальных данных).
- **CatBoost — опциональная зависимость.** Если пакет не установлен, модель
  исключается из набора по умолчанию, а пайплайн не падает. Установка:
  `pip install -e .[ml]` или `pip install catboost`.
- `cuped_adjust` защищён от деления на ~0 (вырожденное предсказание → θ=0);
  в оригинале `transform` такого гард-кейса не имел. На нормальных данных
  результат идентичен.

## Что НЕ мигрировано (политика данных)
- `RealData1_CUPAC.ipynb`, `RealData2_CUPAC.ipynb` — содержат **реальные данные**
  и **остаются приватными в VarWar**. В `rnd_reports` реальные/корпоративные
  данные не коммитятся.
- Их результаты можно **цитировать** как предшествующее свидетельство
  эффективности подхода: снижение дисперсии ~34% (набор 1) и ~33% (набор 2).
- `runner.ipynb`, `R&D synthetic data.ipynb` — демонстрационные ноутбуки на
  синтетике; служат основой для тонкой витрины `rnd/06_safe_intime_cupac/notebook.ipynb`,
  но переносятся не дословно, а по мере наполнения RnD-6.
