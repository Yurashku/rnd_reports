# RnD-8: Сравнение методов множественного тестирования (OR-регион)

Честное сравнение методов множественного тестирования для **OR-claim** A/B-пилота («есть ли в
семействе target-метрик хотя бы один надёжный сигнал и какие именно метрики его дали»).
Сравниваются **Bonferroni, Holm, Westfall–Young (maxT), Romano–Wolf (stepdown)** (контроль
FWER) и **Benjamini–Hochberg** (единственный FDR-ориентир).

- **Статус:** ✅ базовый эксперимент готов: single-table сравнение + Monte-Carlo operating
  characteristics (эмпирический FWER и power по методам vs корреляция метрик).
- **Отчёт:** [`report.md`](report.md) (PDF: `report.pdf`).
- **Код-витрина:** [`notebook.ipynb`](notebook.ipynb) — лаконичная: таблица + 3 графика.
- **Общий код:** `src/rnd_reports/multiple_testing/` (`elementary`, `methods`, `synthetic`,
  `pipeline`, `simulation`); логика — на готовых решениях scipy/statsmodels, своё — только
  Romano–Wolf stepdown. Тесты — `tests/test_rnd8_multiple_testing.py`.

## Своя таблица вместо синтетики

Контракт колонок `id, treatment, target_*`. В `notebook.ipynb` раскомментируй помеченный блок
во 2-й ячейке (`df = pd.read_csv(...)`, `true_effects = None`). При `true_effects=None`
single-table сравнение работает, а оценка power/FDR и Monte-Carlo пропускаются (на реальных
данных правда неизвестна).

## Воспроизведение
```bash
.venv/bin/python -m pytest -q tests/test_rnd8_multiple_testing.py   # тесты пакета
python tools/execute_notebooks.py   # выполнить notebook.ipynb (→ results/08_multiple_testing/figures/)
python tools/generate_pdf.py        # пересобрать report.pdf из report.md
```
