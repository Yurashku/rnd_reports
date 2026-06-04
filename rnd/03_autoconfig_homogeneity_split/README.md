# RnD-3: Data-driven автоконфиг homogeneity test и AB split

Автоматический подбор параметров контроля баланса при рерандомизации: порог принятия из целевого
acceptance-rate, бюджет итераций `max_iter` из требуемой вероятности успеха, приоритизация
top-ковариат. Показано, почему фиксированный жёсткий threshold плохо переносится между сценариями
N/K/correlation.

- **Статус:** ✅ done.
- **Отчёт:** [`report.md`](report.md) (PDF: `report.pdf`).
- **Код-витрина:** [`notebook.ipynb`](notebook.ipynb) (полный исторический демо-ноутбук — в `archive/`).

## Воспроизведение
```bash
python tools/generate_pdf.py        # пересобрать report.pdf из report.md
python tools/execute_notebooks.py   # выполнить notebook.ipynb
```
