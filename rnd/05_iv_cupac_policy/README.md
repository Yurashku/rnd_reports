# RnD-5: IV/CUPAC/policy-анализ пилота с неполным акцептом

Единый сценарий с one-sided noncompliance (`Z` назначение, `D` акцепт, `Y` метрика): baseline A/B
по `Z`, оценки LATE (Wald и 2SLS с bootstrap CI), CUPAC-residualization целевой метрики и
policy-based сравнение политик `random` vs `targeted` через OPE (IPS/SNIPS/DR).

- **Статус:** ⏸ paused.
- **Отчёт:** [`report.md`](report.md) (PDF: `report.pdf`).
- **Код-витрина:** [`notebook.ipynb`](notebook.ipynb). Исторические демо — в `archive/` (`rnd5_late_iv_offpolicy_demo*.ipynb`).
- **Внешние зависимости:** `linearmodels`, `hypex`, `OffPolicyLab` (см. корневой `README.md`).

## Воспроизведение
```bash
python tools/generate_pdf.py        # пересобрать report.pdf из report.md
python tools/execute_notebooks.py   # выполнить notebook.ipynb
```
