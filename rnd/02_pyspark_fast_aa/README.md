# RnD-2: PySpark как альтернатива быстрому AA-тесту из HypEx

Как организовать вычисление AA-теста на spark-подобном бэкенде, чтобы сократить время без потери
корректности метрик. Сравниваются наивная схема (проход на каждую метрику), batch-агрегация
(один проход) и incremental combine (партиционные summary + merge).

- **Статус:** ✅ done.
- **Отчёт:** [`report.md`](report.md) (PDF: `report.pdf`).
- **Код-витрина:** [`notebook.ipynb`](notebook.ipynb) — синтетический spark-like пайплайн.

## Воспроизведение
```bash
python tools/generate_pdf.py        # пересобрать report.pdf из report.md
python tools/execute_notebooks.py   # выполнить notebook.ipynb
```
