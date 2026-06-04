# RnD-4: Trade-off скорость/точность в HypEx matcher (faiss)

Выбор индекса nearest-neighbor для matching-задач с балансом latency и recall: exact numpy
baseline vs approximate (faiss при наличии; иначе graceful fallback на random-projection
shortlist). Оцениваются recall@k относительно exact-эталона и latency на пакет запросов.

- **Статус:** 📄 report-only (основная разработка ведётся вне этого репозитория).
- **Отчёт:** [`report.md`](report.md) (PDF: `report.pdf`).
- **Код-витрина:** [`notebook.ipynb`](notebook.ipynb).

## Воспроизведение
```bash
python tools/generate_pdf.py        # пересобрать report.pdf из report.md
python tools/execute_notebooks.py   # выполнить notebook.ipynb
```
