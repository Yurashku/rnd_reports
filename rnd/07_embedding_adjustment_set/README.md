# RnD-7: Эмбеддинги как adjustment set в нерандомизированном испытании

Могут ли клиентские эмбеддинги (в сокращённом виде) выступать как хотя бы сколько-то хороший
**adjustment set** в НЕрандомизированном испытании — обеспечивать контроль смещения отбора при
оценке причинного эффекта без рандомизации. Исходные данные — pyspark-эмбеддинги
(`epk_id, report_dt, emb_{i}_val`; легаси `col_*` тоже распознаётся) + отдельная таблица трита.

- **Статус:** 🔶 базовый эксперимент готов: тулкит-адаптеры (pyspark) + causal-слой (numpy/sklearn) — оценка ATE с поправкой на эмбеддинги, диагностика баланса/overlap (синтетика, известный ATE).
- **Отчёт:** [`report.md`](report.md) (PDF: `report.pdf`).
- **Код-витрина:** [`notebook.ipynb`](notebook.ipynb).
- **Общий код:** `src/rnd_reports/embeddings/` (`reducer.py`, `propensity.py`, `experiment.py`); конфиги — `configs/07_embedding_adjustment_set/`.
- **Зависимости:** pyspark — опциональный extra (`pip install -e .[spark]`); импорт пакета его не требует.

## Воспроизведение
```bash
python tools/generate_pdf.py        # пересобрать report.pdf из report.md
python tools/execute_notebooks.py   # выполнить notebook.ipynb
```
