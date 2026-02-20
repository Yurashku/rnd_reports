# RnD 01: Дизайн A/B-экспериментов

## Точки входа

1. **Краткий отчёт RnD:** [report.md](./report.md)
2. **Детальные материалы по секциям:** см. таблицу ниже (в каждой секции есть `report.md` + `analysis.ipynb`).

## Карта секций

| Секция | Что смотреть в первую очередь | Код / расчёты |
|---|---|---|
| `01_статьи_и_математика` | [`report.md`](./01_статьи_и_математика/report.md) | [`analysis.ipynb`](./01_статьи_и_математика/analysis.ipynb) |
| `02_множественные_проверки` | [`report.md`](./02_множественные_проверки/report.md) | [`analysis.ipynb`](./02_множественные_проверки/analysis.ipynb) |
| `03_рерандомизация` | [`report.md`](./03_рерандомизация/report.md) | [`analysis.ipynb`](./03_рерандомизация/analysis.ipynb), [`simulation.py`](./03_рерандомизация/simulation.py) |
| `04_план_внедрения` | [`report.md`](./04_план_внедрения/report.md) | [`analysis.ipynb`](./04_план_внедрения/analysis.ipynb) |
| `codex_prompts` | [`report.md`](./codex_prompts/report.md) | [`prompts_codex.md`](./codex_prompts/prompts_codex.md) |

## Как читать этот RnD

- Если нужно быстро понять результат: откройте `report.md` в корне RnD.
- Если нужно проверить расчёты: переходите в нужную секцию и открывайте `analysis.ipynb`.
- Если нужна воспроизводимая симуляция: используйте `03_рерандомизация/simulation.py`.
