# R&D-6: локальные open-source датасеты (setup)

- **Версия:** v1.0
- **Связано:** `docs/06_safe_intime_cupac_implementation_plan.md` (Step 8),
  `src/rnd_reports/datasets/`

R&D-6 проверяется на синтетике (всегда доступна) и на реальных open-source uplift-датасетах.
**Реальные данные в git не коммитятся** — их нужно скачать вручную в gitignored-директорию.

## Куда класть данные
```
data/06_safe_intime_cupac/<dataset>/<file>
```
Директория `/data/` указана в `.gitignore`. Путь по умолчанию — `loaders.DEFAULT_DATA_DIR`.

## Поток подключения
```
catalog.DatasetSpec (метаданные)  →  loaders.LocalFileLoader (читает локальный файл)
                                  →  adapters.*Adapter (разметка A–F)  →  BenchmarkDataset
```
Удобная обёртка:
```python
from rnd_reports.datasets.adapters import load_benchmark_dataset
bds = load_benchmark_dataset("hillstrom")          # путь по умолчанию
bds = load_benchmark_dataset("hillstrom", path="/abs/path/file.csv")
```
Если файла нет — `loader.load(...)` бросает `FileNotFoundError` с подсказкой из каталога
(`download_hint`, `source_url`, лицензия).

## Hillstrom (MineThatData E-Mail) — реализован
- Источник: <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>
- Положить CSV в `data/06_safe_intime_cupac/hillstrom/` (имя файла — см. `loaders.LOADERS["hillstrom"]`).
- Рандомизированный (1/3 Mens / 1/3 Womens / 1/3 No E-Mail).
- Разметка адаптера: treatment = `segment != "No E-Mail"`; target = `spend`;
  класс **A** — клиентские признаки до рассылки (`recency, history, mens, womens, newbie` +
  one-hot `history_segment, zip_code, channel`); `visit`/`conversion` — пост-treatment → класс **E**.
- У датасета нет in-time ковариат (B/C), поэтому здесь работает в первую очередь CUPAC по A.

## Остальные кандидаты (заглушки до своих шагов)
`lenta`, `criteo`, `x5_retailhero`, `megafon` — метаданные в `catalog.CANDIDATE_DATASETS`,
адаптеры пока бросают `NotImplementedError`. Каждый добавляется отдельно: loader (чтение файла) +
adapter (предобработка и ручная разметка A–F).

## Важно про корректность
Если датасет **не** является настоящим рандомизированным экспериментом, это нужно явно отметить в
отчёте (`is_randomized` в каталоге = `None`/`False`): такой датасет годится как demonstration/
semi-synthetic, но не как полноценная проверка causal-корректности.
