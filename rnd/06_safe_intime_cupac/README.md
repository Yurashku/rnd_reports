# 06_safe_intime_cupac

R&D-трек: **Safe in-time covariates для CUPAC-style снижения дисперсии**.

Статус: заглушка — структура подготовлена, методология ещё не реализована.

## Содержимое папки (report-oriented формат)
- `notebook.ipynb` — воспроизводимые эксперименты на синтетике (пока заглушка);
- `report.md` — единый источник правды отчёта;
- `report.pdf` — автогенерация из `report.md` через `tools/generate_pdf.py`;
- `README.md` — этот файл.

## Общий код
Переиспользуемая логика трека живёт **не** в этой папке, а в общем пакете
`src/rnd_reports/` (подпакеты `variance_reduction`, `feature_safety`,
`synthetic`, `benchmark`). Папка RnD остаётся отчётно-ориентированной.

## Контекст
- `../../docs/06_safe_intime_cupac_context.md`
- `../../docs/feature_safety_policy.md`
- `../../docs/variance_reduction_methodology.md`
- Конфиги: `../../configs/06_safe_intime_cupac/`
