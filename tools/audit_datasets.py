"""CLI для сырого аудита открытых uplift-датасетов R&D-6.

Назначение — однократный честный осмотр схемы реальных датасетов перед разметкой
A–F. Скачивание выполняется ТОЛЬКО локально, в gitignored-директорию
``data/06_safe_intime_cupac/<dataset>/``; сырые данные в git не попадают.

Политика данных:

- scikit-uplift импортируется лениво и используется, только если установлен;
- Criteo качается с ``percent10=True`` (для аудита достаточно);
- результаты схемы пишутся (опционально) только в gitignored ``data/...``;
- сырые данные НИКОГДА не пишутся в git-tracked пути.

Запуск::

    .venv/bin/python tools/audit_datasets.py                 # все датасеты
    .venv/bin/python tools/audit_datasets.py lenta criteo    # выбранные
    .venv/bin/python tools/audit_datasets.py --no-write       # не писать .txt-сводки

Если сеть/загрузка недоступны — печатается точная команда и причина, датасет
помечается как локально не подтверждённый (в отчёте — ``docs_only_not_locally_confirmed``).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

# Запускаем из корня репозитория; добавляем src в путь, чтобы не требовать установки.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rnd_reports.datasets.inspect import schema_summary  # noqa: E402

DATA_DIR = ROOT / "data" / "06_safe_intime_cupac"

DATASETS = (
    "lenta",
    "criteo",
    "x5_retailhero",
    "megafon",
    "orange_belgium",
    "lazada_descn",
)


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _summarize(name: str, df: pd.DataFrame, write: bool) -> None:
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
    summary = schema_summary(df)
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
        print(summary.to_string(index=False))
    if write:
        out_dir = DATA_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "schema_summary.txt"
        out_path.write_text(
            f"{name}: {df.shape[0]} rows x {df.shape[1]} cols\n\n"
            + summary.to_string(index=False),
            encoding="utf-8",
        )
        print(f"[written] {out_path}  (gitignored)")


# --- загрузчики через scikit-uplift (ленивый импорт) -----------------------


def _data_home() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR)


def load_lenta() -> pd.DataFrame:
    from sklift.datasets import fetch_lenta

    b = fetch_lenta(data_home=_data_home())
    df = b.data.copy()
    df[b.treatment_name] = b.treatment.values
    df[b.target_name] = b.target.values
    return df


def load_criteo() -> pd.DataFrame:
    from sklift.datasets import fetch_criteo

    b = fetch_criteo(percent10=True, data_home=_data_home())
    df = b.data.copy()
    df[b.treatment_name] = b.treatment.values
    tname = b.target_name if isinstance(b.target_name, str) else "target"
    df[tname] = b.target.values
    return df


def load_x5() -> pd.DataFrame:
    from sklift.datasets import fetch_x5

    b = fetch_x5(data_home=_data_home())
    # b.data — Bunch с clients / train / purchases. Инспектируем клиентские признаки
    # (train содержит treatment/target по client_id).
    clients = b.data["clients"] if "clients" in b.data else b.data.clients
    print(f"[x5] sub-frames: {list(b.data.keys()) if hasattr(b.data, 'keys') else 'bunch'}")
    df = clients.copy()
    return df


def load_megafon() -> pd.DataFrame:
    from sklift.datasets import fetch_megafon

    b = fetch_megafon(data_home=_data_home())
    df = b.data.copy()
    df[b.treatment_name] = b.treatment.values
    df[b.target_name] = b.target.values
    return df


def _download(url: str, dest: Path) -> Path:
    """Скачать файл по прямой ссылке в gitignored data/, если его ещё нет."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    return dest


def load_orange_belgium() -> pd.DataFrame:
    # Не входит в scikit-uplift. Прямой CSV из репозитория-статьи (Verhelst 2025);
    # фолбэк — OpenML id 45580.
    url = "https://www.dropbox.com/s/27kyinnh9jcjdcg/churn_uplift_anonymized.csv?dl=1"
    dest = DATA_DIR / "orange_belgium" / "churn_uplift_anonymized.csv"
    try:
        path = _download(url, dest)
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001 — фолбэк на OpenML
        from sklearn.datasets import fetch_openml

        b = fetch_openml(data_id=45580, as_frame=True, data_home=_data_home())
        return b.frame


def load_lazada_descn() -> pd.DataFrame:
    # Production-датасет Lazada из репозитория DESCN (прямой zip с Dropbox).
    # Инспектируем рандомизированный тест-сет (train — biased by prior policy).
    import zipfile

    out_dir = DATA_DIR / "lazada_descn"
    zpath = _download(
        "https://www.dropbox.com/s/07r7592h9mfijsb/lzd_data_public.zip?dl=1",
        out_dir / "lzd_data_public.zip",
    )
    test_csv = out_dir / "lzd_data_public" / "full_testset.csv"
    if not test_csv.exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out_dir)
    # тест-сет крупный — для аудита достаточно выборки
    return pd.read_csv(test_csv, nrows=200_000)


LOADERS = {
    "lenta": load_lenta,
    "criteo": load_criteo,
    "x5_retailhero": load_x5,
    "megafon": load_megafon,
    "orange_belgium": load_orange_belgium,
    "lazada_descn": load_lazada_descn,
}


# --- delta-эффект: реальные бенчмарки на адаптированных датасетах ------------

# Итоговые колонки сводки delta-эффектов (см. задание Step 4).
DELTA_COLUMNS = [
    "dataset", "validation_level", "method", "predecessor_method", "n", "target",
    "treatment_col", "ate", "se", "ci_low", "ci_high", "adjusted_target_variance",
    "variance_reduction_vs_ab_pct", "sample_size_reduction_vs_ab_pct",
    "variance_reduction_vs_cupac_A_pct", "sample_size_reduction_vs_cupac_A_pct",
    "incremental_variance_reduction_vs_predecessor_pct",
    "incremental_sample_size_reduction_vs_predecessor_pct",
    "feature_groups_used", "n_features_used", "safety_status", "diagnostic_notes",
]

RND_DIR = ROOT / "rnd" / "06_safe_intime_cupac"
DELTA_CSV = RND_DIR / "expanded_dataset_delta_results.csv"
FIGURES_DIR = RND_DIR / "figures"

# Какой максимально-валидный набор методов гонять на каждом датасете.
_A_ONLY_METHODS = ["ab_hypex", "sklearn_cupac_A"]


def _delta_specs() -> dict:
    """Спецификации delta-прогона: loader + тип + уровень валидации + методы.

    Все loader-ы ленивы (читают из gitignored data/); недоступные датасеты
    пропускаются с записью причины. C/D-«песочницы» помечаются явно и НЕ выдаются
    за реальную A+B+C-валидацию.
    """
    from rnd_reports.datasets import expanded_adapters as ea

    def _hillstrom():
        from rnd_reports.datasets.adapters import load_benchmark_dataset

        return load_benchmark_dataset("hillstrom")

    return {
        "hillstrom": dict(load=_hillstrom, dataset_type="real",
                          validation_level="A_only_real", methods=_A_ONLY_METHODS),
        "lenta": dict(load=ea.load_lenta_benchmark_dataset, dataset_type="real",
                      validation_level="A_only_real", methods=_A_ONLY_METHODS),
        "orange_belgium": dict(load=ea.load_orange_belgium_benchmark_dataset,
                               dataset_type="real", validation_level="A_only_real",
                               methods=_A_ONLY_METHODS),
        "lazada_descn": dict(load=ea.load_lazada_descn_benchmark_dataset,
                             dataset_type="real", validation_level="A_only_real",
                             methods=_A_ONLY_METHODS),
        "x5_retailhero": dict(load=ea.load_x5_a_only_benchmark_dataset,
                              dataset_type="real", validation_level="A_only_real",
                              methods=_A_ONLY_METHODS),
        "criteo": dict(load=ea.load_criteo_percent10_benchmark_dataset,
                       dataset_type="real", validation_level="A_only_real",
                       methods=_A_ONLY_METHODS),
        "open_bandit": dict(
            load=lambda: ea.load_open_bandit_research_dataset().benchmark,
            dataset_type="research_sandbox", validation_level="D_F_sandbox",
            methods=_A_ONLY_METHODS),
    }


def _result_rows(results, dataset, validation_level, treatment_col):
    rows = []
    for r in results:
        rows.append({
            "dataset": dataset,
            "validation_level": validation_level,
            "method": r.method,
            "predecessor_method": r.predecessor_method,
            "n": r.n,
            "target": r.target,
            "treatment_col": treatment_col,
            "ate": r.ate,
            "se": r.se,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "adjusted_target_variance": r.adjusted_target_variance,
            "variance_reduction_vs_ab_pct": r.variance_reduction_vs_ab_pct,
            "sample_size_reduction_vs_ab_pct": r.sample_size_reduction_vs_ab_pct,
            "variance_reduction_vs_cupac_A_pct": r.variance_reduction_vs_sklearn_cupac_pct,
            "sample_size_reduction_vs_cupac_A_pct": r.sample_size_reduction_vs_sklearn_cupac_pct,
            "incremental_variance_reduction_vs_predecessor_pct":
                r.incremental_variance_reduction_vs_predecessor_pct,
            "incremental_sample_size_reduction_vs_predecessor_pct":
                r.incremental_sample_size_reduction_vs_predecessor_pct,
            "feature_groups_used": "+".join(r.feature_groups_used) or "—",
            "n_features_used": r.n_features_used,
            "safety_status": r.safety_status,
            "diagnostic_notes": r.diagnostic_notes,
        })
    return rows


def run_delta_effects(datasets=None, random_state: int = 11) -> "pd.DataFrame":
    """Прогнать delta-эффект бенчмарки на адаптированных датасетах и записать CSV."""
    from rnd_reports.benchmark.protocol import run_benchmark

    specs = _delta_specs()
    names = datasets or list(specs)
    all_rows: list[dict] = []
    skipped: dict[str, str] = {}
    for name in names:
        spec = specs.get(name)
        if spec is None:
            skipped[name] = "нет delta-спецификации"
            continue
        _print_header(f"DELTA: {name}")
        try:
            bds = spec["load"]()
            results = run_benchmark(
                bds, methods=spec["methods"], dataset_name=name,
                dataset_type=spec["dataset_type"], random_state=random_state,
            )
            all_rows.extend(_result_rows(
                results, name, spec["validation_level"], bds.treatment_col))
            for r in results:
                print(f"  {r.method:32s} ate={r.ate:.4f} "
                      f"vr_vs_ab={r.variance_reduction_vs_ab_pct}")
            # Инкрементальная запись: партиал-результаты переживают сбой позднего датасета.
            RND_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_rows, columns=DELTA_COLUMNS).round(6).to_csv(
                DELTA_CSV, index=False)
        except Exception as exc:  # noqa: BLE001 — пропускаем недоступные датасеты
            skipped[name] = f"{type(exc).__name__}: {exc}"
            print(f"[SKIP] {name}: {type(exc).__name__}: {exc}")

    df = pd.DataFrame(all_rows, columns=DELTA_COLUMNS)
    RND_DIR.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(DELTA_CSV, index=False)
    print(f"\n[written] {DELTA_CSV}  ({len(df)} rows)")
    if skipped:
        _print_header("DELTA SKIPPED")
        for k, v in skipped.items():
            print(f"  {k}: {v}")
    return df


# --- coverage-карта представительности классов признаков по датасетам --------
# Ручная сводка (см. docs/06_expanded_dataset_scouting.md): 1 = есть/кандидат, 0 = нет.
FEATURE_COVERAGE = {
    #              A  B  C  D  EF
    "hillstrom":  [1, 0, 0, 0, 1],
    "lenta":      [1, 0, 0, 0, 0],
    "x5":         [1, 0, 0, 0, 1],
    "criteo":     [1, 0, 0, 1, 1],
    "megafon":    [1, 0, 0, 0, 0],
    "orange":     [1, 0, 0, 0, 0],
    "lazada":     [1, 0, 0, 1, 0],
    "open_bandit":[1, 0, 0, 1, 1],
    "dunnhumby":  [1, 0, 0, 1, 1],
    "criteo_priv":[1, 0, 0, 1, 1],
}


def make_figures(df=None) -> None:
    """Компактные PNG: ATE±CI, снижение дисперсии, coverage-heatmap классов."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if df is None and DELTA_CSV.exists():
        df = pd.read_csv(DELTA_CSV)

    if df is not None and len(df):
        # (a) ATE±CI по методам, фасет по датасету.
        for ds, sub in df.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(5, 2.6))
            y = np.arange(len(sub))
            ax.errorbar(sub["ate"], y,
                        xerr=[sub["ate"] - sub["ci_low"], sub["ci_high"] - sub["ate"]],
                        fmt="o", capsize=3)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["method"], fontsize=7)
            ax.set_title(f"ATE ± 95% CI — {ds}", fontsize=9)
            ax.axvline(0, color="grey", lw=0.6, ls="--")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / f"ate_ci_{ds}.png", dpi=90)
            plt.close(fig)

        # (b) снижение дисперсии vs A/B по методам и датасетам.
        piv = df.pivot_table(index="dataset", columns="method",
                             values="variance_reduction_vs_ab_pct", aggfunc="first")
        fig, ax = plt.subplots(figsize=(6, 3))
        piv.plot.bar(ax=ax)
        ax.set_ylabel("variance reduction vs A/B, %", fontsize=8)
        ax.set_title("Снижение дисперсии по методам", fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "variance_reduction_by_method.png", dpi=90)
        plt.close(fig)

    # (c) coverage-heatmap классов признаков по датасетам.
    classes = ["A", "B", "C", "D", "E/F"]
    mat = np.array([FEATURE_COVERAGE[k] for k in FEATURE_COVERAGE], dtype=float)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticks(range(len(FEATURE_COVERAGE)))
    ax.set_yticklabels(list(FEATURE_COVERAGE), fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "✓" if mat[i, j] else "·", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Представительность классов A–F", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_coverage_heatmap.png", dpi=90)
    plt.close(fig)
    print(f"[written] figures → {FIGURES_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="R&D-6 raw dataset audit")
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument(
        "--no-write", action="store_true", help="не писать .txt-сводки в data/"
    )
    parser.add_argument(
        "--delta", action="store_true",
        help="прогнать delta-эффект бенчмарки + фигуры (вместо raw-аудита)",
    )
    args = parser.parse_args()
    if args.delta:
        df = run_delta_effects(args.datasets or None)
        make_figures(df)
        return 0
    names = args.datasets or list(DATASETS)
    write = not args.no_write

    failures: dict[str, str] = {}
    for name in names:
        _print_header(f"DATASET: {name}")
        loader = LOADERS.get(name)
        if loader is None:
            print(f"[skip] неизвестный датасет: {name}")
            continue
        try:
            df = loader()
            _summarize(name, df, write)
        except Exception as exc:  # noqa: BLE001 — аудит должен пережить любой сбой загрузки
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)

    _print_header("SUMMARY")
    for name in names:
        status = "FAILED — " + failures[name] if name in failures else "ok"
        print(f"  {name}: {status}")
    if failures:
        print(
            "\nДатасеты со сбоем загрузки помечаются в отчёте как "
            "'docs_only_not_locally_confirmed'."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
