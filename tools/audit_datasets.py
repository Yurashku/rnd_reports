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


def main() -> int:
    parser = argparse.ArgumentParser(description="R&D-6 raw dataset audit")
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument(
        "--no-write", action="store_true", help="не писать .txt-сводки в data/"
    )
    args = parser.parse_args()
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
