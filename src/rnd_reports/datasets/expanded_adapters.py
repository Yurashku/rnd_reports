"""Прототипы адаптеров расширенного датасет-скаутинга R&D-6 (Step 3 итерации).

Превращают локально скачанные открытые датасеты в ``BenchmarkDataset`` (классические
uplift A/B-датасеты) либо в документированный ``ResearchDataset`` (event-log / bandit /
journey / advertising «песочницы», которые НЕ являются прямыми A/B uplift-бенчмарками).

Жёсткие правила (см. docs/06_expanded_dataset_scouting.md, контекст §3):

- Реальные данные читаются из gitignored ``data/06_safe_intime_cupac/<dataset>/`` и
  **не коммитятся**; если файла нет — понятная ``FileNotFoundError`` с подсказкой.
- Признаки CUPAC (класс A) приводятся к числовому виду без NaN (требование
  ``local_cupac_adjust`` / ``safe_intime_linear_adjustment``).
- Анонимные снимки (``f*``/``PC*``/``X_*``) размечаются как A — их **нельзя** объявлять
  безопасными B/C.
- В «песочницах» класс C — это *сконструированные* lag-признаки строго до анкера; они
  помечаются как sandbox-кандидаты и **никогда** не выдаются за реальную B/C-валидацию.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..feature_safety.contracts import FeatureClass
from .contracts import BenchmarkDataset
from .loaders import DEFAULT_DATA_DIR

# Корень для локально скачанных датасетов (gitignored).
EXPANDED_DATA_DIR = DEFAULT_DATA_DIR

# Метки типа датасета.
DATASET_TYPE_REAL = "real"
DATASET_TYPE_RESEARCH = "research_sandbox"
DATASET_TYPE_EVENT_LOG = "event_log_sandbox"


@dataclass
class ResearchDataset:
    """Обёртка для не-A/B «песочниц»: ``BenchmarkDataset`` + анкер/cutoff/обоснование.

    Используется для event-log / bandit / journey / advertising датасетов, где прямой
    A/B-контракт неуместен. Несёт явное определение анкера и time-cutoff, а также
    пер-признаковое обоснование классов A/C/D/E/F. Любой класс C здесь —
    *сконструированный* sandbox-кандидат, не реальная safe-in-time валидация.
    """

    benchmark: BenchmarkDataset
    dataset_type: str  # research_sandbox / event_log_sandbox
    anchor: str
    time_cutoff: str
    feature_rationale: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def data(self) -> pd.DataFrame:
        return self.benchmark.data


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #


def _require_file(path: Path, dataset: str, hint: str) -> Path:
    """Проверить наличие локального файла; иначе — понятная ошибка с подсказкой."""
    if not path.exists():
        raise FileNotFoundError(
            f"Датасет '{dataset}' не найден по пути {path}. Реальные данные не "
            f"коммитятся в git и качаются вручную в gitignored data/. {hint}"
        )
    return path


def _numeric_no_nan(df: pd.DataFrame, cols: list[str], fill: float = 0.0) -> pd.DataFrame:
    """Привести колонки к float и заполнить NaN (требование CUPAC/linear-stage)."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float).fillna(fill)
    return out


def _one_hot(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot для object-категорий; вернуть (data, новые числовые колонки)."""
    if not cols:
        return df, []
    dummies = pd.get_dummies(df[cols].astype(str), prefix=cols).astype(int)
    return pd.concat([df.drop(columns=cols), dummies], axis=1), list(dummies.columns)


# --------------------------------------------------------------------------- #
# Классические uplift-датасеты → BenchmarkDataset (A-only)
# --------------------------------------------------------------------------- #


def load_lenta_benchmark_dataset(
    path: Optional[str] = None, nrows: Optional[int] = None
) -> BenchmarkDataset:
    """Lenta retail uplift → ``BenchmarkDataset`` (A-only, именованные пре-кампейн агрегаты).

    treatment = ``group == 'test'``; target = ``response_att``. Все числовые
    пре-кампейн признаки (history/sale/cheque/k_var/... + age/children/main_format +
    one-hot gender) — класс **A**. Реальных safe in-time B/C нет (окна 15d/1m/.../12m —
    пре-период, не in-time). См. docs/06_real_dataset_raw_audit.md §2.
    """
    p = Path(path) if path else EXPANDED_DATA_DIR / "lenta" / "lenta_dataset.csv.gz"
    _require_file(
        p, "lenta", "sklift.datasets.fetch_lenta(data_home=<DATA_DIR>) → lenta/lenta_dataset.csv.gz."
    )
    df = pd.read_csv(p, nrows=nrows, low_memory=False)

    out = pd.DataFrame()
    out["id"] = np.arange(len(df))
    out["treatment"] = (df["group"].astype(str) == "test").astype(int)
    out["target"] = pd.to_numeric(df["response_att"], errors="coerce").astype(float)
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    df = df.loc[out.index].reset_index(drop=True) if len(out) != len(df) else df

    drop = {"group", "response_att"}
    cat_cols = [c for c in df.columns if c not in drop and df[c].dtype == object]
    num_cols = [c for c in df.columns if c not in drop and c not in cat_cols]

    out = _numeric_no_nan(pd.concat([out, df[num_cols]], axis=1), num_cols)
    out, oh_cols = _one_hot(pd.concat([out, df[cat_cols]], axis=1), cat_cols)

    a_features = num_cols + oh_cols
    registry = {c: "A_pre_treatment" for c in a_features}
    return BenchmarkDataset(
        data=out, id_col="id", treatment_col="treatment",
        target_col="target", feature_registry=registry,
    )


def load_orange_belgium_benchmark_dataset(
    path: Optional[str] = None, nrows: Optional[int] = None
) -> BenchmarkDataset:
    """Orange Belgium churn uplift → ``BenchmarkDataset`` (A-only, анонимные PCA-признаки).

    treatment = ``t``; target = ``y`` (churn). Признаки: ``PC1..PC160`` (PCA-компоненты,
    числовые) + ``FACTOR1..FACTOR18`` (анонимные категории → one-hot) — все класс **A**.
    PCA уничтожает семантику → safe B/C недоступны. См. raw audit §1.
    """
    p = Path(path) if path else EXPANDED_DATA_DIR / "orange_belgium" / "churn_uplift_anonymized.csv"
    _require_file(p, "orange_belgium", "Прямой CSV (Dropbox из TheoVerhelst/Churn-Uplift-Dataset-Paper) или OpenML 45580.")
    df = pd.read_csv(p, nrows=nrows)

    out = pd.DataFrame()
    out["id"] = np.arange(len(df))
    out["treatment"] = pd.to_numeric(df["t"], errors="coerce").astype(int)
    out["target"] = pd.to_numeric(df["y"], errors="coerce").astype(float)

    pc_cols = [c for c in df.columns if c.upper().startswith("PC")]
    factor_cols = [c for c in df.columns if c.upper().startswith("FACTOR")]
    out = _numeric_no_nan(pd.concat([out, df[pc_cols]], axis=1), pc_cols)
    out, oh_cols = _one_hot(pd.concat([out, df[factor_cols]], axis=1), factor_cols)

    a_features = pc_cols + oh_cols
    registry = {c: "A_pre_treatment" for c in a_features}
    return BenchmarkDataset(
        data=out, id_col="id", treatment_col="treatment",
        target_col="target", feature_registry=registry,
    )


def load_lazada_descn_benchmark_dataset(
    path: Optional[str] = None, nrows: Optional[int] = None
) -> BenchmarkDataset:
    """Lazada/DESCN voucher uplift (рандомизированный тест) → ``BenchmarkDataset`` (A-only).

    treatment = ``is_treat``; target = ``label``; ``f0..f82`` — анонимные признаки → A.
    Используется ТОЛЬКО рандомизированный ``full_testset.csv`` (train смещён политикой).
    См. raw audit §6.
    """
    p = (
        Path(path)
        if path
        else EXPANDED_DATA_DIR / "lazada_descn" / "lzd_data_public" / "full_testset.csv"
    )
    _require_file(p, "lazada_descn", "Прямой zip lzd_data_public.zip (Dropbox из репозитория DESCN).")
    df = pd.read_csv(p, nrows=nrows)

    out = pd.DataFrame()
    out["id"] = np.arange(len(df))
    out["treatment"] = pd.to_numeric(df["is_treat"], errors="coerce").astype(int)
    out["target"] = pd.to_numeric(df["label"], errors="coerce").astype(float)

    f_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    out = _numeric_no_nan(pd.concat([out, df[f_cols]], axis=1), f_cols)

    registry = {c: "A_pre_treatment" for c in f_cols}
    return BenchmarkDataset(
        data=out, id_col="id", treatment_col="treatment",
        target_col="target", feature_registry=registry,
    )


def load_x5_a_only_benchmark_dataset(
    data_dir: Optional[str] = None, chunksize: int = 5_000_000
) -> BenchmarkDataset:
    """X5 RetailHero → ``BenchmarkDataset`` (A-only через инженерию истории покупок).

    Мульти-таблица: ``uplift_train`` (client_id, treatment_flg, target) ⨝ ``clients``
    (age, gender) ⨝ агрегаты ``purchases`` (45.8M строк). treatment = ``treatment_flg``;
    target = ``target``. Признаки класса **A** — пер-клиентская история покупок
    (count/sum/mean/points/recency) + age + one-hot gender.

    ВАЖНО (ограничение): дата кампании в публичных данных не опубликована, поэтому
    разделить покупки на pre/in-treatment нельзя — агрегаты берутся по всей истории и
    трактуются как A (best-effort). Защитимых in-time C из этого датасета НЕ строим.
    Покупки читаются чанками, чтобы не держать 45M строк в памяти целиком.
    """
    root = Path(data_dir) if data_dir else EXPANDED_DATA_DIR / "x5"
    up = _require_file(root / "uplift_train.csv.gz", "x5_retailhero", "sklift.datasets.fetch_x5(data_home=<DATA_DIR>).")
    cl = _require_file(root / "clients.csv.gz", "x5_retailhero", "sklift.datasets.fetch_x5(...).")
    pu = _require_file(root / "purchases.csv.gz", "x5_retailhero", "sklift.datasets.fetch_x5(...).")

    train = pd.read_csv(up)
    clients = pd.read_csv(cl)

    # Чанк-аддитивная векторная агрегация по client_id (память-безопасно, без iterrows).
    # Суммы/счётчики накапливаем через .add(fill_value=0); максимум даты — через concat+max.
    usecols = ["client_id", "transaction_datetime", "purchase_sum",
               "regular_points_received", "product_quantity"]
    sums = None  # DataFrame: n_rows, sum_purchase, sum_points, sum_qty
    last_dt = None  # Series: max transaction_datetime по client_id
    max_dt = pd.Timestamp.min
    for chunk in pd.read_csv(pu, usecols=usecols, chunksize=chunksize,
                             parse_dates=["transaction_datetime"]):
        max_dt = max(max_dt, chunk["transaction_datetime"].max())
        g = chunk.groupby("client_id")
        part = pd.DataFrame({
            "n_rows": g.size(),
            "sum_purchase": g["purchase_sum"].sum(),
            "sum_points": g["regular_points_received"].sum(),
            "sum_qty": g["product_quantity"].sum(),
        })
        sums = part if sums is None else sums.add(part, fill_value=0.0)
        part_dt = g["transaction_datetime"].max()
        last_dt = part_dt if last_dt is None else (
            pd.concat([last_dt, part_dt]).groupby(level=0).max())

    hist = sums
    hist["recency_days"] = (max_dt - last_dt).dt.days.astype(float)
    hist["mean_purchase"] = hist["sum_purchase"] / hist["n_rows"].clip(lower=1)
    hist = hist.reset_index().rename(columns={"index": "client_id"})

    df = train.merge(clients, on="client_id", how="left").merge(hist, on="client_id", how="left")

    out = pd.DataFrame()
    out["id"] = np.arange(len(df))
    out["treatment"] = pd.to_numeric(df["treatment_flg"], errors="coerce").astype(int)
    out["target"] = pd.to_numeric(df["target"], errors="coerce").astype(float)

    hist_cols = ["n_rows", "sum_purchase", "sum_points", "sum_qty",
                 "recency_days", "mean_purchase", "age"]
    out = _numeric_no_nan(pd.concat([out, df[hist_cols]], axis=1), hist_cols)
    out, oh_cols = _one_hot(pd.concat([out, df[["gender"]]], axis=1), ["gender"])

    a_features = hist_cols + oh_cols
    registry = {c: "A_pre_treatment" for c in a_features}
    return BenchmarkDataset(
        data=out, id_col="id", treatment_col="treatment",
        target_col="target", feature_registry=registry,
    )


def load_criteo_percent10_benchmark_dataset(
    data_home: Optional[str] = None,
) -> BenchmarkDataset:
    """Criteo Uplift v2.1 (10%) → ``BenchmarkDataset`` (слабый A-only; exposure → E/unsafe).

    treatment = ``treatment``; target = ``visit``; ``f0..f11`` анонимны → класс **A**.
    ``exposure`` (RTB-показ, post-treatment) размечается как **E_mediator_risk** и в
    основной estimator не идёт. Требует локального sklift-кэша/сети. См. raw audit §3.
    """
    from sklift.datasets import fetch_criteo

    home = str(data_home) if data_home else str(EXPANDED_DATA_DIR)
    b = fetch_criteo(percent10=True, data_home=home)
    df = b.data.copy()
    tname = b.treatment_name if isinstance(b.treatment_name, str) else "treatment"
    df[tname] = b.treatment.values
    # target: предпочитаем visit (если несколько таргетов — берём первый/visit)
    targets = b.target if hasattr(b.target, "columns") else None
    if targets is not None and "visit" in getattr(b.target, "columns", []):
        df["visit"] = b.target["visit"].values
    else:
        df["visit"] = np.asarray(b.target).ravel()[: len(df)]

    out = pd.DataFrame()
    out["id"] = np.arange(len(df))
    out["treatment"] = pd.to_numeric(df[tname], errors="coerce").astype(int)
    out["target"] = pd.to_numeric(df["visit"], errors="coerce").astype(float)

    f_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    out = _numeric_no_nan(pd.concat([out, df[f_cols]], axis=1), f_cols)

    registry = {c: "A_pre_treatment" for c in f_cols}
    if "exposure" in df.columns:
        out["exposure"] = pd.to_numeric(df["exposure"], errors="coerce").astype(float)
        registry["exposure"] = "E_mediator_risk"  # post-treatment, unsafe_demo
    return BenchmarkDataset(
        data=out, id_col="id", treatment_col="treatment",
        target_col="target", feature_registry=registry,
    )


# --------------------------------------------------------------------------- #
# Не-A/B «песочницы» → ResearchDataset (event-log / bandit / advertising)
# --------------------------------------------------------------------------- #


def load_open_bandit_research_dataset(
    behavior_policy: str = "random", campaign: str = "all"
) -> ResearchDataset:
    """Open Bandit Dataset (ZOZOTOWN) → ``ResearchDataset`` (logged-bandit песочница).

    НЕ простой A/B: логированная политика рекомендаций. treatment = ``action`` показан
    vs нет (бинаризуем по позиции/факту показа как демо); target = ``reward`` (click).
    Разметка-подсказка: контекст пользователя/товара = **A**; ``position`` / ``pscore``
    (propensity) / ``action`` = **D** (механика политики); сконструированные лаги истории
    пользователя строго до timestamp = sandbox **C**; ``reward``-производные = **F**.
    Требует пакет ``obp`` (ставится в .venv, не в зависимости проекта).
    """
    try:
        from obp.dataset import OpenBanditDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Open Bandit требует пакет 'obp' (локально: .venv/bin/pip install obp)."
        ) from exc

    ds = OpenBanditDataset(behavior_policy=behavior_policy, campaign=campaign)
    fb = ds.obtain_batch_bandit_feedback()
    n = fb["n_rounds"]
    ctx = np.asarray(fb["context"])
    out = pd.DataFrame({c: ctx[:, i] for i, c in
                        enumerate([f"ctx_{j}" for j in range(ctx.shape[1])])})
    out.insert(0, "id", np.arange(n))
    # Демонстрационный treatment: показан ли «топовый» action (position==0).
    out["treatment"] = (np.asarray(fb["position"]) == 0).astype(int)
    out["target"] = np.asarray(fb["reward"]).astype(float)
    out["action"] = np.asarray(fb["action"]).astype(float)
    out["position"] = np.asarray(fb["position"]).astype(float)
    out["pscore"] = np.asarray(fb["pscore"]).astype(float)

    a_features = [c for c in out.columns if c.startswith("ctx_")]
    registry = {c: "A_pre_treatment" for c in a_features}
    registry.update({"action": "D_dag_required", "position": "D_dag_required",
                     "pscore": "D_dag_required"})
    bds = BenchmarkDataset(data=out, id_col="id", treatment_col="treatment",
                           target_col="target", feature_registry=registry)
    return ResearchDataset(
        benchmark=bds, dataset_type=DATASET_TYPE_EVENT_LOG,
        anchor="impression/round timestamp (logged policy decision)",
        time_cutoff="features known at the bidding/recommendation moment",
        feature_rationale={
            "ctx_*": "A: пользовательский/товарный контекст до действия",
            "action/position/pscore": "D: механика политики/propensity (нужен DAG)",
            "reward-derived (lagged within session)": "F: производные от исхода/leakage",
        },
        notes="Logged-bandit, не рандомизированный A/B. treatment бинаризован как демо.",
    )


def load_completejourney_research_dataset(data_dir: Optional[str] = None) -> ResearchDataset:
    """dunnhumby Complete Journey → ``ResearchDataset`` (retail journey, observational).

    НЕ рандомизированный A/B (coupon не назначается случайно). Мульти-таблица с реальными
    датами: transactions/campaigns/coupons/coupon_redempt. Разметка-подсказка относительно
    анкера «дата первого контакта кампании»: пред-анкерная история покупок = **A**;
    in-window активность до cutoff исхода = sandbox **C**; кампании/купоны/контакты =
    **D/E**; пост-анкерные покупки/redemption = **E/F**.

    Чтение требует локально распакованных CSV (data/.../dunnhumby/). При отсутствии —
    понятная ошибка; полноценный observational-контракт строить не следует (demo-only).
    """
    root = Path(data_dir) if data_dir else EXPANDED_DATA_DIR / "dunnhumby"
    trans = _require_file(
        root / "transaction_data.csv", "dunnhumby",
        "dunnhumby Complete Journey (registration-gated) или community-зеркало → dunnhumby/.",
    )
    raise NotImplementedError(
        "Complete Journey — observational sandbox: контракт строится только при наличии "
        f"распакованных таблиц рядом с {trans} и защитимого анкер-определения; "
        "как RCT не используется (см. docs/06_expanded_dataset_scouting.md)."
    )


def load_criteo_private_ad_research_dataset(data_dir: Optional[str] = None) -> ResearchDataset:
    """CriteoPrivateAd → ``ResearchDataset`` (advertising logs, privacy-конструкция).

    НЕ классический treatment/control. Подсказка: pre-display контекст/request = **A**;
    display/order/campaign/privacy-механика = **D**; delayed click/sale/report-производные
    = **F/leakage**; защитимый in-time C — только при очень аккуратном анкере (обычно нет).
    Требует HF ``datasets`` и локально скачанный parquet (многогигабайтный).
    """
    root = Path(data_dir) if data_dir else EXPANDED_DATA_DIR / "criteo_private_ad"
    raise NotImplementedError(
        "CriteoPrivateAd — advertising-песочница (D/F): прямого A/B-контракта нет. "
        f"Нужен локальный parquet в {root} и аккуратный анкер; как safe B/C не используется."
    )


__all__ = [
    "ResearchDataset",
    "EXPANDED_DATA_DIR",
    "load_lenta_benchmark_dataset",
    "load_orange_belgium_benchmark_dataset",
    "load_lazada_descn_benchmark_dataset",
    "load_x5_a_only_benchmark_dataset",
    "load_criteo_percent10_benchmark_dataset",
    "load_open_bandit_research_dataset",
    "load_completejourney_research_dataset",
    "load_criteo_private_ad_research_dataset",
]
