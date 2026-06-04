"""Lightweight tests for the R&D-6 real-dataset raw audit.

No downloads: ``schema_summary`` is a pure function and is exercised on a tiny
in-memory DataFrame; catalog/doc checks are static.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rnd_reports.datasets.catalog import CATALOG, get_dataset_spec
from rnd_reports.datasets.inspect import schema_summary

_EXPECTED_COLS = [
    "column",
    "dtype",
    "missing_pct",
    "n_unique",
    "example_values",
    "looks_like_datetime",
    "looks_like_id",
    "looks_like_binary",
    "looks_like_target_or_treatment",
]


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signup_date": ["2021-01-01", "2021-02-15", "2021-03-30", None],
            "treatment": [0, 1, 0, 1],
            "amount": [1.5, 2.0, 3.5, 4.0],
        }
    )


def test_schema_summary_returns_expected_columns() -> None:
    out = schema_summary(_tiny_df())
    assert list(out.columns) == _EXPECTED_COLS
    assert len(out) == 3  # one row per input column


def test_schema_summary_flags_datetime_column() -> None:
    out = schema_summary(_tiny_df()).set_index("column")
    assert bool(out.loc["signup_date", "looks_like_datetime"]) is True
    assert bool(out.loc["amount", "looks_like_datetime"]) is False
    # treatment-like name flagged, binary detected
    assert bool(out.loc["treatment", "looks_like_target_or_treatment"]) is True
    assert bool(out.loc["treatment", "looks_like_binary"]) is True


def test_catalog_contains_orange_belgium() -> None:
    assert "orange_belgium" in CATALOG
    spec = get_dataset_spec("orange_belgium")
    assert spec.source_url


def test_megafon_marked_synthetic() -> None:
    assert get_dataset_spec("megafon").kind == "synthetic"


def test_criteo_marked_randomized() -> None:
    assert get_dataset_spec("criteo").is_randomized is True


def test_audit_doc_exists() -> None:
    doc = Path(__file__).resolve().parents[1] / "docs" / "06_real_dataset_raw_audit.md"
    assert doc.exists()
