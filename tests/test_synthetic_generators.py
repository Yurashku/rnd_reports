"""Тесты портированного из VarWar генератора синтетики."""

from __future__ import annotations

import pandas as pd

from rnd_reports.synthetic.generators import DataGenerator

EXPECTED_COLUMNS = {
    "X1", "X1_lag", "X2", "X2_lag",
    "y0", "y0_lag_1", "y0_lag_2",
    "z", "U", "D", "d", "y1", "y",
}


def test_generate_shape_and_columns() -> None:
    df = DataGenerator(n_samples=500, seed=42).generate()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert EXPECTED_COLUMNS.issubset(df.columns)


def test_generate_is_reproducible_with_seed() -> None:
    df1 = DataGenerator(n_samples=300, seed=123).generate()
    df2 = DataGenerator(n_samples=300, seed=123).generate()
    pd.testing.assert_frame_equal(df1, df2)


def test_observed_outcome_matches_potential_outcomes() -> None:
    df = DataGenerator(n_samples=400, seed=7).generate()
    # y = y1 если принят (d==1), иначе y0
    assert (df.loc[df["d"] == 1, "y"] == df.loc[df["d"] == 1, "y1"]).all()
    assert (df.loc[df["d"] == 0, "y"] == df.loc[df["d"] == 0, "y0"]).all()
