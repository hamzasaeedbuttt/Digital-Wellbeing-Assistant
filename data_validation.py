"""
Formal data validation pipeline for the Digital Wellbeing dataset.

Exposes ``validate_data(df)`` which raises ``ValueError`` with a descriptive
message on the first failed check. Designed to be called *before* preprocessing
so corrupt or out-of-range data is caught early and never reaches the model.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


NUMERIC_COLUMNS: Tuple[str, ...] = (
    "sleep_hours",
    "workload_hours",
    "physical_activity_min",
    "screen_time_hours",
    "social_interaction_hours",
    "water_intake_liters",
    "caffeine_mg",
    "mood_score",
    "breaks_count",
)
TARGET_COLUMNS: Tuple[str, ...] = ("stress", "energy", "productivity")
EXPECTED_COLUMNS: Tuple[str, ...] = NUMERIC_COLUMNS + TARGET_COLUMNS
EXPECTED_NUM_COLUMNS: int = len(EXPECTED_COLUMNS)

MAX_MISSING_FRACTION: float = 0.20
HOURS_MIN: float = 0.0
HOURS_MAX: float = 24.0


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the raw dataset before preprocessing.

    Raises:
        ValueError: with a clear, actionable message on the first failed check.
    """
    if df is None or df.empty:
        raise ValueError("Dataset validation failed: dataset is empty (no rows).")

    if df.shape[1] != EXPECTED_NUM_COLUMNS:
        raise ValueError(
            f"Dataset validation failed: expected {EXPECTED_NUM_COLUMNS} columns "
            f"({list(EXPECTED_COLUMNS)}), but found {df.shape[1]} ({list(df.columns)})."
        )

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset validation failed: missing required columns: {missing_cols}."
        )

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_fraction = total_missing / total_cells if total_cells > 0 else 0.0
    if missing_fraction >= MAX_MISSING_FRACTION:
        raise ValueError(
            f"Dataset validation failed: too many missing values "
            f"({total_missing} / {total_cells} = {missing_fraction:.2%}); "
            f"must be below {MAX_MISSING_FRACTION:.0%}."
        )

    for col in ("sleep_hours", "workload_hours"):
        series = df[col].dropna()
        out_of_range = series[(series < HOURS_MIN) | (series > HOURS_MAX)]
        if not out_of_range.empty:
            raise ValueError(
                f"Dataset validation failed: '{col}' has {len(out_of_range)} value(s) "
                f"outside [{HOURS_MIN}, {HOURS_MAX}] (e.g., {out_of_range.iloc[0]})."
            )

    for col in NUMERIC_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        negatives = series[series < 0]
        if not negatives.empty:
            raise ValueError(
                f"Dataset validation failed: numeric column '{col}' contains "
                f"{len(negatives)} negative value(s) (e.g., {negatives.iloc[0]})."
            )
