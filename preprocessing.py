"""
Preprocessing utilities for the Digital Wellbeing DNN project.

Pipeline:
- Load dataset CSV
- Handle missing values (numeric features)
- Standardize features (StandardScaler)
- One-hot encode each target separately (stress, energy, productivity)
- Train/test split (80/20)

Outputs are returned in a model-friendly format:
    X_train, X_test: np.ndarray of shape (n, 9)
    y_train, y_test: dict[str, np.ndarray] where each value is one-hot encoded (n, 3)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMNS: Tuple[str, str, str] = ("stress", "energy", "productivity")
FEATURE_COLUMNS: Tuple[str, ...] = (
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


@dataclass
class PreprocessingArtifacts:
    feature_imputer: SimpleImputer
    scaler: StandardScaler
    target_encoders: Dict[str, OneHotEncoder]


@dataclass
class PreprocessedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: Dict[str, np.ndarray]
    y_test: Dict[str, np.ndarray]
    artifacts: PreprocessingArtifacts


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load the dataset and validate expected columns exist.
    """
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in (*FEATURE_COLUMNS, *TARGET_COLUMNS) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_cols)
            + f". Found columns: {list(df.columns)}"
        )
    # Enforce a consistent column order to prevent feature/label misalignment
    # when training or evaluating models across different runs/environments.
    df = df[list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)]
    return df


def _validate_targets(df: pd.DataFrame) -> None:
    for col in TARGET_COLUMNS:
        if df[col].isna().any():
            raise ValueError(f"Target column '{col}' contains missing values. Regenerate dataset or fix labels.")
        # Keep validation flexible, but reject empty strings.
        if (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"Target column '{col}' contains empty/blank labels.")


def preprocess_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: Literal["stress", "energy", "productivity", "joint", "none"] = "joint",
) -> PreprocessedData:
    """
    Preprocess a dataset DataFrame into train/test arrays.

    Stratification:
    - 'joint' stratifies using combined labels "stress|energy|productivity" for better multi-output balance.
    - If stratification fails (e.g., too few samples per class), fallback to 'none'.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    _validate_targets(df)

    X = df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    y_raw = {col: df[col].astype(str).to_numpy(copy=True) for col in TARGET_COLUMNS}

    if stratify_by == "none":
        stratify = None
    elif stratify_by in TARGET_COLUMNS:
        stratify = y_raw[stratify_by]
    elif stratify_by == "joint":
        stratify = np.char.add(
            np.char.add(y_raw["stress"], "|"),
            np.char.add(np.char.add(y_raw["energy"], "|"), y_raw["productivity"]),
        )
    else:
        raise ValueError("Invalid stratify_by. Choose from: stress, energy, productivity, joint, none.")

    # Split indices first to avoid leakage from imputation/scaling/encoding.
    idx = np.arange(len(df))
    try:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=None
        )

    X_train_raw, X_test_raw = X[idx_train], X[idx_test]

    # Imputation is required because real-world digital wellbeing logs can be incomplete
    # (e.g., missing sleep or activity measurements). Most ML/DL models cannot train with NaNs.
    feature_imputer = SimpleImputer(strategy="median")
    X_train_imp = feature_imputer.fit_transform(X_train_raw).astype(np.float32, copy=False)
    X_test_imp = feature_imputer.transform(X_test_raw).astype(np.float32, copy=False)

    # StandardScaler ensures each feature has comparable scale (zero mean, unit variance).
    # This stabilizes and speeds up gradient descent by preventing large-scale features
    # from dominating the loss/gradients during training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test_imp).astype(np.float32, copy=False)

    # We use separate OneHotEncoders because the model is multi-output with separate heads:
    # each target (stress/energy/productivity) is its own 3-class classification problem.
    target_encoders: Dict[str, OneHotEncoder] = {}
    y_train: Dict[str, np.ndarray] = {}
    y_test: Dict[str, np.ndarray] = {}

    for col in TARGET_COLUMNS:
        # Use `sparse=False` for compatibility with older scikit-learn versions.
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        y_train_col = y_raw[col][idx_train].reshape(-1, 1)
        y_test_col = y_raw[col][idx_test].reshape(-1, 1)

        y_train[col] = enc.fit_transform(y_train_col).astype(np.float32, copy=False)
        y_test[col] = enc.transform(y_test_col).astype(np.float32, copy=False)
        target_encoders[col] = enc

        # Sanity check: expected 3 classes for each output.
        if y_train[col].shape[1] != 3:
            raise ValueError(
                f"Target '{col}' has {y_train[col].shape[1]} classes after encoding; expected 3. "
                "Check dataset label values and balance."
            )

    artifacts = PreprocessingArtifacts(
        feature_imputer=feature_imputer,
        scaler=scaler,
        target_encoders=target_encoders,
    )
    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        artifacts=artifacts,
    )


def preprocess_from_csv(
    csv_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: Literal["stress", "energy", "productivity", "joint", "none"] = "joint",
) -> PreprocessedData:
    df = load_dataset(csv_path)
    return preprocess_dataframe(df, test_size=test_size, random_state=random_state, stratify_by=stratify_by)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess the synthetic wellbeing dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("Digital_Wellbeing_DNN") / "data" / "synthetic_v1.csv"),
        help="Input CSV path (relative or absolute).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction. Default: 0.2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="joint",
        choices=["stress", "energy", "productivity", "joint", "none"],
        help="Stratification strategy. Default: joint",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data = preprocess_from_csv(
        csv_path=Path(args.input),
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify_by=args.stratify_by,  # type: ignore[arg-type]
    )

    print("Preprocessing complete.")
    print(f"- X_train: {data.X_train.shape}, X_test: {data.X_test.shape}")
    for k in TARGET_COLUMNS:
        print(f"- y_train[{k}]: {data.y_train[k].shape}, y_test[{k}]: {data.y_test[k].shape}")
    print(f"- Feature columns ({len(FEATURE_COLUMNS)}): {list(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()

