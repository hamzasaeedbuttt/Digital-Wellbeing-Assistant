"""
End-to-end orchestration for the Digital Wellbeing DNN.

This module is a thin orchestrator that wires together existing components
without duplicating any of their logic:

    Step 1: Generate dataset       (data.data_generator)
    Step 2: Load + validate data   (data.preprocessing + utils.data_validation)
    Step 3: Run preprocessing      (data.preprocessing)
    Step 4: Train the model        (train.main)

Run from the ``Digital_Wellbeing_DNN`` directory::

    python pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_generator import generate_dataset, save_dataset_csv
from data.preprocessing import load_dataset, preprocess_dataframe
from utils.data_validation import validate_data
import train


DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "data" / "synthetic_v1.csv"
DEFAULT_N_SAMPLES: int = 3000
DEFAULT_SEED: int = 42
DEFAULT_EPOCHS: int = 75


def _section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def run_pipeline(
    csv_path: Path = DEFAULT_CSV_PATH,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
) -> None:
    """Execute the full ML pipeline end-to-end by composing existing modules."""

    _section("[Step 1/4] Generating synthetic dataset")
    df = generate_dataset(n_samples=n_samples, seed=seed)
    save_dataset_csv(df, output_path=csv_path)
    print(f"Dataset saved: {csv_path.resolve()} ({len(df)} rows)")

    _section("[Step 2/4] Loading dataset and running data validation")
    df = load_dataset(csv_path)
    validate_data(df)
    print(f"Validation passed: {df.shape[0]} rows, {df.shape[1]} columns.")

    _section("[Step 3/4] Running preprocessing")
    data = preprocess_dataframe(df, test_size=0.2, random_state=seed)
    print(f"Preprocessed shapes: X_train={data.X_train.shape}, X_test={data.X_test.shape}")

    _section("[Step 4/4] Training the model (delegating to train.main())")
    saved_argv = sys.argv
    sys.argv = [
        "train.py",
        "--csv", str(csv_path),
        "--seed", str(seed),
        "--epochs", str(epochs),
    ]
    try:
        train.main()
    finally:
        sys.argv = saved_argv

    _section("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
