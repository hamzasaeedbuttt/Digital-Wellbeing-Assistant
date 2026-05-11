"""
Interactive CLI for the Digital Wellbeing Assistant.

Collects all 9 raw lifestyle features, applies the **same** preprocessing pipeline used in
training (median imputation + ``StandardScaler``, fitted on the training split of the
dataset), runs the saved multi-output DNN, then prints predictions + rule-based guidance.

Run from ``Digital_Wellbeing_DNN``::

    python interface/cli.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing import FEATURE_COLUMNS, PreprocessingArtifacts, preprocess_from_csv  # noqa: E402
from models.dnn_model import OUTPUT_NAMES  # noqa: E402
from utils.evaluation import load_model  # noqa: E402
from utils.recommendations import format_bundle_as_text, recommendations_from_model_dict  # noqa: E402


PREDICTIONS_LOG_PATH = _PROJECT_ROOT / "artifacts" / "predictions.log"


def log_prediction(
    X_raw: np.ndarray,
    pred_dict: Dict[str, np.ndarray],
    artifacts: PreprocessingArtifacts,
    log_path: Path = PREDICTIONS_LOG_PATH,
) -> None:
    """Append a single prediction record (inputs, labels, confidences) to the log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    raw_values = X_raw.reshape(-1).tolist()
    inputs = {
        col: (None if np.isnan(v) else float(v))
        for col, v in zip(FEATURE_COLUMNS, raw_values)
    }

    predicted_labels: Dict[str, str] = {}
    confidence_scores: Dict[str, float] = {}
    for head in OUTPUT_NAMES:
        probs = np.asarray(pred_dict[head]).reshape(-1)
        idx = int(np.argmax(probs))
        categories = artifacts.target_encoders[head].categories_[0]
        predicted_labels[head] = str(categories[idx])
        confidence_scores[head] = float(probs[idx])

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs,
        "predicted_labels": predicted_labels,
        "confidence_scores": confidence_scores,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


FEATURE_PROMPTS: Dict[str, str] = {
    "sleep_hours": "Sleep hours last night (approx.; blank if unknown)",
    "workload_hours": "Total workload hours yesterday (work/study; blank if unknown)",
    "physical_activity_min": "Physical activity minutes yesterday (blank if unknown)",
    "screen_time_hours": "Non-work recreational screen time hours (blank if unknown)",
    "social_interaction_hours": "In-person social interaction hours (blank if unknown)",
    "water_intake_liters": "Approx. water intake liters (blank if unknown)",
    "caffeine_mg": "Approx. caffeine mg (coffee/tea/soda; blank if unknown)",
    "mood_score": "Mood score 1–10 (1 worst, 10 best; blank if unknown)",
    "breaks_count": "Number of intentional breaks taken (blank if unknown)",
}


def _parse_optional_float(text: str) -> float:
    s = text.strip()
    if s == "":
        return float("nan")
    return float(s)


def _parsed_feature_invalid_reason(col: str, val: float) -> Optional[str]:
    """Return an error message if a parsed numeric value is outside allowed ranges."""
    if col == "mood_score":
        if not (1.0 <= val <= 10.0):
            return "Mood score must be between 1 and 10."
    elif col in {"sleep_hours", "workload_hours"}:
        if not (0.0 <= val <= 24.0):
            return f"{col} must be between 0 and 24 hours."
    elif col in {
        "physical_activity_min",
        "screen_time_hours",
        "social_interaction_hours",
        "water_intake_liters",
        "caffeine_mg",
        "breaks_count",
    }:
        if val < 0.0:
            return f"{col} must be >= 0."
    return None


def prompt_raw_features_row() -> np.ndarray:
    """Prompt the user for each feature in ``FEATURE_COLUMNS`` order."""
    print("Example input (approximate values):")
    print("  sleep_hours=7, workload_hours=8, physical_activity_min=30, screen_time_hours=2")
    print("  social_interaction_hours=1.5, water_intake_liters=2.0, caffeine_mg=100")
    print("  mood_score=7, breaks_count=3\n")

    print("\nEnter your wellbeing signals (press Enter to leave a value unknown/missing).\n")
    values: list[float] = []
    for col in FEATURE_COLUMNS:
        prompt = FEATURE_PROMPTS.get(col, col)
        attempts = 0
        assigned = False
        while attempts < 3 and not assigned:
            raw = input(f"- {col}: {prompt}\n  > ")
            try:
                val = _parse_optional_float(raw)
            except ValueError:
                attempts += 1
                print("  Invalid number. Try again (or leave blank for unknown).")
                continue

            if np.isnan(val):
                values.append(float("nan"))
                assigned = True
                break

            reason = _parsed_feature_invalid_reason(col, val)
            if reason is not None:
                attempts += 1
                print(f"  {reason} Try again (or leave blank for unknown).")
                continue

            values.append(val)
            assigned = True

        if not assigned:
            print("Too many invalid attempts. Setting value as missing.")
            values.append(float("nan"))

    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def transform_raw_features(X_raw: np.ndarray, artifacts: PreprocessingArtifacts) -> np.ndarray:
    """Apply median imputation + StandardScaler using artifacts fitted on training data."""
    X_imp = artifacts.feature_imputer.transform(X_raw).astype(np.float32, copy=False)
    return artifacts.scaler.transform(X_imp).astype(np.float32, copy=False)


def print_probability_snapshot(pred_dict: Dict[str, np.ndarray]) -> None:
    """Compact view of softmax probabilities for each head."""
    print("\nSoftmax probabilities (per class, training encoder order):")
    for head in OUTPUT_NAMES:
        p = np.asarray(pred_dict[head]).reshape(-1)
        print(f"  - {head}: {np.round(p, 4).tolist()} (sum={float(np.sum(p)):.4f})")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive CLI: predict stress/energy/productivity + recommendations.")
    p.add_argument(
        "--csv",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "synthetic_v1.csv"),
        help="Dataset CSV used to fit preprocessing (must match training file for comparable transforms).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str(_PROJECT_ROOT / "models" / "wellbeing_dnn_trained.keras"),
        help="Trained multi-output Keras model path (.keras).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for preprocessing split (match training).")
    p.add_argument("--test-size", type=float, default=0.2, help="Train/test split fraction (match training).")
    p.add_argument(
        "--stratify-by",
        type=str,
        default="joint",
        choices=["stress", "energy", "productivity", "joint", "none"],
        help="Stratification strategy for preprocessing split (match training).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    print("Loading preprocessing pipeline (fits imputer/scaler on TRAIN split of dataset)...")
    data = preprocess_from_csv(
        csv_path=Path(args.csv),
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify_by=args.stratify_by,  # type: ignore[arg-type]
    )

    print(f"Loading model: {args.model}")
    model = load_model(Path(args.model))

    X_raw = prompt_raw_features_row()
    X_scaled = transform_raw_features(X_raw, data.artifacts)

    preds = model.predict(X_scaled, verbose=0)
    if not isinstance(preds, dict):
        raise TypeError("Expected dict predictions from multi-output model.")

    log_prediction(X_raw, preds, data.artifacts)

    bundle = recommendations_from_model_dict(preds, data.artifacts, row_index=0)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print_probability_snapshot(preds)
    print("")
    print(format_bundle_as_text(bundle))


if __name__ == "__main__":
    main()
