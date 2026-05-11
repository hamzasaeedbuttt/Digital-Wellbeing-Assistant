"""
Evaluate the trained multi-output Digital Wellbeing DNN.

Loads ``models/wellbeing_dnn_trained.keras``, runs preprocessing consistent with training,
computes per-output classification metrics and confusion matrices.

Run from ``Digital_Wellbeing_DNN``:

    python utils/evaluation.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

# Running as ``python utils/evaluation.py`` puts ``utils/`` on sys.path[0]; ensure project root resolves ``data.*``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing import preprocess_from_csv  # noqa: E402
from models.dnn_model import OUTPUT_NAMES  # noqa: E402

DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "wellbeing_dnn_trained.keras"
DEFAULT_CSV_PATH = _PROJECT_ROOT / "data" / "synthetic_v1.csv"


@dataclass(frozen=True)
class OutputEvaluation:
    """Per-head classification summary."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> keras.Model:
    """Load a trained Keras model from disk."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path.resolve()}")
    return keras.models.load_model(path)


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: Dict[str, np.ndarray],
) -> Dict[str, OutputEvaluation]:
    """
    Evaluate each output head separately.

    Args:
        model: Compiled/trained multi-output model (dict outputs).
        X_test: Scaled feature matrix for the held-out split.
        y_test: One-hot targets per output name.

    Returns:
        Mapping output name -> OutputEvaluation.
    """
    raw_preds = model.predict(X_test, verbose=0)
    if not isinstance(raw_preds, dict):
        raise TypeError("Expected dict predictions from multi-output model; check model outputs.")

    results: Dict[str, OutputEvaluation] = {}
    labels = np.arange(3)

    for name in OUTPUT_NAMES:
        if name not in raw_preds:
            raise KeyError(f"Missing predictions for output '{name}'. Keys: {list(raw_preds.keys())}")
        if name not in y_test:
            raise KeyError(f"Missing y_test for output '{name}'.")

        proba = np.asarray(raw_preds[name])
        y_true = np.argmax(y_test[name], axis=1)
        y_pred = np.argmax(proba, axis=1)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        results[name] = OutputEvaluation(
            name=name,
            accuracy=float(acc),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            confusion_matrix=cm,
        )

    return results


def print_metrics(results: Dict[str, OutputEvaluation], order: List[str] | None = None) -> None:
    """Pretty-print metrics for each output head."""
    names = order if order is not None else list(OUTPUT_NAMES)
    for name in names:
        ev = results[name]
        title = name.upper()
        print(f"=== {title} ===")
        print(f"Accuracy: {ev.accuracy:.4f}")
        print(f"Precision: {ev.precision:.4f}")
        print(f"Recall: {ev.recall:.4f}")
        print(f"F1-score: {ev.f1:.4f}")
        print("Confusion Matrix:")
        print(ev.confusion_matrix)
        print()


DEFAULT_CM_DIR = _PROJECT_ROOT / "artifacts" / "confusion_matrices"


def plot_confusion_matrices(
    results: Dict[str, OutputEvaluation],
    artifacts: "PreprocessingArtifacts",
    save_dir: Path = DEFAULT_CM_DIR,
) -> None:
    """
    Plot and save a confusion-matrix heatmap for each output head.

    Each plot is annotated with the class labels from the corresponding
    target encoder and saved as ``<head>_confusion_matrix.png``.
    """
    from data.preprocessing import PreprocessingArtifacts  # noqa: F811

    save_dir.mkdir(parents=True, exist_ok=True)

    for name in OUTPUT_NAMES:
        ev = results[name]
        labels = list(artifacts.target_encoders[name].categories_[0])

        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=ev.confusion_matrix,
            display_labels=labels,
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
        ax.set_title(f"{name.capitalize()} — Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        fig.tight_layout()
        out_path = save_dir / f"{name}_confusion_matrix.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained multi-output wellbeing DNN.")
    p.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained .keras model.",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help="Dataset CSV (same schema as training).",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Must match training split ratio. Default: 0.2.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Must match training random_state for preprocessing. Default: 42.",
    )
    p.add_argument(
        "--stratify-by",
        type=str,
        default="joint",
        choices=["stress", "energy", "productivity", "joint", "none"],
        help="Must match training stratification. Default: joint.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    data = preprocess_from_csv(
        csv_path=Path(args.csv),
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify_by=args.stratify_by,  # type: ignore[arg-type]
    )

    model = load_model(Path(args.model))
    results = evaluate_model(model, data.X_test, data.y_test)
    print_metrics(results)
    plot_confusion_matrices(results, data.artifacts)


if __name__ == "__main__":
    main()
