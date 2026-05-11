"""
Baseline multi-output classifiers for comparison with the DNN.

Each baseline trains **separate** estimators per output head (stress, energy, productivity)
on the **same** preprocessed train split as the neural model, and is evaluated on the same
test split using the **same** metrics as ``utils/evaluation.py``.

Run from ``Digital_Wellbeing_DNN``::

    python models/baseline_models.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing import PreprocessedData, preprocess_from_csv  # noqa: E402
from models.dnn_model import OUTPUT_NAMES  # noqa: E402
from utils.evaluation import (  # noqa: E402
    DEFAULT_MODEL_PATH as DEFAULT_DNN_PATH,
    OutputEvaluation,
    evaluate_model,
    load_model,
    print_metrics,
)


def train_logistic_regression_models(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    random_state: int = 42,
) -> Dict[str, LogisticRegression]:
    """Fit one multinomial logistic regression model per output head."""
    models: Dict[str, LogisticRegression] = {}
    for name in OUTPUT_NAMES:
        y_idx = np.argmax(y_train[name], axis=1)
        clf = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=random_state,
        )
        clf.fit(X_train, y_idx)
        models[name] = clf
    return models


def train_random_forest_models(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    random_state: int = 42,
) -> Dict[str, RandomForestClassifier]:
    """Fit one random forest classifier per output head."""
    models: Dict[str, RandomForestClassifier] = {}
    for name in OUTPUT_NAMES:
        y_idx = np.argmax(y_train[name], axis=1)
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_idx)
        models[name] = clf
    return models


def evaluate_sklearn_heads(
    models_by_head: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: Dict[str, np.ndarray],
) -> Dict[str, OutputEvaluation]:
    """Same metrics as DNN evaluation: accuracy + weighted P/R/F1 + confusion matrix per head."""
    results: Dict[str, OutputEvaluation] = {}
    labels = np.arange(3)

    for name in OUTPUT_NAMES:
        model = models_by_head[name]
        y_pred = model.predict(X_test)
        y_true = np.argmax(y_test[name], axis=1)

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


def print_comparison_summary(
    dnn: Dict[str, OutputEvaluation],
    logistic: Dict[str, OutputEvaluation],
    forest: Dict[str, OutputEvaluation],
) -> None:
    """Side-by-side comparison of weighted metrics across models."""
    print("\n" + "=" * 72)
    print("MODEL COMPARISON (same test split; sklearn metrics: weighted P/R/F1)")
    print("=" * 72)

    for head in OUTPUT_NAMES:
        d, l, r = dnn[head], logistic[head], forest[head]
        print(f"\n--- {head.upper()} ---")
        print(f"{'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 56)
        print(f"{'DNN':<14} {d.accuracy:10.4f} {d.precision:10.4f} {d.recall:10.4f} {d.f1:10.4f}")
        print(f"{'LogisticReg':<14} {l.accuracy:10.4f} {l.precision:10.4f} {l.recall:10.4f} {l.f1:10.4f}")
        print(f"{'RandomForest':<14} {r.accuracy:10.4f} {r.precision:10.4f} {r.recall:10.4f} {r.f1:10.4f}")

    print("\n" + "=" * 72 + "\n")


DEFAULT_COMPARISON_DIR = _PROJECT_ROOT / "artifacts" / "model_comparison"


def plot_model_comparison(
    dnn: Dict[str, OutputEvaluation],
    logistic: Dict[str, OutputEvaluation],
    forest: Dict[str, OutputEvaluation],
    save_dir: Path = DEFAULT_COMPARISON_DIR,
) -> None:
    """Save a grouped bar chart comparing Accuracy and F1-score across all models and heads."""
    save_dir.mkdir(parents=True, exist_ok=True)

    heads = list(OUTPUT_NAMES)
    model_names = ["DNN", "Logistic Regression", "Random Forest"]
    results_by_model = [dnn, logistic, forest]
    colors = ["#2563eb", "#16a34a", "#dc2626"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    x = np.arange(len(heads))
    width = 0.25

    for ax, metric_key, metric_label in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1", "F1-Score"),
    ]:
        for i, (name, res, color) in enumerate(
            zip(model_names, results_by_model, colors)
        ):
            vals = [getattr(res[h], metric_key) for h in heads]
            bars = ax.bar(x + i * width, vals, width, label=name, color=color, edgecolor="white", linewidth=0.6)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

        ax.set_xlabel("Output Head", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=13, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([h.capitalize() for h in heads], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = save_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


def run_full_comparison(data: PreprocessedData, dnn_model_path: Path) -> None:
    """Train baselines, evaluate DNN + baselines, print detailed blocks + summary table."""
    print("\n>>> Training Logistic Regression (3 independent heads)...\n")
    lr_models = train_logistic_regression_models(data.X_train, data.y_train)

    print(">>> Training Random Forest (3 independent heads)...\n")
    rf_models = train_random_forest_models(data.X_train, data.y_train)

    lr_results = evaluate_sklearn_heads(lr_models, data.X_test, data.y_test)
    rf_results = evaluate_sklearn_heads(rf_models, data.X_test, data.y_test)

    dnn = load_model(dnn_model_path)
    dnn_results = evaluate_model(dnn, data.X_test, data.y_test)

    print("\n############################  DNN  ############################\n")
    print_metrics(dnn_results)

    print("############################  LOGISTIC REGRESSION  ############################\n")
    print_metrics(lr_results)

    print("############################  RANDOM FOREST  ############################\n")
    print_metrics(rf_results)

    print_comparison_summary(dnn_results, lr_results, rf_results)
    plot_model_comparison(dnn_results, lr_results, rf_results)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train baseline models and compare with DNN.")
    p.add_argument(
        "--csv",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "synthetic_v1.csv"),
        help="Dataset CSV (same as training).",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Train/test split; must match training (default 0.2).")
    p.add_argument("--seed", type=int, default=42, help="Random seed; must match training (default 42).")
    p.add_argument(
        "--stratify-by",
        type=str,
        default="joint",
        choices=["stress", "energy", "productivity", "joint", "none"],
        help="Stratification; must match training (default joint).",
    )
    p.add_argument(
        "--dnn-model",
        type=str,
        default=str(DEFAULT_DNN_PATH),
        help="Path to trained DNN .keras file.",
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

    run_full_comparison(data, dnn_model_path=Path(args.dnn_model))


if __name__ == "__main__":
    main()
