"""
Train the multi-output Digital Wellbeing DNN.

Loads data via preprocessing (scaled features, one-hot targets), fits with
``model.fit(..., y=dict)`` and ``validation_data``, saves the trained model,
and writes loss/accuracy plots under ``artifacts/plots/``.

Run from this directory (``Digital_Wellbeing_DNN``):

    python train.py --epochs 75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

from tensorflow.keras.callbacks import EarlyStopping, History

from data.preprocessing import load_dataset, preprocess_dataframe
from models.dnn_model import OUTPUT_NAMES, build_and_compile_wellbeing_dnn
from utils.data_validation import validate_data

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "synthetic_v1.csv"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "models" / "wellbeing_dnn_trained.keras"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_PLOTS_DIR = DEFAULT_ARTIFACTS_DIR / "plots"
DEFAULT_TRAINING_LOG = DEFAULT_ARTIFACTS_DIR / "training_log.json"


def save_training_log(history: History, log_path: Path) -> None:
    """Persist Keras ``history.history`` as a JSON file (creating parent dirs)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def plot_training_loss(history: History, save_path: Path) -> None:
    """Plot total and per-output training/validation loss."""
    h: Dict[str, Any] = history.history
    epochs = np.arange(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, h["loss"], label="train (total)")
    axes[0].plot(epochs, h["val_loss"], label="val (total)")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total loss (weighted sum of output losses)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    for name in OUTPUT_NAMES:
        tk = f"{name}_loss"
        vk = f"val_{name}_loss"
        if tk in h:
            axes[1].plot(epochs, h[tk], label=f"{name} train")
        if vk in h:
            axes[1].plot(epochs, h[vk], linestyle="--", label=f"{name} val")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Per-output loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_accuracy(history: History, save_path: Path) -> None:
    """Plot per-output training/validation accuracy."""
    h: Dict[str, Any] = history.history
    epochs = np.arange(1, len(h["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for name in OUTPUT_NAMES:
        ak = f"{name}_accuracy"
        vk = f"val_{name}_accuracy"
        if ak in h:
            ax.plot(epochs, h[ak], label=f"{name} train")
        if vk in h:
            ax.plot(epochs, h[vk], linestyle="--", label=f"{name} val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-output accuracy")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train multi-output wellbeing DNN.")
    p.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help="Path to synthetic_v1.csv",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=75,
        help="Training epochs (must be between 50 and 100). Default: 75.",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Batch size. Default: 32.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for preprocessing split.")
    p.add_argument(
        "--model-out",
        type=str,
        default=str(DEFAULT_MODEL_OUT),
        help="Where to save trained Keras model (.keras).",
    )
    p.add_argument(
        "--plots-dir",
        type=str,
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory for training curve PNGs.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not (50 <= args.epochs <= 100):
        raise SystemExit("--epochs must be between 50 and 100 (inclusive).")

    df = load_dataset(Path(args.csv))
    try:
        validate_data(df)
    except ValueError as exc:
        raise SystemExit(f"Aborting training: {exc}")

    data = preprocess_dataframe(df, test_size=0.2, random_state=int(args.seed))

    model = build_and_compile_wellbeing_dnn()

    y_train: Dict[str, np.ndarray] = {k: data.y_train[k] for k in OUTPUT_NAMES}
    y_val: Dict[str, np.ndarray] = {k: data.y_test[k] for k in OUTPUT_NAMES}

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        data.X_train,
        y_train,
        validation_data=(data.X_test, y_val),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        verbose=1,
        callbacks=[early_stop],
    )

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    plots_dir = Path(args.plots_dir)
    plot_training_loss(history, plots_dir / "training_loss.png")
    plot_training_accuracy(history, plots_dir / "training_accuracy.png")

    save_training_log(history, DEFAULT_TRAINING_LOG)

    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved plots: {(plots_dir / 'training_loss.png').resolve()}")
    print(f"             {(plots_dir / 'training_accuracy.png').resolve()}")
    print(f"Saved training log: {DEFAULT_TRAINING_LOG.resolve()}")


if __name__ == "__main__":
    main()
