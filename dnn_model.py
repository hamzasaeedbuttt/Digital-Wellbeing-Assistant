"""
Multi-output Deep Neural Network for Digital Wellbeing prediction.

Architecture (Keras Functional API):
- Input: 9 features
- Shared trunk: Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU)
- Three separate heads (stress, energy, productivity): each Dense(3, softmax)

Each output uses its own categorical cross-entropy loss (compiled as a dict of losses).
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Tuple

from tensorflow import keras
from tensorflow.keras import layers

INPUT_DIM: int = 9
NUM_CLASSES: int = 3
OUTPUT_NAMES: Tuple[str, ...] = ("stress", "energy", "productivity")


def build_wellbeing_dnn(
    input_dim: int = INPUT_DIM,
    num_classes: int = NUM_CLASSES,
    name: str = "digital_wellbeing_multi_output_dnn",
) -> keras.Model:
    """
    Build a multi-output model with shared hidden layers and separate softmax heads.

    Args:
        input_dim: Number of input features (default 9).
        num_classes: Classes per output head (default 3).
        name: Keras model name.

    Returns:
        Functional API Model mapping one input tensor to three named softmax outputs.
    """
    # Input features are expected to be preprocessed (normalized/scaled)
    # using StandardScaler or similar, before being passed into the model.
    inputs = keras.Input(shape=(input_dim,), name="features")

    # Shared representation (single trunk for all tasks).
    x = layers.Dense(64, activation="relu", name="hidden_64")(inputs)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.Dense(32, activation="relu", name="hidden_32")(x)
    shared = layers.Dense(16, activation="relu", name="hidden_16")(x)

    # Separate heads: each task gets its own weights and softmax over classes.
    stress_out = layers.Dense(num_classes, activation="softmax", name="stress")(shared)
    energy_out = layers.Dense(num_classes, activation="softmax", name="energy")(shared)
    productivity_out = layers.Dense(num_classes, activation="softmax", name="productivity")(shared)

    return keras.Model(
        inputs=inputs,
        outputs={
            "stress": stress_out,
            "energy": energy_out,
            "productivity": productivity_out,
        },
        name=name,
    )


def compile_wellbeing_model(
    model: keras.Model,
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    loss_weights: Optional[Dict[str, float]] = None,
) -> keras.Model:
    """
    Compile with Adam and separate categorical cross-entropy per output head.

    Each output has its own loss entry so gradients combine according to `loss_weights`
    (defaults to equal weight per head).

    Args:
        model: Model from build_wellbeing_dnn().
        optimizer: Defaults to Adam with learning_rate=0.001.
        loss_weights: Optional per-output weights for the total loss.

    Returns:
        The same model instance (compiled in-place).
    """
    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

    losses = {head: "categorical_crossentropy" for head in OUTPUT_NAMES}
    metrics = {head: ["accuracy"] for head in OUTPUT_NAMES}

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model


def build_and_compile_wellbeing_dnn(
    input_dim: int = INPUT_DIM,
    num_classes: int = NUM_CLASSES,
    loss_weights: Optional[Dict[str, float]] = None,
) -> keras.Model:
    """Convenience: build + compile in one call."""
    model = build_wellbeing_dnn(input_dim=input_dim, num_classes=num_classes)
    return compile_wellbeing_model(model, loss_weights=loss_weights)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and summarize the Digital Wellbeing multi-output DNN.")
    p.add_argument("--input-dim", type=int, default=INPUT_DIM, help="Input feature dimension (default: 9).")
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Classes per head (default: 3).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    model = build_and_compile_wellbeing_dnn(
        input_dim=int(args.input_dim),
        num_classes=int(args.num_classes),
    )
    model.summary()


if __name__ == "__main__":
    main()
