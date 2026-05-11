"""
Rule-based wellbeing recommendations driven by multi-output model predictions.

Maps each head's softmax probabilities to a class label using the **same** OneHotEncoder
category order as training, then applies **separate** rule sets for stress, energy, and
productivity.

Run from ``Digital_Wellbeing_DNN``::

    python utils/recommendations.py --row 0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import OneHotEncoder

LABELS = ["low", "medium", "high"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing import PreprocessingArtifacts, preprocess_from_csv  # noqa: E402
from models.dnn_model import OUTPUT_NAMES  # noqa: E402
from utils.evaluation import load_model  # noqa: E402


def softmax_row_to_label(probs_1d: np.ndarray, encoder: OneHotEncoder) -> str:
    """Argmax on softmax probabilities → human-readable class from encoder categories."""
    idx = int(np.argmax(np.asarray(probs_1d).reshape(-1)))
    cats = encoder.categories_[0]
    if idx < 0 or idx >= len(cats):
        raise IndexError(f"Predicted index {idx} out of range for categories {list(cats)}.")
    return str(cats[idx])


def stress_advice(level: str) -> List[str]:
    """Rule-based suggestions for predicted stress level (handled independently)."""
    level_n = level.strip().lower()
    if level_n == "high":
        return [
            "Stress looks elevated: prioritize reducing workload intensity this week (fewer parallel deadlines, delegate where possible).",
            "Protect sleep (aim for a stable bedtime/wake time) and schedule short recovery breaks between focused blocks.",
            "Try brief grounding or breathing exercises when workload spikes; reduce late-night screen use if possible.",
        ]
    if level_n == "medium":
        return [
            "Stress looks moderate: keep workload predictable—batch similar tasks and avoid frequent context switching.",
            "Maintain consistent sleep and add one planned offline wind-down block before bed.",
        ]
    if level_n == "low":
        return [
            "Stress looks relatively low: keep the routines that are working; avoid gradually increasing overload without recovery.",
        ]
    return [f"Stress level '{level}' is unclear; review habits manually and consider collecting more consistent logs."]


def energy_advice(level: str) -> List[str]:
    """Rule-based suggestions for predicted energy level (handled independently)."""
    level_n = level.strip().lower()
    if level_n == "low":
        return [
            "Energy looks low: gradually increase daily movement (e.g., two brisk walks) and stabilize sleep duration.",
            "Hydrate consistently and reduce reliance on late caffeine; bright morning light can help daytime alertness.",
            "Break work into smaller chunks with scheduled recovery to avoid an all-or-nothing crash cycle.",
        ]
    if level_n == "medium":
        return [
            "Energy looks moderate: keep activity steady week-to-week and avoid large sudden sleep cuts.",
        ]
    if level_n == "high":
        return [
            "Energy looks strong: sustain healthy limits—schedule rest even on productive days to prevent burnout.",
        ]
    return [f"Energy level '{level}' is unclear; review sleep/activity patterns over several days."]


def productivity_advice(level: str) -> List[str]:
    """Rule-based suggestions for predicted productivity level (handled independently)."""
    level_n = level.strip().lower()
    if level_n == "low":
        return [
            "Productivity looks constrained: reduce distractions (notifications), use focused time blocks (e.g., Pomodoro), and clarify top 1–3 tasks.",
            "Pair harder tasks with your naturally alert hours; keep breaks intentional rather than reactive scrolling.",
        ]
    if level_n == "medium":
        return [
            "Productivity looks steady: refine priorities weekly and protect deep-work windows on your calendar.",
        ]
    if level_n == "high":
        return [
            "Productivity looks strong: lock in sustainable pacing—avoid stacking overtime that raises stress and lowers recovery.",
        ]
    return [f"Productivity level '{level}' is unclear; inspect workload spikes vs. deep-work time."]


_ADVICE_DISPATCH = {
    "stress": stress_advice,
    "energy": energy_advice,
    "productivity": productivity_advice,
}


@dataclass
class RecommendationBundle:
    """Decoded predictions plus per-head advice (not merged into one score)."""

    labels: Dict[str, str]
    advice: Dict[str, List[str]] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        for head in OUTPUT_NAMES:
            lbl = self.labels[head]
            conf = self.confidence.get(head, float("nan"))
            lines.append(f"{head.capitalize()}: {lbl} (confidence ≈ {conf:.2f})")
        return lines


def recommendations_from_model_dict(
    pred_dict: Dict[str, np.ndarray],
    artifacts: PreprocessingArtifacts,
    row_index: int = 0,
) -> RecommendationBundle:
    """
    Turn Keras multi-output predictions into labels + human-readable guidance.

    Args:
        pred_dict: Output of ``model.predict(...)`` — dict[str, ndarray] with shape (batch, 3)
                   softmax probabilities per head.
        artifacts: Preprocessing bundle containing per-head ``OneHotEncoder`` instances.
        row_index: Which batch row to interpret.

    Returns:
        RecommendationBundle with separate advice lists per head.
    """
    labels: Dict[str, str] = {}
    confidence: Dict[str, float] = {}
    advice: Dict[str, List[str]] = {}

    encoders = artifacts.target_encoders
    for head in OUTPUT_NAMES:
        if head not in pred_dict:
            raise KeyError(f"Missing predictions for '{head}'. Got keys: {list(pred_dict.keys())}")
        if head not in encoders:
            raise KeyError(f"Missing encoder for '{head}'.")

        row = np.asarray(pred_dict[head])[row_index]
        labels[head] = softmax_row_to_label(row, encoders[head])
        confidence[head] = float(np.max(row))
        advice[head] = _ADVICE_DISPATCH[head](labels[head])
        if confidence[head] < 0.5:
            advice[head].append(
                "Prediction confidence is low — consider collecting more consistent or additional data."
            )

    if labels["stress"].lower() == "high" and labels["energy"].lower() == "low":
        advice["stress"].append(
            "Combined signal suggests possible burnout risk — consider reducing workload and prioritizing recovery."
        )

    return RecommendationBundle(labels=labels, advice=advice, confidence=confidence)


def format_bundle_as_text(bundle: RecommendationBundle) -> str:
    """Plain-text report suitable for CLI or logs."""
    lines: List[str] = []
    lines.append("Predictions:")
    lines.extend(f"  - {s}" for s in bundle.summary_lines())
    lines.append("")
    for head in OUTPUT_NAMES:
        lines.append(f"{head.upper()} — suggestions:")
        for item in bundle.advice[head]:
            lines.append(f"  • {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate rule-based recommendations from trained DNN predictions.")
    p.add_argument("--csv", type=str, default=str(_PROJECT_ROOT / "data" / "synthetic_v1.csv"))
    p.add_argument("--model", type=str, default=str(_PROJECT_ROOT / "models" / "wellbeing_dnn_trained.keras"))
    p.add_argument("--row", type=int, default=0, help="Index into X_test for demo prediction.")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    data = preprocess_from_csv(csv_path=Path(args.csv), random_state=int(args.seed))
    model = load_model(Path(args.model))

    i = int(args.row)
    if i < 0 or i >= len(data.X_test):
        raise SystemExit(f"--row must be in [0, {len(data.X_test) - 1}] for current split.")

    x = data.X_test[i : i + 1]
    preds = model.predict(x, verbose=0)
    if not isinstance(preds, dict):
        raise TypeError("Expected dict predictions from multi-output model.")

    bundle = recommendations_from_model_dict(preds, data.artifacts, row_index=0)
    print(format_bundle_as_text(bundle))


if __name__ == "__main__":
    main()
