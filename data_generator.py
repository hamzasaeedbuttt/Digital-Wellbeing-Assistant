"""
Synthetic dataset generator for a Digital Wellbeing Assistant project.

This module generates a balanced, rule-based dataset with:
- 9 numeric input features
- 3 categorical outputs (stress, energy, productivity), each with 3 classes

Note on feature choice:
- The feature set is slightly enhanced compared to a minimal "wellbeing" proposal (e.g., adding `mood_score`,
  hydration via `water_intake_liters`, and `breaks_count`). These additions improve realism and provide
  richer signals for the model to learn from.

Rules (soft, with randomness):
- Low sleep + high workload -> higher stress
- High activity + good sleep -> higher energy
- Productivity tends to be higher with lower stress, adequate energy, moderate workload,
  reasonable screen time, and regular breaks

The generator enforces class balance by sampling evenly across all 27 combinations
of (stress, energy, productivity) labels, then generating features conditioned on labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LABELS: Tuple[str, str, str] = ("low", "medium", "high")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    min_value: float
    max_value: float


FEATURE_SPECS: Tuple[FeatureSpec, ...] = (
    FeatureSpec("sleep_hours", 3.0, 10.0),
    FeatureSpec("workload_hours", 0.0, 14.0),
    FeatureSpec("physical_activity_min", 0.0, 180.0),
    FeatureSpec("screen_time_hours", 0.0, 14.0),
    FeatureSpec("social_interaction_hours", 0.0, 8.0),
    FeatureSpec("water_intake_liters", 0.5, 4.0),
    FeatureSpec("caffeine_mg", 0.0, 600.0),
    FeatureSpec("mood_score", 1.0, 10.0),
    FeatureSpec("breaks_count", 0.0, 20.0),
)


def _clip_to_spec(value: float, spec: FeatureSpec) -> float:
    return float(np.clip(value, spec.min_value, spec.max_value))


def _triangular(rng: np.random.Generator, low: float, mode: float, high: float) -> float:
    return float(rng.triangular(left=low, mode=mode, right=high))


def _label_index(label: str) -> int:
    try:
        return LABELS.index(label)
    except ValueError as exc:
        raise ValueError(f"Invalid label '{label}'. Expected one of {LABELS}.") from exc


def _base_features_for_labels(
    rng: np.random.Generator, stress: str, energy: str, productivity: str
) -> Dict[str, float]:
    """
    Generate a feature vector conditioned on the requested labels.

    The generator is intentionally redundant: multiple features contribute to each target,
    and noise is added to avoid perfectly separable classes.
    """
    s = _label_index(stress)  # 0 low, 1 medium, 2 high
    e = _label_index(energy)
    p = _label_index(productivity)

    # Sleep: strongly inversely related to stress, positively to energy.
    sleep_mode = {0: 8.2, 1: 6.8, 2: 5.2}[s]
    sleep_shift = {0: 0.6, 1: 0.0, 2: -0.6}[e]
    sleep_hours = _triangular(rng, 3.0, sleep_mode + sleep_shift, 10.0) + rng.normal(0.0, 0.25)

    # Workload: higher workload tends to increase stress; productivity prefers moderate workload.
    workload_mode_by_stress = {0: 4.5, 1: 7.0, 2: 10.0}[s]
    workload_adjust_for_prod = {0: -1.0, 1: 0.0, 2: -0.5}[p]  # very high productivity: not extreme
    workload_hours = (
        _triangular(rng, 0.0, workload_mode_by_stress + workload_adjust_for_prod, 14.0)
        + rng.normal(0.0, 0.6)
    )

    # Physical activity: boosts energy, reduces stress (slightly), helps productivity indirectly.
    activity_mode_by_energy = {0: 20.0, 1: 55.0, 2: 95.0}[e]
    activity_adjust_for_stress = {0: 10.0, 1: 0.0, 2: -10.0}[s]
    physical_activity_min = (
        _triangular(rng, 0.0, activity_mode_by_energy + activity_adjust_for_stress, 180.0)
        + rng.normal(0.0, 8.0)
    )

    # Screen time: tends to be higher for lower productivity and higher stress.
    screen_mode = 5.0 + 1.2 * s - 1.0 * p
    screen_time_hours = _triangular(rng, 0.0, float(np.clip(screen_mode, 1.0, 12.0)), 14.0) + rng.normal(
        0.0, 0.6
    )

    # Social interaction: can improve mood and reduce stress; too low may correlate with stress.
    social_mode = 1.5 + 0.9 * (2 - s) + 0.4 * e
    social_interaction_hours = _triangular(rng, 0.0, float(np.clip(social_mode, 0.5, 6.5)), 8.0) + rng.normal(
        0.0, 0.35
    )

    # Hydration: mild correlation with energy and productivity.
    water_mode = 1.6 + 0.35 * e + 0.2 * p
    water_intake_liters = _triangular(rng, 0.5, float(np.clip(water_mode, 0.8, 3.5)), 4.0) + rng.normal(
        0.0, 0.15
    )

    # Caffeine: tends to be higher when energy is low or stress is high; helps productivity slightly at medium.
    caffeine_mode = 120.0 + 90.0 * s + 110.0 * (2 - e) + (40.0 if p == 1 else 0.0)
    caffeine_mg = _triangular(rng, 0.0, float(np.clip(caffeine_mode, 40.0, 520.0)), 600.0) + rng.normal(
        0.0, 35.0
    )

    # Breaks: correlate with higher productivity and lower stress.
    breaks_mode = 4.0 + 3.0 * p + 2.0 * (2 - s)
    breaks_count = _triangular(rng, 0.0, float(np.clip(breaks_mode, 1.0, 16.0)), 20.0) + rng.normal(0.0, 1.0)

    # Mood: driven by sleep, stress, energy, social time; add randomness.
    mood_score = (
        5.0
        + 0.4 * (sleep_hours - 6.5)
        - 0.9 * s
        + 0.7 * e
        + 0.25 * (social_interaction_hours - 2.0)
        + rng.normal(0.0, 0.9)
    )

    values: Dict[str, float] = {
        "sleep_hours": sleep_hours,
        "workload_hours": workload_hours,
        "physical_activity_min": physical_activity_min,
        "screen_time_hours": screen_time_hours,
        "social_interaction_hours": social_interaction_hours,
        "water_intake_liters": water_intake_liters,
        "caffeine_mg": caffeine_mg,
        "mood_score": mood_score,
        "breaks_count": breaks_count,
    }

    # Clip all values to their declared specs.
    spec_map = {s_.name: s_ for s_ in FEATURE_SPECS}
    for k, v in list(values.items()):
        values[k] = _clip_to_spec(float(v), spec_map[k])

    return values


def _derive_targets_from_features(
    rng: np.random.Generator, features: Dict[str, float]
) -> Tuple[str, str, str]:
    """
    Apply the project's *rule-based* logic to derive labels from features.

    This is used only as a sanity check and to optionally re-label a small fraction
    of rows, keeping the dataset realistic.
    """
    sleep = features["sleep_hours"]
    workload = features["workload_hours"]
    activity = features["physical_activity_min"]
    screen = features["screen_time_hours"]
    breaks = features["breaks_count"]
    mood = features["mood_score"]
    caffeine = features["caffeine_mg"]

    # Stress score: higher with low sleep, high workload, high screen, low breaks.
    stress_score = (
        1.2 * (6.5 - sleep)
        + 0.9 * (workload - 7.0)
        + 0.35 * (screen - 5.0)
        - 0.25 * (breaks - 6.0)
        - 0.25 * (mood - 5.5)
        + 0.15 * (caffeine - 180.0) / 100.0
        + rng.normal(0.0, 0.6)
    )

    # Energy score: higher with sleep and activity; too much workload reduces energy.
    energy_score = (
        0.9 * (sleep - 6.5)
        + 0.012 * (activity - 50.0)
        - 0.25 * (workload - 7.0)
        - 0.12 * (screen - 5.0)
        + 0.18 * (mood - 5.5)
        + rng.normal(0.0, 0.55)
    )

    # Productivity score: prefers moderate workload, lower stress, adequate energy, breaks.
    productivity_score = (
        -0.22 * abs(workload - 7.0)
        - 0.45 * stress_score
        + 0.55 * energy_score
        - 0.10 * (screen - 5.0)
        + 0.10 * (breaks - 6.0)
        + 0.10 * (mood - 5.5)
        + rng.normal(0.0, 0.5)
    )

    def bucket(score: float) -> str:
        if score <= -0.6:
            return "low"
        if score <= 0.6:
            return "medium"
        return "high"

    # Higher `stress_score` means higher stress, and `bucket()` maps score ranges to {low, medium, high}.
    stress_label = bucket(stress_score)
    energy_label = bucket(energy_score)
    productivity_label = bucket(productivity_score)

    return stress_label, energy_label, productivity_label


def generate_dataset(
    n_samples: int,
    seed: int,
    missing_rate: float = 0.02,
    relabel_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Generate a balanced synthetic dataset.

    Balance strategy:
    - Sample uniformly across all 27 label combinations
    - For each combination, generate features conditioned on the labels
    - Optionally relabel a small fraction using rule-based derivation for realism

    Args:
        n_samples: total rows to generate (1000..5000 recommended).
        seed: RNG seed.
        missing_rate: fraction of feature cells set to NaN to simulate missingness.
        relabel_rate: fraction of rows whose labels are replaced by derived rule labels.

    Returns:
        DataFrame with 9 feature columns and 3 target columns:
        ['stress', 'energy', 'productivity'] as string labels in {'low','medium','high'}.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if not (0.0 <= missing_rate < 0.2):
        raise ValueError("missing_rate must be in [0.0, 0.2).")
    if not (0.0 <= relabel_rate < 0.5):
        raise ValueError("relabel_rate must be in [0.0, 0.5).")

    rng = np.random.default_rng(seed)

    combos: List[Tuple[str, str, str]] = [(s, e, p) for s in LABELS for e in LABELS for p in LABELS]
    per_combo = n_samples // len(combos)
    remainder = n_samples % len(combos)

    rows: List[Dict[str, float]] = []
    targets: List[Tuple[str, str, str]] = []

    # Generate balanced rows per label-combination.
    for combo_idx, (s, e, p) in enumerate(combos):
        count = per_combo + (1 if combo_idx < remainder else 0)
        for _ in range(count):
            feat = _base_features_for_labels(rng, stress=s, energy=e, productivity=p)
            rows.append(feat)
            targets.append((s, e, p))

    df = pd.DataFrame(rows)
    df["stress"], df["energy"], df["productivity"] = zip(*targets)

    # Introduce missing values in features only.
    if missing_rate > 0.0:
        feature_cols = [fs.name for fs in FEATURE_SPECS]
        mask = rng.random((len(df), len(feature_cols))) < missing_rate
        for j, col in enumerate(feature_cols):
            df.loc[mask[:, j], col] = np.nan

    # Relabel a small fraction using rule-based labels to avoid overly synthetic conditioning.
    if relabel_rate > 0.0:
        relabel_mask = rng.random(len(df)) < relabel_rate
        idxs = np.where(relabel_mask)[0]
        for i in idxs:
            feat = {k: float(df.at[i, k]) for k in [fs.name for fs in FEATURE_SPECS]}
            # If NaNs exist, skip relabeling this row.
            if any(np.isnan(v) for v in feat.values()):
                continue
            s2, e2, p2 = _derive_targets_from_features(rng, feat)
            df.at[i, "stress"] = s2
            df.at[i, "energy"] = e2
            df.at[i, "productivity"] = p2

    # Shuffle rows.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Ensure target labels are valid.
    for col in ("stress", "energy", "productivity"):
        if not set(df[col].unique()).issubset(set(LABELS)):
            raise RuntimeError(f"Unexpected labels in column '{col}'.")

    return df


def _class_balance_report(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    report: Dict[str, Dict[str, int]] = {}
    for col in ("stress", "energy", "productivity"):
        counts = df[col].value_counts().reindex(list(LABELS), fill_value=0)
        report[col] = {k: int(v) for k, v in counts.to_dict().items()}
    return report


def save_dataset_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a balanced synthetic digital wellbeing dataset.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3000,
        help="Number of samples to generate (recommended 1000..5000). Default: 3000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.02,
        help="Fraction of feature cells set to NaN. Default: 0.02.",
    )
    parser.add_argument(
        "--relabel-rate",
        type=float,
        default=0.05,
        help="Fraction of rows re-labeled using rule-based scoring. Default: 0.05.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("Digital_Wellbeing_DNN") / "data" / "synthetic_v1.csv"),
        help="Output CSV path (relative or absolute).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not (1000 <= args.n_samples <= 5000):
        raise SystemExit("n-samples must be between 1000 and 5000 (inclusive) as required.")

    df = generate_dataset(
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        missing_rate=float(args.missing_rate),
        relabel_rate=float(args.relabel_rate),
    )

    output_path = Path(args.output)
    save_dataset_csv(df, output_path=output_path)

    report = _class_balance_report(df)
    print(f"Saved dataset: {output_path.resolve()}")
    print("Class balance (counts):")
    for target, counts in report.items():
        print(f"- {target}: {counts}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

