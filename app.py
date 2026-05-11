"""
Streamlit web frontend for the Digital Wellbeing Assistant.

Reuses the *exact* preprocessing, model loading, and recommendation logic
already shipped with the CLI — no ML logic is reimplemented here.

Run from the ``Digital_Wellbeing_DNN`` directory::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    PreprocessingArtifacts,
    preprocess_from_csv,
)
from interface.cli import transform_raw_features  # noqa: E402
from models.dnn_model import OUTPUT_NAMES  # noqa: E402
from utils.evaluation import load_model  # noqa: E402
from utils.recommendations import recommendations_from_model_dict  # noqa: E402


DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "data" / "synthetic_v1.csv"
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "models" / "wellbeing_dnn_trained.keras"

FEATURE_INPUT_SPECS: Dict[str, Tuple[float, float, float, float]] = {
    "sleep_hours":              (0.0,  24.0, 7.0, 0.5),
    "workload_hours":           (0.0,  24.0, 8.0, 0.5),
    "physical_activity_min":    (0.0, 300.0, 30.0, 5.0),
    "screen_time_hours":        (0.0,  24.0, 2.0, 0.5),
    "social_interaction_hours": (0.0,  24.0, 1.5, 0.5),
    "water_intake_liters":      (0.0,   5.0, 2.0, 0.1),
    "caffeine_mg":              (0.0, 500.0, 100.0, 10.0),
    "mood_score":               (1.0,  10.0, 7.0, 1.0),
    "breaks_count":             (0.0,  20.0, 3.0, 1.0),
}

FEATURE_DISPLAY_LABELS: Dict[str, str] = {
    "sleep_hours":              "Sleep Hours (avg/night)",
    "workload_hours":           "Workload Hours (per day)",
    "physical_activity_min":    "Physical Activity (min/day)",
    "screen_time_hours":        "Screen Time (hours/day)",
    "social_interaction_hours": "Social Interaction (hours/day)",
    "water_intake_liters":      "Water Intake (L/day)",
    "caffeine_mg":              "Caffeine Intake (mg/day)",
    "mood_score":               "Mood Score (1\u201310)",
    "breaks_count":             "Breaks Count (per day)",
}


@st.cache_resource(show_spinner="Fitting preprocessing pipeline…")
def get_preprocessing_artifacts(
    csv_path: str, seed: int = 42, test_size: float = 0.2
) -> PreprocessingArtifacts:
    """Fit (once) the same imputer/scaler/encoders the model was trained with."""
    data = preprocess_from_csv(
        csv_path=Path(csv_path),
        test_size=test_size,
        random_state=seed,
    )
    return data.artifacts


@st.cache_resource(show_spinner="Loading trained model…")
def get_model(model_path: str):
    return load_model(Path(model_path))


def collect_user_inputs() -> np.ndarray:
    """Render input widgets and return raw feature row in FEATURE_COLUMNS order."""
    values = []
    cols = st.columns(2)
    for i, name in enumerate(FEATURE_COLUMNS):
        lo, hi, default, step = FEATURE_INPUT_SPECS[name]
        with cols[i % 2]:
            v = st.slider(
                label=FEATURE_DISPLAY_LABELS.get(name, name),
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
            )
        values.append(float(v))
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def render_predictions(pred_dict: Dict[str, np.ndarray], artifacts: PreprocessingArtifacts) -> None:
    cols = st.columns(len(OUTPUT_NAMES))
    for col, head in zip(cols, OUTPUT_NAMES):
        probs = np.asarray(pred_dict[head]).reshape(-1)
        idx = int(np.argmax(probs))
        label = str(artifacts.target_encoders[head].categories_[0][idx])
        confidence = float(probs[idx])
        with col:
            st.metric(label=head.capitalize(), value=label, delta=f"{confidence:.1%} confidence")
            cats = list(artifacts.target_encoders[head].categories_[0])
            st.bar_chart({c: [float(p)] for c, p in zip(cats, probs)})


def render_recommendations(pred_dict: Dict[str, np.ndarray], artifacts: PreprocessingArtifacts) -> None:
    bundle = recommendations_from_model_dict(pred_dict, artifacts, row_index=0)
    for head in OUTPUT_NAMES:
        st.markdown(f"**{head.upper()}** — {bundle.labels[head]} (confidence ≈ {bundle.confidence[head]:.2f})")
        for tip in bundle.advice[head]:
            st.markdown(f"- {tip}")
        st.write("")


def main() -> None:
    st.set_page_config(page_title="Digital Wellbeing Assistant", page_icon=None, layout="wide")
    st.title("Digital Wellbeing Assistant")
    st.caption("Predicts stress, energy, and productivity from daily lifestyle signals, then suggests next steps.")

    with st.sidebar:
        st.header("Configuration")
        csv_path = st.text_input("Dataset CSV (used to fit preprocessing)", value=str(DEFAULT_CSV_PATH))
        model_path = st.text_input("Trained model (.keras)", value=str(DEFAULT_MODEL_PATH))
        seed = st.number_input("Random seed (must match training)", value=42, step=1)

    st.header("Input")
    X_raw = collect_user_inputs()
    predict_clicked = st.button("Predict", type="primary")

    if not predict_clicked:
        st.info("Adjust the inputs above, then click **Predict** to see results.")
        return

    try:
        artifacts = get_preprocessing_artifacts(csv_path=csv_path, seed=int(seed))
        model = get_model(model_path=model_path)
    except Exception as exc:
        st.error(f"Failed to load preprocessing/model: {exc}")
        return

    X_scaled = transform_raw_features(X_raw, artifacts)
    preds = model.predict(X_scaled, verbose=0)
    if not isinstance(preds, dict):
        st.error("Model did not return dict-style predictions; check the saved multi-output model.")
        return

    st.header("Predictions")
    render_predictions(preds, artifacts)

    st.header("Recommendations")
    render_recommendations(preds, artifacts)


if __name__ == "__main__":
    main()
