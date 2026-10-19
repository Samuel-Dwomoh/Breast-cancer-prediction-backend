from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"

FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Save your trained model before starting the API."
        )

    return joblib.load(MODEL_PATH)


def predict(model, features: list[float]) -> str:
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, "
            f"but received {len(features)}."
        )

    input_df = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(input_df)[0]

    # LabelEncoder maps B -> 0 and M -> 1 for this dataset.
    diagnosis_labels = {0: "B", 1: "M"}

    return diagnosis_labels[int(prediction)]