"""Training and inference helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .data import FEATURES, TARGET


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if frame[FEATURES].isna().any().any() or frame[TARGET].isna().any():
        raise ValueError("Dataset contains missing feature or target values")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in FEATURES):
        raise ValueError("All sensor feature columns must be numeric")
    if frame[TARGET].nunique() < 2:
        raise ValueError("Training requires at least two fault classes")
    return frame


def train_model(data_path: str | Path, model_path: str | Path, seed: int = 42) -> dict:
    frame = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.2, random_state=seed, stratify=frame[TARGET]
    )
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, destination)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_),
        "labels": list(model.classes_),
        "model_path": str(destination),
    }


def predict(model_path: str | Path, readings: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    features = bundle["features"]
    missing = set(features) - set(readings.columns)
    if missing:
        raise ValueError(f"Readings are missing required columns: {', '.join(sorted(missing))}")
    values = readings[features]
    if values.isna().any().any() or not all(pd.api.types.is_numeric_dtype(values[c]) for c in features):
        raise ValueError("Sensor readings must be numeric and contain no missing values")
    model = bundle["model"]
    result = readings.copy()
    result["predicted_fault"] = model.predict(values)
    probabilities = model.predict_proba(values)
    result["confidence"] = probabilities.max(axis=1)
    return result
