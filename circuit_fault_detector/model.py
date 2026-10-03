"""Random Forest training, evaluation, and inference for LTspice features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS

LABEL_COLUMN = "fault_label"


def validate_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*FEATURE_COLUMNS, LABEL_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {', '.join(sorted(missing))}")
    if frame[list(FEATURE_COLUMNS)].isna().any().any() or frame[LABEL_COLUMN].isna().any():
        raise ValueError("Feature table contains missing feature or label values")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in FEATURE_COLUMNS):
        raise ValueError("All eight waveform feature columns must be numeric")
    if frame[LABEL_COLUMN].nunique() < 2:
        raise ValueError("Training requires at least two fault classes")
    counts = frame[LABEL_COLUMN].value_counts()
    if (counts < 2).any():
        raise ValueError("Each fault class needs at least two examples for a stratified split")
    return frame


def train_random_forest(
    feature_path: str | Path,
    model_path: str | Path = "models/random_forest.joblib",
    report_dir: str | Path = "reports",
    seed: int = 42,
) -> dict:
    frame = validate_feature_table(pd.read_csv(feature_path))
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLUMNS], frame[LABEL_COLUMN], test_size=0.2, random_state=seed, stratify=frame[LABEL_COLUMN]
    )
    model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(frame[LABEL_COLUMN].unique())
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(FEATURE_COLUMNS), "labels": labels}, model_path)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "random_forest_metrics.json").write_text(
        json.dumps({"accuracy": accuracy, "classification_report": report, "labels": labels}, indent=2),
        encoding="utf-8",
    )
    _write_plots(model, y_test, predictions, labels, report_dir)
    return {"accuracy": accuracy, "report": report, "labels": labels, "model_path": model_path}


def _write_plots(model, y_test, predictions, labels, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="Random Forest — confusion matrix", xlabel="Predicted fault", ylabel="Actual fault")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "random_forest_confusion_matrix.png", dpi=160)
    plt.close(fig)

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    ax = importance.plot.barh(figsize=(8, 5), title="Random Forest — feature importance")
    ax.set_xlabel("Mean decrease in impurity")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "random_forest_feature_importance.png", dpi=160)
    plt.close(ax.figure)


def load_model(path: str | Path = "models/random_forest.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def predict_features(model, feature_row: pd.DataFrame) -> tuple[str, float, dict[str, float]]:
    values = feature_row.loc[:, FEATURE_COLUMNS]
    probabilities = model.predict_proba(values)[0]
    classes = list(model.classes_)
    best = int(np.argmax(probabilities))
    return str(classes[best]), float(probabilities[best]), {str(c): float(p) for c, p in zip(classes, probabilities)}
