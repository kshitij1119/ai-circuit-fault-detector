"""Create a clearly labeled synthetic dataset for demos and reproducibility."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

FEATURES = ["supply_voltage_v", "load_current_a", "resistance_ohm", "temperature_c"]
TARGET = "fault_type"
CLASSES = ["normal", "open_circuit", "short_circuit", "overload", "component_drift"]


def generate_dataset(rows_per_class: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate illustrative readings; these are not measurements from hardware."""
    if rows_per_class < 2:
        raise ValueError("rows_per_class must be at least 2")
    rng = random.Random(seed)
    rows = []
    for label in CLASSES:
        for _ in range(rows_per_class):
            voltage = rng.gauss(12.0, 0.35)
            resistance = rng.gauss(100.0, 5.0)
            current = voltage / resistance
            temperature = rng.gauss(32.0, 2.5)
            if label == "open_circuit":
                current = abs(rng.gauss(0.004, 0.003))
                resistance = abs(rng.gauss(1800, 500))
            elif label == "short_circuit":
                resistance = abs(rng.gauss(2.0, 1.0))
                current = abs(rng.gauss(3.0, 0.45))
                temperature = rng.gauss(52.0, 6.0)
            elif label == "overload":
                current = abs(rng.gauss(0.42, 0.08))
                temperature = rng.gauss(68.0, 9.0)
            elif label == "component_drift":
                resistance = abs(rng.gauss(145.0, 12.0))
                current = voltage / max(resistance, 1.0)
                temperature = rng.gauss(38.0, 4.0)
            else:
                current = abs(rng.gauss(0.12, 0.015))
            rows.append({
                "supply_voltage_v": round(voltage, 4),
                "load_current_a": round(current, 5),
                "resistance_ohm": round(resistance, 3),
                "temperature_c": round(temperature, 2),
                "fault_type": label,
            })
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def save_dataset(path: str | Path, rows_per_class: int = 200, seed: int = 42) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(rows_per_class, seed).to_csv(destination, index=False)
    return destination
