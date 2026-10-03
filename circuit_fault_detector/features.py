"""Extract the eight project-plan features from a transient waveform."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_COLUMNS = [
    "peak_amplitude_v",
    "rms_v",
    "rise_time_s",
    "settling_time_s",
    "thd_percent",
    "zero_crossings",
    "std_dev_v",
    "skewness",
]


def _crossing_time(t: np.ndarray, y: np.ndarray, level: float) -> float:
    indices = np.flatnonzero((y[:-1] < level) & (y[1:] >= level))
    if not len(indices):
        return float("nan")
    i = int(indices[0])
    dy = y[i + 1] - y[i]
    if dy == 0:
        return float(t[i])
    return float(t[i] + (level - y[i]) * (t[i + 1] - t[i]) / dy)


def _thd_percent(t: np.ndarray, y: np.ndarray, fundamental_hz: float) -> float:
    if len(t) < 16 or fundamental_hz <= 0:
        return 0.0
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return 0.0
    uniform_t = np.linspace(float(t[0]), float(t[-1]), len(t))
    uniform_y = np.interp(uniform_t, t, y)
    uniform_y = uniform_y - np.mean(uniform_y)
    window = np.hanning(len(uniform_y))
    spectrum = np.abs(np.fft.rfft(uniform_y * window))
    frequencies = np.fft.rfftfreq(len(uniform_y), d=float(np.median(np.diff(uniform_t))))
    fundamental_bin = int(np.argmin(np.abs(frequencies - fundamental_hz)))
    if fundamental_bin == 0 or spectrum[fundamental_bin] <= np.finfo(float).eps:
        return 0.0
    fundamental = spectrum[fundamental_bin]
    harmonic_power = 0.0
    for harmonic in range(2, 6):
        index = fundamental_bin * harmonic
        if index < len(spectrum):
            harmonic_power += float(spectrum[index] ** 2)
    return float(100.0 * np.sqrt(harmonic_power) / fundamental)


def extract_features(time_s, voltage_v, fundamental_hz: float = 1000.0) -> dict[str, float]:
    """Return peak, RMS, rise/settling time, THD, zero crossings, std dev and skewness.

    Rise time is the first positive 10–90% crossing. Settling time is the start of
    the first full cycle after which cycle amplitudes remain within 5% of the final
    steady-state amplitude. Missing rise/settling crossings are NaN.
    """
    t = np.asarray(time_s, dtype=float).reshape(-1)
    y = np.asarray(voltage_v, dtype=float).reshape(-1)
    if len(t) != len(y) or len(t) < 8:
        raise ValueError("A waveform must contain matching time and voltage arrays with at least 8 samples")
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    order = np.argsort(t)
    t, y = t[order], y[order]
    unique = np.concatenate(([True], np.diff(t) > 0))
    t, y = t[unique], y[unique]
    if len(t) < 8:
        raise ValueError("Waveform must contain at least 8 distinct finite time samples")

    tail = y[int(len(y) * 0.8):]
    amplitude = float((np.max(tail) - np.min(tail)) / 2.0)
    mean_tail = float(np.mean(tail))
    first_period_end = float(t[0] + 1.5 / fundamental_hz)
    first = t <= first_period_end
    first_t, first_y = t[first], y[first]
    waveform_duration = float(t[-1] - t[0])
    rise_time = waveform_duration
    if len(first_t) >= 2 and amplitude > 0:
        low = mean_tail + 0.1 * amplitude
        high = mean_tail + 0.9 * amplitude
        low_t = _crossing_time(first_t, first_y, low)
        high_t = _crossing_time(first_t, first_y, high)
        if np.isfinite(low_t) and np.isfinite(high_t) and high_t >= low_t:
            rise_time = high_t - low_t

    period = 1.0 / fundamental_hz
    final_t = float(t[-1])
    first_cycle = max(0, int(np.ceil((t[0] - t[0]) / period)))
    final_cycle = int((final_t - t[0]) / period)
    cycle_amplitudes = []
    cycle_starts = []
    for cycle in range(first_cycle, final_cycle):
        start = float(t[0] + cycle * period)
        mask = (t >= start) & (t < start + period)
        if np.count_nonzero(mask) >= 4:
            values = y[mask]
            cycle_amplitudes.append(float((np.max(values) - np.min(values)) / 2.0))
            cycle_starts.append(start)
    settling_time = 0.0 if amplitude <= 1e-15 else waveform_duration
    if cycle_amplitudes and amplitude > 0:
        tolerance = max(0.05 * amplitude, 1e-12)
        for i, start in enumerate(cycle_starts):
            if all(abs(value - amplitude) <= tolerance for value in cycle_amplitudes[i:]):
                settling_time = max(0.0, start - float(t[0]))
                break

    centered = y - float(np.mean(y))
    std = float(np.std(centered))
    skewness = float(np.mean(centered ** 3) / (std ** 3)) if std > 1e-15 else 0.0
    crossings = int(np.count_nonzero(np.diff(np.signbit(centered))))
    return {
        "peak_amplitude_v": amplitude,
        "rms_v": float(np.sqrt(np.mean(y ** 2))),
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "thd_percent": _thd_percent(t, y, fundamental_hz),
        "zero_crossings": crossings,
        "std_dev_v": std,
        "skewness": skewness,
    }


def extract_manifest(manifest_path: str | Path, output_path: str | Path, fundamental_hz: float = 1000.0) -> Path:
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required = {"sample_id", "fault_code", "fault_label", "waveform_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    rows = []
    for entry in manifest.itertuples(index=False):
        waveform_path = manifest_path.parent / entry.waveform_path
        if not waveform_path.is_file():
            raise FileNotFoundError(f"Waveform file listed by the manifest is missing: {waveform_path}")
        waveform = pd.read_csv(waveform_path)
        if not {"time_s", "output_v"}.issubset(waveform.columns):
            raise ValueError(f"Waveform requires time_s and output_v columns: {waveform_path}")
        features = extract_features(waveform["time_s"], waveform["output_v"], fundamental_hz)
        rows.append({"sample_id": entry.sample_id, "fault_code": entry.fault_code,
                     "fault_label": entry.fault_label, **features})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["sample_id", "fault_code", "fault_label", *FEATURE_COLUMNS]).to_csv(output, index=False)
    return output
