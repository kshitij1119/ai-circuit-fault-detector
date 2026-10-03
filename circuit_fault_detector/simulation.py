"""LTspice batch simulation and waveform dataset generation."""

from __future__ import annotations

import csv
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PyLTSpice import LTspice, RawRead, SimRunner

from .faults import FAULTS, Fault, fault_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_FILE = PROJECT_ROOT / "circuits" / "sallen_key_bandpass.inc"
def find_ltspice(explicit: str | Path | None = None) -> str | Path:
    """Resolve an LTspice executable from an argument, environment, or PATH."""
    import os

    configured = explicit or os.environ.get("LTSPICE_EXE")
    candidates = [Path(configured)] if configured else []
    if not configured:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "Programs" / "ADI" / "LTspice" / "LTspice.exe")
        candidates.append(Path(r"C:\Program Files\ADI\LTspice\LTspice.exe"))
        on_path = shutil.which("LTspice") or shutil.which("LTspice.exe")
        if on_path:
            return on_path
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if configured:
        raise FileNotFoundError(f"LTspice executable not found: {configured}")
    # Use PyLTSpice's platform discovery when it is installed in a standard location.
    return LTspice


def render_netlist(fault: Fault, rng: random.Random) -> str:
    """Render one transient deck, applying ±10% tolerance to non-faulted parts."""
    values = {
        "RHP": 10_000.0 * rng.uniform(0.9, 1.1),
        "CHP": 100e-9 * rng.uniform(0.9, 1.1),
        "R1V": fault.r1 if fault.code == "F1" else 10_000.0 * rng.uniform(0.9, 1.1),
        "R2V": fault.r2 if fault.code == "F2" else 10_000.0 * rng.uniform(0.9, 1.1),
        "C1V": fault.c1 if fault.code == "F3" else 10e-9 * rng.uniform(0.9, 1.1),
        "C2V": fault.c2 if fault.code == "F4" else 10e-9 * rng.uniform(0.9, 1.1),
        "RC2SHORT": fault.c2_short_r,
        "AOL": fault.opamp_gain,
        "ROUT": fault.output_resistance,
    }
    body = INCLUDE_FILE.read_text(encoding="utf-8")
    overrides = ".param " + " ".join(f"{key}={value:.12g}" for key, value in values.items())
    # LTspice rejects duplicate .param definitions, so replace the nominal
    # parameter declarations in this rendered deck with the randomized values.
    lines = [line for line in body.splitlines() if not line.lstrip().lower().startswith(".param")]
    lines.insert(2, overrides)
    return "\n".join([
        f"* {fault.code} {fault.name} circuit fault simulation",
        *lines,
        ".tran 0 50m 0 10u",
        ".end",
        "",
    ])


def _read_waveform(raw_path: str | Path) -> pd.DataFrame:
    raw = RawRead(str(raw_path), traces_to_read=["V(in)", "V(out)"], verbose=False)
    time_s = np.asarray(raw.get_axis(), dtype=float)
    input_v = np.asarray(raw.get_trace("V(in)").get_wave(0), dtype=float)
    output_v = np.asarray(raw.get_trace("V(out)").get_wave(0), dtype=float)
    if not (len(time_s) == len(input_v) == len(output_v)):
        raise ValueError(f"Mismatched traces in LTspice output {raw_path}")
    return pd.DataFrame({"time_s": time_s, "input_v": input_v, "output_v": output_v})


def generate_waveforms(
    output_dir: str | Path,
    samples_per_class: int = 200,
    seed: int = 42,
    ltspice: str | Path | None = None,
    limit: int | None = None,
) -> Path:
    """Run six fault classes and write waveform CSVs plus a manifest.

    Default generation is 6 × 200 = 1,200 LTspice runs. Pass ``limit`` for a small
    smoke run, or use ``--samples-per-class`` to control dataset size.
    """
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")
    output = Path(output_dir)
    wave_dir = output / "waveforms"
    run_dir = output / "ltspice_runs"
    wave_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    simulator = find_ltspice(ltspice)
    runner = SimRunner(output_folder=str(run_dir), simulator=simulator, parallel_sims=1, timeout=180)
    rng = random.Random(seed)
    manifest_rows = []
    count = 0
    target_count = samples_per_class * len(FAULTS)
    for fault in FAULTS:
        label = fault_label(fault.code)
        class_dir = wave_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(samples_per_class):
            if limit is not None and count >= limit:
                break
            sample_id = f"{fault.code}_{sample_index:04d}"
            deck_path = run_dir / f"{sample_id}.cir"
            deck_path.write_text(render_netlist(fault, rng), encoding="utf-8")
            raw_file, log_file = runner.run_now(deck_path, run_filename=deck_path.stem, timeout=180)
            if not raw_file or not Path(raw_file).is_file():
                log_hint = Path(log_file).read_text(errors="replace")[-2000:] if log_file and Path(log_file).is_file() else "No LTspice log was produced."
                raise RuntimeError(f"LTspice failed for {sample_id}:\n{log_hint}")
            waveform = _read_waveform(raw_file)
            csv_path = class_dir / f"{sample_id}.csv"
            waveform.to_csv(csv_path, index=False, float_format="%.10g")
            manifest_rows.append({
                "sample_id": sample_id,
                "fault_code": fault.code,
                "fault_label": label,
                "waveform_path": csv_path.relative_to(output).as_posix(),
                "points": len(waveform),
                "seed": seed,
            })
            count += 1
            print(f"[{count}/{min(target_count, limit) if limit else target_count}] {sample_id}")
        if limit is not None and count >= limit:
            break
    manifest = output / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LTspice simulations for the six circuit fault classes.")
    parser.add_argument("--output", default="data/ltspice")
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ltspice", help="LTspice executable path (or set LTSPICE_EXE)")
    parser.add_argument("--limit", type=int, help="run only this many simulations for a smoke check")
    args = parser.parse_args()
    manifest = generate_waveforms(args.output, args.samples_per_class, args.seed, args.ltspice, args.limit)
    print(f"Waveform manifest written to {manifest}")

