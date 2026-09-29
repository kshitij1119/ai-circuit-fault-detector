# AI Circuit Fault Detector

A small Python machine-learning baseline for classifying circuit states from tabular sensor readings. It includes a reproducible synthetic-data generator, a Random Forest training pipeline, evaluation metrics, and batch CSV inference.

> **Safety and scope:** This is an educational prototype. The included data is synthetic and the model is not validated for real circuits, safety monitoring, or control decisions. Do not connect it to live equipment or rely on predictions to prevent harm. Validate any real deployment with representative measurements and qualified engineering review.

## Fault classes

The demonstration generator creates examples for `normal`, `open_circuit`, `short_circuit`, `overload`, and `component_drift`. Features are supply voltage (V), load current (A), resistance (Ω), and temperature (°C). These simplified distributions exist to exercise the software pipeline; they do not represent a particular circuit or sensor.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
circuit-fault generate-data
circuit-fault train
```

You can also invoke it without installing the package as a command:

```bash
python -m circuit_fault_detector.cli generate-data
python -m circuit_fault_detector.cli train
```

## Predict from your readings

Create a CSV with one row per observation and these numeric columns (order does not matter):

```csv
supply_voltage_v,load_current_a,resistance_ohm,temperature_c
12.1,0.12,100.8,31.4
```

Then run:

```bash
circuit-fault predict --input readings.csv --output predictions.csv
```

Output rows retain the input columns and add `predicted_fault` and `confidence`. Confidence is the model's maximum class probability, not a guarantee of correctness. The model file is created locally at `models/circuit_fault_model.joblib` and is not committed.

## Train on labeled measurements

Provide a CSV containing all four features plus a `fault_type` target column. Use labels that are consistent with your data collection protocol. Then run:

```bash
circuit-fault train --data path/to/labeled_readings.csv --model models/site_model.joblib
```

The command makes a stratified train/test split and prints accuracy and per-class precision, recall, and F1. Evaluate on data collected independently from training where possible; random row splits can overstate performance when measurements from the same circuit or session appear in both sets.

## Commands

- `generate-data --output PATH --rows-per-class N --seed N`: generate synthetic demo data.
- `train --data PATH --model PATH --seed N`: fit and evaluate a classifier.
- `predict --model PATH --input PATH --output PATH`: classify CSV readings.

## Repository layout

```text
circuit_fault_detector/  Python package and CLI
data/                    Dataset guidance; generated data is local
models/                  Local model output directory
```

## GitHub repository

This project is hosted at [github.com/kshitij1119/ai-circuit-fault-detector](https://github.com/kshitij1119/ai-circuit-fault-detector). Clone it with:

```bash
git clone https://github.com/kshitij1119/ai-circuit-fault-detector.git
cd ai-circuit-fault-detector
```

## License

MIT. See [LICENSE](LICENSE).

