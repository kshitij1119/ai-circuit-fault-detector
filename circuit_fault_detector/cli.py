"""Command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import save_dataset
from .model import predict, train_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="circuit-fault", description="Classify circuit faults from sensor readings.")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-data", help="create a synthetic demo dataset")
    generate.add_argument("--output", default="data/synthetic_circuit_readings.csv")
    generate.add_argument("--rows-per-class", type=int, default=200)
    generate.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="train a classifier and print evaluation metrics")
    train.add_argument("--data", default="data/synthetic_circuit_readings.csv")
    train.add_argument("--model", default="models/circuit_fault_model.joblib")
    train.add_argument("--seed", type=int, default=42)
    infer = commands.add_parser("predict", help="predict labels for a CSV of sensor readings")
    infer.add_argument("--model", default="models/circuit_fault_model.joblib")
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    if args.command == "generate-data":
        path = save_dataset(args.output, args.rows_per_class, args.seed)
        print(f"Wrote synthetic demo data to {path}")
    elif args.command == "train":
        result = train_model(args.data, args.model, args.seed)
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(result["report"])
        print(f"Saved model to {result['model_path']}")
    else:
        readings = pd.read_csv(args.input)
        result = predict(args.model, readings)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"Wrote predictions to {output}")


if __name__ == "__main__":
    main()
