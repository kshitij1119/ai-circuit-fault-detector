"""Unified command line interface for the circuit-fault workflow."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="circuit-fault", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="generate LTspice waveform CSVs")
    simulate.add_argument("--output", default="data/ltspice")
    simulate.add_argument("--samples-per-class", type=int, default=200)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--ltspice", help="LTspice executable path")
    simulate.add_argument("--limit", type=int, help="small smoke run limit")

    extract = commands.add_parser("extract", help="extract the eight planned waveform features")
    extract.add_argument("--manifest", default="data/ltspice/manifest.csv")
    extract.add_argument("--output", default="data/processed/features.csv")
    extract.add_argument("--fundamental-hz", type=float, default=1000.0)

    train = commands.add_parser("train", help="train/evaluate the Random Forest model")
    train.add_argument("--features", default="data/processed/features.csv")
    train.add_argument("--model", default="models/random_forest.joblib")
    train.add_argument("--reports", default="reports")
    train.add_argument("--seed", type=int, default=42)

    predict = commands.add_parser("predict", help="classify a single waveform CSV")
    predict.add_argument("--input", required=True)
    predict.add_argument("--model", default="models/random_forest.joblib")
    predict.add_argument("--fundamental-hz", type=float, default=1000.0)

    args = parser.parse_args()
    if args.command == "simulate":
        from .simulation import generate_waveforms
        result = generate_waveforms(args.output, args.samples_per_class, args.seed, args.ltspice, args.limit)
        print(f"Waveform manifest written to {result}")
    elif args.command == "extract":
        from .features import extract_manifest
        result = extract_manifest(args.manifest, args.output, args.fundamental_hz)
        print(f"Feature table written to {result}")
    elif args.command == "train":
        from .model import train_random_forest
        result = train_random_forest(args.features, args.model, args.reports, args.seed)
        print(f"Held-out accuracy: {result['accuracy']:.3f}")
        print(f"Saved model to {result['model_path']}")
    else:
        import pandas as pd
        from .features import FEATURE_COLUMNS, extract_features
        from .model import load_model, predict_features
        waveform = pd.read_csv(args.input)
        if not {"time_s", "output_v"}.issubset(waveform.columns):
            raise ValueError("Input CSV must contain time_s and output_v columns")
        features = pd.DataFrame([extract_features(waveform.time_s, waveform.output_v, args.fundamental_hz)], columns=FEATURE_COLUMNS)
        model, _ = load_model(args.model)
        label, confidence, probabilities = predict_features(model, features)
        print(f"Prediction: {label} ({confidence:.1%})")
        for class_name, probability in sorted(probabilities.items(), key=lambda item: -item[1]):
            print(f"  {class_name}: {probability:.1%}")


if __name__ == "__main__":
    main()
