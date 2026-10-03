"""Train and evaluate the Random Forest on extracted LTspice waveform features."""

import argparse

from circuit_fault_detector.model import train_random_forest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--model", default="models/random_forest.joblib")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_random_forest(args.features, args.model, args.reports, args.seed)
    print(f"Held-out accuracy: {result['accuracy']:.3f}")
    print(f"Model written to {result['model_path']}")
    print(f"Metrics and plots written to {args.reports}")


if __name__ == "__main__":
    main()
