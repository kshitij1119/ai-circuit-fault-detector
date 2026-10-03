"""Extract the eight planned signal features from an LTspice waveform manifest."""

import argparse
from pathlib import Path

from circuit_fault_detector.features import extract_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/ltspice/manifest.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    parser.add_argument("--fundamental-hz", type=float, default=1000.0)
    args = parser.parse_args()
    print(f"Feature table written to {extract_manifest(args.manifest, args.output, args.fundamental_hz)}")


if __name__ == "__main__":
    main()
