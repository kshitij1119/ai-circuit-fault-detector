"""Generate exploratory waveform and feature plots from local simulation output."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from circuit_fault_detector.features import FEATURE_COLUMNS


def main() -> None:
    feature_path = Path("data/processed/features.csv")
    manifest_path = Path("data/ltspice/manifest.csv")
    output_dir = Path("reports/eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(feature_path)

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=features, x="fault_label", y="peak_amplitude_v")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "peak_amplitude_by_fault.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(features[FEATURE_COLUMNS].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation.png", dpi=160)
    plt.close()

    manifest = pd.read_csv(manifest_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, group in manifest.groupby("fault_label"):
        trace = pd.read_csv(manifest_path.parent / group.iloc[0]["waveform_path"])
        ax.plot(trace["time_s"], trace["output_v"], label=label, linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Output voltage (V)", title="Representative LTspice waveforms")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "waveforms_by_fault.png", dpi=160)
    plt.close(fig)
    print(f"EDA plots written to {output_dir}")


if __name__ == "__main__":
    main()
