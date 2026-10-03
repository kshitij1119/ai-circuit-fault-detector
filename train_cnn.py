"""Train a compact 1D-CNN directly on LTspice transient waveforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


class WaveformDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, root: Path, labels: dict[str, int], points: int = 2048):
        self.samples = []
        for row in manifest.itertuples(index=False):
            waveform = pd.read_csv(root / row.waveform_path)
            t = waveform["time_s"].to_numpy(dtype=float)
            y = waveform["output_v"].to_numpy(dtype=float)
            if len(t) < 8 or not np.isfinite(y).all():
                raise ValueError(f"Invalid waveform in {row.waveform_path}")
            uniform_t = np.linspace(float(t[0]), float(t[-1]), points)
            resampled = np.interp(uniform_t, t, y)
            # Absolute amplitude is retained because it carries fault information.
            x = torch.from_numpy(resampled.astype(np.float32)).unsqueeze(0)
            self.samples.append((x, labels[row.fault_label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FaultCNN(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, class_count))

    def forward(self, x):
        return self.classifier(self.features(x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/ltspice/manifest.csv")
    parser.add_argument("--model", default="models/waveform_cnn.pt")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest_path = Path(args.manifest)
    frame = pd.read_csv(manifest_path)
    if not {"fault_label", "waveform_path"}.issubset(frame.columns):
        raise ValueError("Manifest must contain fault_label and waveform_path columns")
    labels = sorted(frame["fault_label"].unique())
    if (frame["fault_label"].value_counts() < 5).any():
        raise ValueError("Each fault class requires at least five waveform samples for train/validation/test splits")
    train_val, test_rows = train_test_split(frame, test_size=0.2, random_state=args.seed, stratify=frame["fault_label"])
    train_rows, val_rows = train_test_split(train_val, test_size=0.2, random_state=args.seed, stratify=train_val["fault_label"])
    label_ids = {name: index for index, name in enumerate(labels)}
    train_loader = DataLoader(WaveformDataset(train_rows, manifest_path.parent, label_ids), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(WaveformDataset(val_rows, manifest_path.parent, label_ids), batch_size=args.batch_size)
    test_loader = DataLoader(WaveformDataset(test_rows, manifest_path.parent, label_ids), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultCNN(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float("inf")
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y)
        train_loss /= len(train_rows)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += float(loss_fn(model(x), y).item()) * len(y)
        val_loss /= len(val_rows)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "labels": labels, "points": 2048}, model_path)
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    saved = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            actual.extend(y.numpy().tolist())
    report = classification_report(actual, predicted, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
    report_dir = Path(args.reports)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cnn_metrics.json").write_text(json.dumps({"accuracy": accuracy_score(actual, predicted), "classification_report": report, "labels": labels}, indent=2), encoding="utf-8")
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(labels))))
    np.savetxt(report_dir / "cnn_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    print(f"Held-out accuracy: {accuracy_score(actual, predicted):.3f}; model: {model_path}")


if __name__ == "__main__":
    main()
