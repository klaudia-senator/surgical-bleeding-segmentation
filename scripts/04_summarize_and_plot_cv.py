from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surgical_bleeding.metrics import summarize_folds


def histories(root: Path) -> list[pd.DataFrame]:
    paths = sorted(root.glob("fold_*/history.csv"), key=lambda p: int(p.parent.name.rsplit("_", 1)[-1]))
    return [pd.read_csv(path) for path in paths]


def plot(values: list[pd.DataFrame], train_key: str, val_key: str, label: str, output: Path) -> None:
    plt.figure(figsize=(8, 5))
    for frame in values:
        plt.plot(frame.epoch, frame[train_key], color="#377eb8", alpha=0.25)
        plt.plot(frame.epoch, frame[val_key], color="#e41a1c", alpha=0.25, linestyle="--")
    epochs = sorted(set().union(*(set(frame.epoch) for frame in values)))
    train = [np.mean([frame.loc[frame.epoch == epoch, train_key].iloc[0] for frame in values if epoch in set(frame.epoch)]) for epoch in epochs]
    validation = [np.mean([frame.loc[frame.epoch == epoch, val_key].iloc[0] for frame in values if epoch in set(frame.epoch)]) for epoch in epochs]
    plt.plot(epochs, train, color="#377eb8", linewidth=2.5, label="Train mean")
    plt.plot(epochs, validation, color="#e41a1c", linewidth=2.5, linestyle="--", label="Validation mean")
    plt.xlabel("Epoch"); plt.ylabel(label); plt.legend(); plt.tight_layout(); plt.savefig(output, dpi=220); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results-dir", type=Path, required=True); args = parser.parse_args()
    values = histories(args.results_dir)
    if not values:
        raise FileNotFoundError("No fold histories found")
    metrics = pd.read_csv(args.results_dir / "fold_metrics.csv")
    summary = {"n_folds": len(metrics), "std_definition": "sample SD (ddof=1)"}
    for source, name in [("best_val_dice", "dice"), ("best_val_iou", "iou"), ("best_val_loss", "val_loss")]:
        summary[f"{name}_mean"], summary[f"{name}_std"] = summarize_folds(metrics[source].to_numpy())
    (args.results_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(args.results_dir / "cv_summary.csv", index=False)
    for train, val, label, filename in [("train_loss", "val_loss", "Loss", "training_curves_loss.png"), ("train_dice", "val_dice", "Dice", "training_curves_dice.png"), ("train_iou", "val_iou", "IoU", "training_curves_iou.png")]:
        plot(values, train, val, label, args.results_dir / filename)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

