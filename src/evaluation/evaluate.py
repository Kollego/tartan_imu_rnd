"""Evaluate any checkpoint supporting the common model interface."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import IMUDataset, read_index
from src.inference import load_model, predict
from src.metrics import velocity_metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/tartan-imu-challenge-iros2026")
    parser.add_argument("--split", choices=["cv", "val"], default="val")
    parser.add_argument("--output", required=True, help="New directory for metrics and predictions")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint)
    split = "train" if args.split == "cv" else "val"
    frame = read_index(args.data_root, split)
    if args.split == "cv":
        frame = frame[frame.traj_id.isin(checkpoint["validation_trajectories"])].reset_index(drop=True)
    platform = checkpoint["config"]["platform"]
    if platform is not None:
        frame = frame[frame.platform.eq(platform)].reset_index(drop=True)
    if set(frame.traj_id) & set(checkpoint["train_trajectories"]):
        raise ValueError("Evaluation trajectories overlap the training set")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    predictions = predict(model, IMUDataset(args.data_root, split, frame))
    metrics = {"checkpoint": args.checkpoint, "split": args.split,
               "model": velocity_metrics(frame, predictions),
               "zero": velocity_metrics(frame, np.zeros_like(predictions))}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    frame[["window_id"]].assign(vx=predictions[:, 0], vy=predictions[:, 1], vz=predictions[:, 2]).to_csv(output / "predictions.csv", index=False)
    print(json.dumps({"model_macro_ave": metrics["model"]["macro_ave"],
                      "zero_macro_ave": metrics["zero"]["macro_ave"]}, indent=2))


if __name__ == "__main__":
    main()
