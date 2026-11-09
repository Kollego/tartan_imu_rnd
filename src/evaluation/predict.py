"""Generate a Kaggle-format CSV with one unified model. Does not upload it."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import IMUDataset, read_index
from src.inference import load_model, predict


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/tartan-imu-challenge-iros2026")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    model, checkpoint = load_model(args.checkpoint)
    if checkpoint["config"]["platform"] is not None:
        raise ValueError("Use a unified generalist checkpoint for the competition submission")
    frame = read_index(args.data_root, "test", with_targets=False)
    predictions = predict(model, IMUDataset(args.data_root, "test", frame))
    prediction_frame = frame[["window_id"]].copy()
    prediction_frame[["vx", "vy", "vz"]] = predictions
    template = pd.read_csv(Path(args.data_root) / "sample_submission.csv")
    if not template.window_id.is_unique or set(template.window_id) != set(frame.window_id):
        raise ValueError("Submission template and test index IDs differ")
    submission = template[["window_id"]].merge(prediction_frame, on="window_id", how="left", sort=False, validate="one_to_one")
    submission = submission[template.columns]
    if not np.isfinite(submission[["vx", "vy", "vz"]]).all().all():
        raise ValueError("Nonfinite submission predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Saved {len(submission)} predictions to {output}")


if __name__ == "__main__":
    main()
