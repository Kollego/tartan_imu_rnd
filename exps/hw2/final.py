"""Refit the CV-selected generalist, evaluate official val once and write the submission."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from exps.baseline.train_steps import train_fold, write_json
from exps.hw2.results import OUT
from exps.hw2.study import ROOT, config_for
from src.data import IMUDataset, read_index
from src.inference import load_model, predict
from src.metrics import velocity_metrics


def main():
    selection = json.loads((OUT / "selection.json").read_text())
    config = config_for(selection["variant"], "all")
    config.update(eval_every_steps=config["max_steps"], folds=[0], output_dir=str(ROOT / "final"))
    root = Path(config["data_root"])
    output = ROOT / "final"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", selection)
    train, val = (read_index(root, split) for split in ("train", "val"))
    if set(train.traj_id) & set(val.traj_id) or set(train.window_id) & set(val.window_id):
        raise ValueError("Official train/val overlap")
    train_data = IMUDataset(root, "train", train)
    val_data = IMUDataset(root, "val", val)
    data = ConcatDataset([train_data, val_data])
    data.x = np.concatenate([train_data.x, val_data.x])
    frame = pd.concat([train, val], ignore_index=True)
    # train_fold holds out fold 0. Only the official train rows enter its
    # optimizer and normalization. The sole evaluation is at the fixed end.
    assignments = pd.concat([train[["traj_id"]].drop_duplicates().assign(fold=-1),
                             val[["traj_id"]].drop_duplicates().assign(fold=0)])
    torch.set_num_threads(config["num_threads"])
    snapshots, _, _ = train_fold(data, frame, assignments, 0, config, output)
    _, predictions = snapshots[config["max_steps"]]
    checkpoint = output / "fold_0/final.pt"
    model, state = load_model(checkpoint)
    if set(state["train_trajectories"]) != set(train.traj_id):
        raise ValueError("Final checkpoint does not cover the full training split")
    saved = predictions[["pred_vx", "pred_vy", "pred_vz"]].to_numpy()
    np.testing.assert_array_equal(saved, predict(model, val_data))
    metrics = velocity_metrics(val, saved)
    predictions.to_csv(output / "val_predictions.csv", index=False)
    write_json(output / "val_metrics.json", metrics)
    submission = output / "submission.csv"
    subprocess.run([sys.executable, "-m", "scripts.predict", "--checkpoint", str(checkpoint),
                    "--data-root", str(root), "--output", str(submission)], check=True)
    actual = pd.read_csv(submission)
    template = pd.read_csv(root / "sample_submission.csv")
    if list(actual.columns) != list(template.columns) or not actual.window_id.equals(template.window_id):
        raise ValueError("Submission schema or row order differs from the official template")
    if not actual.window_id.is_unique or not np.isfinite(actual[["vx", "vy", "vz"]]).all().all():
        raise ValueError("Invalid submission rows")
    write_json(output / "verified.json", {"variant": selection["variant"], "training_trajectories": int(train.traj_id.nunique()),
               "training_windows": len(train), "val_trajectories": int(val.traj_id.nunique()), "val_windows": len(val),
               "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
               "val_macro_ave": metrics["macro_ave"], "val_per_platform": metrics["per_platform"],
               "test_rows": len(actual), "columns": list(actual.columns), "submission_uploaded": False})
    (OUT / "official_val.json").write_bytes((output / "verified.json").read_bytes())
    print((output / "verified.json").read_text())


if __name__ == "__main__":
    main()
