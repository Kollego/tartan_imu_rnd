"""Run and persist a single fixed-budget HW2 generalist fold."""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch

from exps.baseline.train_steps import train_fold, write_json
from exps.hw2.study import FOLDS
from src.data import IMUDataset, make_folds, read_index


STEPS = (20_000, 40_000)


def train_generalist_fold(fold, output, config, source, metadata=None):
    train_fixed_fold(fold, output, config, source, STEPS, metadata)


def train_fixed_fold(fold, output, config, source, steps, metadata=None):
    config = dict(config, max_steps=steps[-1], snapshot_steps=list(steps),
                  eval_every_steps=steps[-1], output_dir=str(output))
    destination = output / f"fold_{fold}"
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["config"] == config and all(
                (destination / f"step_{step}_oof.csv").is_file() and (destination / f"step_{step}.pt").is_file()
                for step in steps):
            return
        raise ValueError(f"Existing fold differs from this configuration: {destination}")
    frame = read_index(config["data_root"], "train")
    assignments = pd.read_csv(FOLDS)
    pd.testing.assert_frame_equal(assignments, make_folds(frame, config["n_splits"], 42))
    if config["platform"] is not None:
        frame = frame[frame.platform.eq(config["platform"])].reset_index(drop=True)
    sources = [Path(__file__), source, FOLDS, Path("uv.lock"), Path("exps/baseline/train_steps.py"),
               Path("exps/hw2/study.py"), Path("exps/baseline/configs/mlp_steps.py"),
               *sorted(Path("src").rglob("*.py")),
               *sorted((Path(config["data_root"]) / "index").glob("train*.csv"))]
    manifest = {"config": config, "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                **(metadata or {})}
    torch.set_num_threads(config["num_threads"])
    dataset = IMUDataset(config["data_root"], "train", frame)
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / f"attempt_{fold}_{time.time_ns()}"
    attempt.mkdir()
    directory = attempt / f"fold_{fold}"

    def save_snapshot(step, predictions):
        path = directory / f"step_{step}_oof.csv"
        temporary = path.with_suffix(".tmp")
        predictions.to_csv(temporary, index=False)
        temporary.replace(path)

    train_fold(dataset, frame, assignments, fold, config, attempt, on_snapshot=save_snapshot)
    write_json(directory / "manifest.json", manifest)
    directory.rename(destination)
    attempt.rmdir()
    print(f"Completed fold {fold}", flush=True)
