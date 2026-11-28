"""Run one architecture/domain with five folds, reusing finished fold outputs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exps.baseline.train_steps import train_fold, write_json
from exps.hw2.study import FOLDS, PATCH_SIZES, ROOT, VARIANTS, config_for
from src.data import IMUDataset, PLATFORMS, make_folds, read_index


def run(variant, platform, steps=None):
    config = config_for(variant, platform)
    output = Path(config["output_dir"])
    if steps is not None:
        if not steps or any(step <= 0 for step in steps) or sorted(set(steps)) != steps:
            raise ValueError("Snapshot steps must be positive, unique, and ascending")
        budget = "steps_" + "_".join(map(str, steps))
        output = ROOT / budget / variant / platform
        config.update(max_steps=steps[-1], snapshot_steps=steps, output_dir=str(output))
    output.mkdir(exist_ok=True, parents=True)
    sources = [Path(__file__), Path("exps/baseline/train_steps.py"), Path("exps/hw2/study.py"),
               *sorted(Path("src").rglob("*.py")), FOLDS, Path("uv.lock")]
    manifest = {"config": config, "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    path = output / "manifest.json"
    if (path.exists() and json.loads(path.read_text())["config"] == config
            and all((output / f"fold_{fold}/oof.csv").is_file() for fold in config["folds"])):
        # A finished run keeps the source manifest from the actual training process.
        return
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"Run sources/configuration changed: {output}")
    write_json(path, manifest)
    frame = read_index(config["data_root"], "train")
    assignments = pd.read_csv(FOLDS)
    pd.testing.assert_frame_equal(assignments, make_folds(frame, 5, 42))
    if platform != "all":
        frame = frame[frame.platform.eq(platform)].reset_index(drop=True)
    torch.set_num_threads(config["num_threads"])
    dataset = IMUDataset(config["data_root"], "train", frame)
    for fold in config["folds"]:
        destination = output / f"fold_{fold}"
        if (destination / "oof.csv").is_file():
            continue
        # Keep partial attempts isolated so a failed process can be restarted.
        attempt = output / f"attempt_{fold}_{time.time_ns()}"
        attempt.mkdir()
        snapshots, _, _ = train_fold(dataset, frame, assignments, fold, config, attempt)
        _, oof = snapshots[config["max_steps"]]
        directory = attempt / f"fold_{fold}"
        if steps is not None:
            for step, (_, predictions) in snapshots.items():
                predictions.to_csv(directory / f"step_{step}_oof.csv", index=False)
        oof.to_csv(directory / "oof.csv", index=False)
        directory.rename(destination)
        attempt.rmdir()
    print(f"Completed {variant}/{platform}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("variant", choices=(*VARIANTS, *PATCH_SIZES))
    parser.add_argument("platform", choices=("all", *PLATFORMS))
    parser.add_argument("--steps", nargs="+", type=int, help="Fixed snapshot steps for a separate training run")
    args = parser.parse_args()
    run(args.variant, args.platform, args.steps)


if __name__ == "__main__":
    main()
