"""P25 seed study on fixed trajectory folds, same 10k steps per model."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

import pandas as pd
import torch

from exps.baseline.train_steps import train_fold, write_json
from exps.hw2.results import PRED, expected_rows, validate
from exps.hw2.study import FOLDS, config_for
from exps.hw3.audit_score import attach_pose, load_scorer, summarize
from exps.hw3.budget_score import METRICS
from exps.hw3.epoch_score import slices
from src.data import IMUDataset, PLATFORMS, make_folds

RUN = Path("runs/hw2/seed_stability_p25_steps_10000")
REFERENCE = Path("runs/hw2/temporal_gate_p25")
SEEDS = (42, 52, 62)
STEPS = 10000

def train(seed, domain, run):
    destination = run / str(seed) / domain
    destination.mkdir(parents=True, exist_ok=True)
    config = config_for("temporal_gate_p25", domain)
    config.update(seed=seed, output_dir=str(destination), max_steps=STEPS,
                  snapshot_steps=[STEPS], eval_every_steps=STEPS, num_threads=1)
    write_json(destination / "config.json", config)
    assignments = pd.read_csv(FOLDS)
    frame = expected_rows()
    frame = frame if domain == "all" else frame[frame.platform.eq(domain)]
    frame = frame.drop(columns="fold").reset_index(drop=True)
    dataset = IMUDataset(config["data_root"], "train", frame)
    torch.set_num_threads(config["num_threads"])
    for fold in config["folds"]:
        saved = destination / f"fold_{fold}/oof.csv"
        if saved.is_file():
            continue
        attempt = destination / f"attempt_{fold}_{os.getpid()}"
        attempt.mkdir()
        snapshots, _, _ = train_fold(dataset, frame, assignments, fold, config, attempt)
        _, predictions = snapshots[STEPS]
        predictions.to_csv(attempt / f"fold_{fold}/oof.csv", index=False)
        (attempt / f"fold_{fold}").rename(saved.parent)
        attempt.rmdir()
        print(f"seed={seed} domain={domain} fold={fold} done", flush=True)

def score(run, scorer_path):
    rows = expected_rows()
    pd.testing.assert_frame_equal(pd.read_csv(FOLDS), make_folds(rows, 5, 42))
    scorer = load_scorer(scorer_path)
    poses = attach_pose(rows, "train", (100, 199))
    results = []
    for seed in SEEDS:
        root = REFERENCE if seed == 42 else run / str(seed)
        for role, domains in [("generalist", ("all",)), ("specialists", PLATFORMS)]:
            predictions = []
            for domain in domains:
                for fold in range(5):
                    path = root / domain / f"fold_{fold}"
                    checkpoint = torch.load(path / f"step_{STEPS}.pt", weights_only=True)
                    expected = rows[rows.fold.eq(fold)]
                    if domain != "all":
                        expected = expected[expected.platform.eq(domain)]
                    training = rows[rows.fold.ne(fold)]
                    if domain != "all":
                        training = training[training.platform.eq(domain)]
                    if ((checkpoint["fold"], checkpoint["step"], checkpoint["seed"]) != (fold, STEPS, seed + fold)
                            or set(checkpoint["validation_trajectories"]) != set(expected.traj_id)
                            or set(checkpoint["train_trajectories"]) != set(training.traj_id)
                            or checkpoint["model_kwargs"] != config_for("temporal_gate_p25", domain)["model_kwargs"]):
                        raise ValueError(f"Unexpected seed study checkpoint in {path}")
                    predictions.append(validate(pd.read_csv(path / "oof.csv"), expected))
            predictions = validate(pd.concat(predictions, ignore_index=True), rows)
            scored = poses.merge(predictions[["window_id", *PRED]], on="window_id", validate="one_to_one")
            _, per_trajectory = summarize(scorer, scored)
            for platform, subset in slices(per_trajectory):
                metrics = (subset.groupby(level=["fold", "platform"])[METRICS].mean()
                           .groupby(level="fold").mean().mean())
                results.append(dict(base_seed=seed, role=role, platform=platform, **metrics.to_dict()))
    result = pd.DataFrame(results)
    result.to_csv(run / "summary.csv", index=False)
    print(result.query("platform == 'all'").to_string(index=False), flush=True)

def launch(seed, domain, run):
    with (run / f"seed_{seed}.{domain}.log").open("ab") as log:
        cmd = [sys.executable, "-m", "exps.hw3.seed_stability", "folds", str(seed), domain, "--output", str(run)]
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"seed={seed} domain={domain} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    folds = commands.add_parser("folds")
    folds.add_argument("seed", type=int, choices=SEEDS[1:])
    folds.add_argument("domain", choices=("all", *PLATFORMS))
    folds.add_argument("--output", type=Path, default=RUN)
    for name in ("run", "score"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, default=RUN)
        command.add_argument("--scorer", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "folds":
        train(args.seed, args.domain, args.output)
        return
    if args.command == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = [executor.submit(launch, seed, domain, args.output)
                    for domain in ("all", *PLATFORMS) for seed in SEEDS[1:]]
            for job in jobs:
                job.result()
    score(args.output, args.scorer)


if __name__ == "__main__":
    main()
