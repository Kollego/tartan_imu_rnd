"""Train temporal_gate_p25 with matched data passes and matched optimizer steps."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch

from exps.baseline.train_steps import train_fold, write_json
from exps.hw2.results import expected_rows
from exps.hw2.study import FOLDS, ROOT, config_for
from src.data import IMUDataset, PLATFORMS, make_folds


VARIANT = "temporal_gate_p25"
DOMAINS = ("all", *PLATFORMS)
REGIMES = ("generalist", "same_epochs", "same_steps_per_specialist")
EPOCHS = (1, 2, 3, 4)


def output_for(epochs=EPOCHS, *, steps=None):
    unit, budgets = ("epochs", epochs) if steps is None else ("steps", steps)
    return ROOT / (f"matched_{unit}_" + "_".join(map(str, budgets))) / VARIANT


RUN = output_for(EPOCHS)


def budget_plan(rows, config, epochs=EPOCHS, *, steps=None):
    budgets = tuple(epochs if steps is None else steps)
    if not budgets or budgets[0] < 1 or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Budgets must be positive, unique, and ascending")
    budget_column = "epoch_budget" if steps is None else "generalist_steps"
    batch_size = config["batch_size"]
    records = []
    for fold in config["folds"]:
        train = rows[rows.fold.ne(fold)]
        counts = train.platform.value_counts()
        generalist_batches = len(train) // batch_size
        for domain in DOMAINS:
            windows = len(train) if domain == "all" else int(counts[domain])
            batches = windows // batch_size
            if not batches:
                raise ValueError(f"No full batches for {domain}, fold {fold}")
            regimes = REGIMES[:1] if domain == "all" else REGIMES[1:]
            for regime in regimes:
                for budget in budgets:
                    generalist_steps = budget * generalist_batches if steps is None else budget
                    updates = generalist_steps
                    if regime == "same_epochs":
                        # Scale by full batches per pass, rounded to the nearest update.
                        updates = max(1, (generalist_steps * batches + generalist_batches // 2) // generalist_batches)
                    record = dict(fold=fold, train_seed=config["seed"] + fold, regime=regime, domain=domain)
                    record[budget_column] = budget
                    records.append(dict(record, train_windows=windows, batches_per_epoch=batches,
                                        optimizer_steps=updates, effective_epochs=updates / batches,
                                        windows_seen=updates * batch_size))
    return pd.DataFrame(records)


def train_domain(domain, output, epochs=EPOCHS, *, steps=None):
    rows = expected_rows()
    config = config_for(VARIANT, domain)
    config["output_dir"] = str(output)
    plan = budget_plan(rows, config, epochs, steps=steps).query("domain == @domain")
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), FOLDS, Path("uv.lock"), Path("exps/baseline/train_steps.py"),
               Path("exps/hw2/study.py"), Path("exps/baseline/configs/mlp_steps.py"),
               *sorted(Path("src").rglob("*.py")),
               *sorted((Path(config["data_root"]) / "index").glob("train*.csv"))]
    manifest = {"config": config, "budgets": plan.to_dict("records"),
                "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous["config"] == config and previous["budgets"] == manifest["budgets"] and all(
                (output / f"fold_{row.fold}/step_{row.optimizer_steps}_oof.csv").is_file()
                for row in plan.itertuples()):
            print(f"Completed {VARIANT}/{domain}", flush=True)
            return
        if previous != manifest:
            raise ValueError(f"Training inputs changed for {domain}")
    write_json(manifest_path, manifest)
    plan.to_csv(output / "budgets.csv", index=False)
    assignments = pd.read_csv(FOLDS)
    pd.testing.assert_frame_equal(assignments, make_folds(rows, config["n_splits"], config["seed"]))
    frame = rows if domain == "all" else rows[rows.platform.eq(domain)]
    frame = frame.drop(columns="fold").reset_index(drop=True)
    torch.set_num_threads(config["num_threads"])
    dataset = IMUDataset(config["data_root"], "train", frame)
    for fold, budgets in plan.groupby("fold"):
        snapshot_steps = sorted(budgets.optimizer_steps.unique().tolist())
        destination = output / f"fold_{fold}"
        if destination.is_dir():
            if all((destination / f"step_{step}_oof.csv").is_file() for step in snapshot_steps):
                continue
            raise ValueError(f"Incomplete completed fold directory: {destination}")
        attempt = output / f"attempt_{fold}_{time.time_ns()}"
        attempt.mkdir()
        fold_config = dict(config, snapshot_steps=snapshot_steps, max_steps=snapshot_steps[-1],
                           eval_every_steps=snapshot_steps[-1])
        snapshots, _, _ = train_fold(dataset, frame, assignments, int(fold), fold_config, attempt)
        directory = attempt / f"fold_{fold}"
        for step, (metric, predictions) in snapshots.items():
            if metric["train_windows"] != int(budgets.train_windows.iloc[0]):
                raise ValueError(f"Training-window count differs from the budget for {domain}, fold {fold}")
            predictions.to_csv(directory / f"step_{step}_oof.csv", index=False)
        directory.rename(destination)
        attempt.rmdir()
    print(f"Completed {VARIANT}/{domain}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=DOMAINS)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    budgets.add_argument("--steps", nargs="+", type=int, help="Use exact generalist step budgets")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or output_for(args.epochs, steps=args.steps)
    train_domain(args.domain, output / args.domain, args.epochs, steps=args.steps)


if __name__ == "__main__":
    main()
