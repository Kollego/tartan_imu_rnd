"""MLP experiment with fixed optimizer-step budgets and measured CPU time."""

import argparse
import hashlib
import json
import os
import platform
import random
import runpy
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data import PLATFORMS, IMUDataset, compute_normalization, make_folds, read_index
from src.inference import predict
from src.metrics import velocity_metrics
from src.model import build_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference_budgets(frame, assignments, config):
    """Convert generalist passes to steps; split the shared budget exactly."""
    epochs = config["reference_epochs"]
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("reference_epochs must be a positive integer")
    rows = []
    for fold in config["folds"]:
        validation_ids = set(assignments.loc[assignments.fold.eq(fold), "traj_id"])
        train_windows = int((~frame.traj_id.isin(validation_ids)).sum())
        steps = epochs * (train_windows // config["batch_size"])
        if steps < len(PLATFORMS):
            raise ValueError("Generalist budget must provide at least one step per specialist")
        shared_steps = steps
        if config["platform"] is not None:
            index = PLATFORMS.index(config["platform"])
            base, remainder = divmod(steps, len(PLATFORMS))
            shared_steps = base + int((index - fold) % len(PLATFORMS) < remainder)
        rows.append({"fold": fold, "generalist_train_windows": train_windows,
                     "generalist_steps": steps, "equal_total_steps": shared_steps})
    return pd.DataFrame(rows)


def train_fold(dataset, frame, assignments, fold, config, output):
    """Save fixed-step checkpoints and their metrics."""
    directory = Path(output) / f"fold_{fold}"
    directory.mkdir()
    fold_started, fold_cpu_started = time.perf_counter(), time.process_time()
    seed = config["seed"] + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    validation_ids = set(assignments.loc[assignments.fold.eq(fold), "traj_id"])
    validation_mask = frame.traj_id.isin(validation_ids).to_numpy()
    train_rows, val_rows = np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)
    if len(train_rows) < config["batch_size"] or not len(val_rows):
        raise ValueError("Need a full training batch and a nonempty validation fold")
    norm_started, norm_cpu_started = time.perf_counter(), time.process_time()
    mean, std = compute_normalization(dataset.x[train_rows])
    normalization_seconds = time.perf_counter() - norm_started
    normalization_cpu_seconds = time.process_time() - norm_cpu_started
    kwargs = {"hidden_sizes": config["hidden_sizes"]}
    model = build_model("mlp", **kwargs, mean=mean, std=std)
    train_set, val_set = Subset(dataset, train_rows), Subset(dataset, val_rows)
    val_frame = frame.iloc[val_rows].reset_index(drop=True)
    loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    zero = velocity_metrics(val_frame, np.zeros((len(val_frame), 3)))
    training_ids = sorted(frame.iloc[train_rows].traj_id.unique().tolist())
    iterator = None
    train_seconds = train_cpu_seconds = evaluation_seconds = evaluation_cpu_seconds = 0.0
    interval_loss = 0.0
    interval_examples = validation_calls = 0
    best_ave, best_step = float("inf"), None
    history, snapshots = [], {}
    for step in range(1, config["max_steps"] + 1):
        started, cpu_started = time.perf_counter(), time.process_time()
        if iterator is None:
            iterator = iter(loader)
        try:
            imu, target = next(iterator)
        except StopIteration:
            # Restart the loader to obtain a NEW shuffled pass, never cache batches.
            iterator = iter(loader)
            imu, target = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(imu), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        interval_loss += loss.item() * len(imu)
        interval_examples += len(imu)
        train_seconds += time.perf_counter() - started
        train_cpu_seconds += time.process_time() - cpu_started
        evaluate = step % config["eval_every_steps"] == 0 or step in config["snapshot_steps"]
        if not evaluate:
            continue
        model.eval()
        started, cpu_started = time.perf_counter(), time.process_time()
        predictions = predict(model, val_set)
        metrics = velocity_metrics(val_frame, predictions)
        evaluation_seconds += time.perf_counter() - started
        evaluation_cpu_seconds += time.process_time() - cpu_started
        validation_calls += 1
        if metrics["macro_ave"] < best_ave:
            best_ave, best_step = metrics["macro_ave"], step
        compute = {
            "train_steps": step, "train_windows": len(train_rows),
            "batch_size": config["batch_size"], "windows_seen": step * config["batch_size"],
            "train_seconds": train_seconds, "train_cpu_seconds": train_cpu_seconds,
            "evaluation_seconds": evaluation_seconds, "evaluation_cpu_seconds": evaluation_cpu_seconds,
            "validation_calls": validation_calls, "normalization_seconds": normalization_seconds,
            "normalization_cpu_seconds": normalization_cpu_seconds,
            "elapsed_seconds": time.perf_counter() - fold_started,
            "elapsed_cpu_seconds": time.process_time() - fold_cpu_started,
        }
        history.append({"fold": fold, "step": step, "train_mse": interval_loss / interval_examples,
                        "cv_macro_ave": metrics["macro_ave"], "best_step_so_far": best_step,
                        "best_ave_so_far": best_ave, **compute})
        interval_loss = 0.0
        interval_examples = 0
        checkpoint = {
            "model_name": "mlp", "model_kwargs": kwargs,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "fold": fold, "step": step, "seed": seed, "config": config,
            "cv_macro_ave": metrics["macro_ave"], "compute": compute,
            "train_trajectories": training_ids,
            "validation_trajectories": sorted(val_frame.traj_id.unique().tolist()),
        }
        if best_step == step:
            torch.save(checkpoint, directory / "best.pt")
        if step in config["snapshot_steps"]:
            torch.save(checkpoint, directory / f"step_{step}.pt")
            if step == config["max_steps"]:
                torch.save(checkpoint, directory / "final.pt")
            oof = val_frame[["window_id", "platform", "traj_id", "win_idx", "vx", "vy", "vz"]].copy()
            oof[["pred_vx", "pred_vy", "pred_vz"]] = predictions
            oof["fold"] = fold
            row = {"fold": fold, "step": step, "macro_ave": metrics["macro_ave"],
                   "zero_macro_ave": zero["macro_ave"], "best_step_so_far": best_step,
                   "best_ave_so_far": best_ave, **compute, **metrics["per_platform"]}
            snapshots[step] = (row, oof)
            write_json(directory / f"step_{step}_metrics.json", {"model": metrics, "zero": zero, **compute})
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"fold={fold} step={step} cv_AVE={metrics['macro_ave']:.5f} "
              f"train_wall={train_seconds:.2f}s train_CPU={train_cpu_seconds:.2f}s", flush=True)
    return snapshots, history, {
        "fold": fold, "total_wall_seconds": time.perf_counter() - fold_started,
        "total_cpu_seconds": time.process_time() - fold_cpu_started, **compute,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="exps/baseline/configs/mlp_steps.py")
    parser.add_argument("--output")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    config = dict(runpy.run_path(args.config)["CONFIG"])
    if args.output:
        config["output_dir"] = args.output
    if args.fold is not None:
        config["folds"] = [args.fold]
    reference_mode = "reference_epochs" in config
    if (config["device"] != "cpu" or config["drop_last"] is not True
            or config["eval_every_steps"] < 1 or config["batch_size"] < 1 or not config["folds"]
            or any(f not in range(config["n_splits"]) for f in config["folds"])):
        raise ValueError("Invalid CPU step-budget configuration")
    if not reference_mode and (config["max_steps"] < 1 or not config["snapshot_steps"]
            or len(set(config["snapshot_steps"])) != len(config["snapshot_steps"])
            or max(config["snapshot_steps"]) != config["max_steps"]
            or min(config["snapshot_steps"]) < 1):
        raise ValueError("Invalid fixed step budget")
    output, report = Path(config["output_dir"]), Path("reports") / Path(config["output_dir"]).name
    if output.exists() or report.exists():
        raise FileExistsError("Choose a new output name; previous runs/reports are never overwritten")
    output.mkdir(parents=True)
    run_started, run_cpu_started = time.perf_counter(), time.process_time()
    torch.set_num_threads(config["num_threads"])
    write_json(output / "config.json", config)
    sources = sorted(Path("src").rglob("*.py")) + [Path(__file__), Path(args.config)]
    write_json(output / "environment.json", {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "pandas": pd.__version__, "system": platform.platform(),
        "machine": platform.machine(), "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "index_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((Path(config["data_root"]) / "index").glob("*.csv"))},
    })
    frame = read_index(config["data_root"], "train")
    assignments = make_folds(frame, config["n_splits"], config["seed"])
    assignments.to_csv(output / "folds.csv", index=False)
    if reference_mode:
        budgets = reference_budgets(frame, assignments, config)
        budgets.to_csv(output / "fold_budgets.csv", index=False)
        budgets = budgets.set_index("fold")
    if config["platform"] is not None:
        frame = frame[frame.platform.eq(config["platform"])].reset_index(drop=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    print(f"Loading {len(frame)} windows; platform={config['platform'] or 'all'}", flush=True)
    dataset = IMUDataset(config["data_root"], "train", frame)
    load_seconds, load_cpu_seconds = time.perf_counter() - started, time.process_time() - cpu_started
    if reference_mode:
        collected = {"final": []}
        if config["platform"] is not None:
            collected["equal_total"] = []
    else:
        collected = {step: [] for step in config["snapshot_steps"]}
    all_history, fold_compute = [], []
    for fold in config["folds"]:
        fold_config = dict(config)
        if reference_mode:
            final_step = int(budgets.loc[fold, "generalist_steps"])
            shared_step = int(budgets.loc[fold, "equal_total_steps"])
            fold_config.update(max_steps=final_step, snapshot_steps=sorted({shared_step, final_step}))
        snapshots, history, compute = train_fold(dataset, frame, assignments, fold, fold_config, output)
        if reference_mode:
            collected["final"].append(snapshots[final_step])
            if config["platform"] is not None:
                collected["equal_total"].append(snapshots[shared_step])
        else:
            for step, values in snapshots.items():
                collected[step].append(values)
        all_history.extend(history)
        fold_compute.append(compute)
    report.mkdir(parents=True)
    for step, values in collected.items():
        name = step if isinstance(step, str) else f"step_{step}"
        destination = output / name
        destination.mkdir()
        fold_metrics = pd.DataFrame([row for row, _ in values])
        oof = pd.concat([oof for _, oof in values], ignore_index=True).sort_values("window_id")
        metrics = velocity_metrics(oof, oof[["pred_vx", "pred_vy", "pred_vz"]].to_numpy())
        summary = {
            "step": int(fold_metrics.step.iloc[0]) if fold_metrics.step.nunique() == 1 else None,
            "steps_per_fold": {int(row.fold): int(row.step) for row in fold_metrics.itertuples()},
            "checkpoint_selection": "fixed_step",
            "completed_folds": config["folds"], "fold_mean_ave": float(fold_metrics.macro_ave.mean()),
            "fold_std_ave": float(fold_metrics.macro_ave.std(ddof=1)) if len(values) > 1 else None,
            "oof": metrics, "zero_oof": velocity_metrics(oof, np.zeros((len(oof), 3))),
            "note": "Metrics use the fixed-step snapshot. Earlier snapshots share the same training prefix. AVE only.",
        }
        write_json(destination / "summary.json", summary)
        fold_metrics.to_csv(destination / "fold_metrics.csv", index=False)
        oof.to_csv(destination / "oof_predictions.csv", index=False)
        snapshot_report = report / name
        snapshot_report.mkdir()
        for name in ("summary.json", "fold_metrics.csv"):
            (snapshot_report / name).write_bytes((destination / name).read_bytes())
    history = pd.DataFrame(all_history)
    history.to_csv(output / "history.csv", index=False)
    write_json(output / "run_compute.json", {
        "total_wall_seconds": time.perf_counter() - run_started,
        "total_cpu_seconds": time.process_time() - run_cpu_started,
        "data_load_wall_seconds": load_seconds, "data_load_cpu_seconds": load_cpu_seconds,
        "folds": fold_compute,
        "notes": ["CPU times sum CPU usage across threads; wall time measures elapsed real time.",
                  "Training timers include batch fetching, forward, backward and optimizer step; exclude validation and checkpoint IO.",
                  "Evaluation timers include prediction and AVE calculation; IO is included only in total elapsed times.",
                  "Snapshot elapsed times stop after validation, before that snapshot is saved.",
                  "Shared-prefix snapshots do not represent two independent training runs. Times are measurements, not FLOPs."],
    })
    names = ["config.json", "environment.json", "folds.csv", "history.csv", "run_compute.json"]
    if reference_mode:
        names.append("fold_budgets.csv")
    for name in names:
        (report / name).write_bytes((output / name).read_bytes())
    print(f"Saved fixed-step run: {output}", flush=True)


if __name__ == "__main__":
    main()
