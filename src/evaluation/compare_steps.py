"""Compare generalists and specialists at explicit optimizer-update budgets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PLATFORMS
from src.metrics import velocity_metrics

PREDICTIONS = ["pred_vx", "pred_vy", "pred_vz"]
METADATA = ["platform", "traj_id", "win_idx", "fold", "vx", "vy", "vz"]

COMPUTE = ["train_seconds", "train_cpu_seconds", "evaluation_seconds",
           "evaluation_cpu_seconds", "validation_calls", "normalization_seconds",
           "normalization_cpu_seconds", "elapsed_seconds", "elapsed_cpu_seconds"]


def align_oof(generalist, specialists):
    """Match IDs before comparing targets/metadata; row order is irrelevant."""
    generalist = generalist.set_index("window_id").sort_index()
    specialists = specialists.set_index("window_id").sort_index()
    if not generalist.index.is_unique or not specialists.index.is_unique:
        raise ValueError("Repeated OOF window IDs")
    if not generalist.index.equals(specialists.index):
        raise ValueError("Generalist and specialists must cover the exact same windows")
    try:
        pd.testing.assert_frame_equal(generalist[METADATA], specialists[METADATA])
    except AssertionError as error:
        raise ValueError("Targets, trajectories or folds differ for matched window IDs") from error
    return generalist.reset_index(), specialists.reset_index()


def read_run(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    reference = config.get("reference_epochs")
    if (config["n_splits"] != 5 or sorted(config["folds"]) != list(range(5))
            or config["eval_every_steps"] != 250 or config["batch_size"] != 256):
        raise ValueError(f"Unexpected training configuration: {directory}")
    if reference is None:
        maximum, snapshots = config.get("max_steps"), config.get("snapshot_steps")
        if (not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1
                or not isinstance(snapshots, list) or len(snapshots) != 2
                or any(not isinstance(s, int) or isinstance(s, bool) for s in snapshots)
                or not 0 < snapshots[0] < snapshots[1] == maximum):
            raise ValueError(f"Unexpected fixed-step training configuration: {directory}")
    elif not isinstance(reference, int) or isinstance(reference, bool) or reference < 1:
        raise ValueError("reference_epochs must be a positive integer")
    if config["drop_last"] is not True:
        raise ValueError("The comparison requires full training batches (drop_last=True)")
    folds = pd.read_csv(directory / "folds.csv").sort_values("traj_id").reset_index(drop=True)
    if not folds.traj_id.is_unique or set(folds.fold) != set(range(5)):
        raise ValueError(f"Invalid trajectory folds: {directory}")
    environment = json.loads((directory / "environment.json").read_text())
    compute = json.loads((directory / "run_compute.json").read_text())
    run = {"directory": directory, "config": config, "folds": folds,
           "environment": environment, "run_compute": compute}
    if reference is not None:
        budgets = pd.read_csv(directory / "fold_budgets.csv")
        columns = ["fold", "generalist_train_windows", "generalist_steps", "equal_total_steps"]
        if (not set(columns).issubset(budgets.columns) or not budgets.fold.is_unique
                or set(budgets.fold) != set(range(5))):
            raise ValueError(f"Incomplete fold budgets: {directory}")
        values = budgets[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or not (values == np.floor(values)).all():
            raise ValueError(f"Invalid fold budgets: {directory}")
        budgets = budgets.set_index("fold").sort_index().astype(int)
        expected = reference * (budgets.generalist_train_windows // config["batch_size"])
        platform = config["platform"]
        if platform is None:
            short = expected
        elif platform in PLATFORMS:
            domain_index = PLATFORMS.index(platform)
            short = expected // len(PLATFORMS) + ((domain_index - budgets.index) % len(PLATFORMS) < expected % len(PLATFORMS)).astype(int)
        else:
            raise ValueError(f"Unknown specialist platform: {platform}")
        if (not budgets.generalist_steps.eq(expected).all() or not budgets.equal_total_steps.eq(short).all()
                or (budgets.generalist_steps < 1).any() or (budgets.equal_total_steps < 1).any()):
            raise ValueError(f"Fold budgets do not match the reference passes: {directory}")
        run["fold_budgets"] = budgets
    return run


def read_snapshot(run, step):
    if "fold_budgets" in run:
        if step not in ("final", "equal_total") or (step == "equal_total" and run["config"]["platform"] is None):
            raise ValueError("Expected final or specialist equal_total snapshot")
        directory = run["directory"] / step
        column = "generalist_steps" if step == "final" else "equal_total_steps"
        expected_steps = run["fold_budgets"][column]
    else:
        directory = run["directory"] / f"step_{step}"
        expected_steps = pd.Series(step, index=range(5))
    frame = pd.read_csv(directory / "oof_predictions.csv")
    required = {"window_id", *METADATA, *PREDICTIONS}
    if not required.issubset(frame.columns) or frame.empty or not frame.window_id.is_unique:
        raise ValueError(f"Incomplete or repeated OOF predictions: {directory}")
    if frame.duplicated(["traj_id", "win_idx"]).any():
        raise ValueError(f"Repeated trajectory/window pairs: {directory}")
    used_folds = frame[["traj_id", "platform", "fold"]].drop_duplicates().sort_values("traj_id").reset_index(drop=True)
    expected = run["folds"]
    platform = run["config"]["platform"]
    if platform is not None:
        expected = expected[expected.platform.eq(platform)].reset_index(drop=True)
    pd.testing.assert_frame_equal(used_folds, expected[["traj_id", "platform", "fold"]])
    metrics = pd.read_csv(directory / "fold_metrics.csv")
    required_metrics = {"fold", "step", "train_windows", "train_steps", "windows_seen", *COMPUTE}
    if not required_metrics.issubset(metrics.columns) or not metrics.fold.is_unique or set(metrics.fold) != set(range(5)):
        raise ValueError(f"Incomplete fold compute measurements: {directory}")
    expected_train_windows = metrics.fold.map(lambda fold: int(frame.fold.ne(fold).sum()))
    steps = metrics.fold.map(expected_steps)
    if not (metrics.step.eq(steps).all() and metrics.train_steps.eq(steps).all()
            and metrics.windows_seen.eq(steps * run["config"]["batch_size"]).all()
            and metrics.train_windows.eq(expected_train_windows).all()):
        raise ValueError(f"Training counters do not match the fixed budget: {directory}")
    if "fold_budgets" in run and platform is None:
        windows = metrics.fold.map(run["fold_budgets"].generalist_train_windows)
        if not metrics.train_windows.eq(windows).all():
            raise ValueError("Reference generalist windows differ from the OOF folds")
    if (metrics.train_windows < run["config"]["batch_size"]).any():
        raise ValueError("Training set is too small for a full batch")
    values = metrics[COMPUTE].to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or (values < 0).any() or (metrics.validation_calls < 1).any()
            or not metrics.validation_calls.eq(np.floor(metrics.validation_calls)).all()):
        raise ValueError(f"Invalid measured compute counters: {directory}")
    return frame, metrics


def compute_totals(budget):
    """Sums of measured model-fold prefixes; eval time is kept separate."""
    result = {}
    for group, rows in budget.groupby("group", sort=True):
        result[group] = {"model_folds": len(rows),
                         "train_steps_across_folds": int(rows.train_steps.sum()),
                         "windows_seen_across_folds": int(rows.windows_seen.sum()),
                         **{column: int(rows[column].sum()) if column == "validation_calls"
                            else float(rows[column].sum()) for column in COMPUTE}}
    return result


def compare_scenario(generalist, specialists, budgets, regime, output, reference_epochs=None,
                     specialist_budget=None):
    generalist, specialists = align_oof(generalist, specialists)
    gm = velocity_metrics(generalist, generalist[PREDICTIONS].to_numpy())
    sm = velocity_metrics(specialists, specialists[PREDICTIONS].to_numpy())
    per_platform = pd.DataFrame([
        {"platform": p, "generalist_ave": gm["per_platform"][p], "specialist_ave": sm["per_platform"][p]}
        for p in PLATFORMS])
    per_platform["delta"] = per_platform.specialist_ave - per_platform.generalist_ave
    fold_results = []
    for fold in range(5):
        g, s = generalist[generalist.fold.eq(fold)], specialists[specialists.fold.eq(fold)]
        fold_results.append({"fold": fold,
                             "generalist_ave": velocity_metrics(g, g[PREDICTIONS].to_numpy())["macro_ave"],
                             "specialist_ave": velocity_metrics(s, s[PREDICTIONS].to_numpy())["macro_ave"]})
    fold_results = pd.DataFrame(fold_results)
    fold_results["delta"] = fold_results.specialist_ave - fold_results.generalist_ave
    trajectories = pd.DataFrame(gm["per_trajectory"]).rename(columns={"ave": "generalist_ave"}).merge(
        pd.DataFrame(sm["per_trajectory"]).rename(columns={"ave": "specialist_ave"}),
        on=["platform", "traj_id"], validate="one_to_one")
    trajectories["delta"] = trajectories.specialist_ave - trajectories.generalist_ave
    budget = pd.concat(budgets, ignore_index=True)
    generalist_steps = budget[budget.group.eq("generalist")].set_index("fold").train_steps
    specialist_steps = budget[budget.group.eq("specialists")].pivot(index="fold", columns="model", values="train_steps")
    if regime == "per_model":
        expected_total = generalist_steps * len(PLATFORMS)
    elif specialist_budget is not None:
        if not specialist_steps.eq(specialist_budget).all().all():
            raise ValueError("Every specialist must match its requested update budget")
        expected_total = pd.Series(len(PLATFORMS) * specialist_budget, index=generalist_steps.index)
    else:
        expected_total = generalist_steps
    if not specialist_steps.sum(axis=1).eq(expected_total).all():
        raise ValueError("Specialist update totals do not match the comparison budget")
    if regime == "per_model" and not specialist_steps.eq(generalist_steps, axis=0).all().all():
        raise ValueError("Every specialist must match the generalist update budget")
    ratios = specialist_steps.sum(axis=1) / generalist_steps
    group_ratio = float(specialist_steps.to_numpy().sum() / generalist_steps.sum())
    summary = {
        "generalist_steps_by_fold": {str(fold): int(steps) for fold, steps in generalist_steps.items()},
        "specialist_steps_by_fold": {str(fold): {p: int(row[p]) for p in PLATFORMS}
                                     for fold, row in specialist_steps.iterrows()},
        "generalist": {"oof_macro_ave": gm["macro_ave"],
                       "fold_mean_ave": float(fold_results.generalist_ave.mean()),
                       "fold_std_ave": float(fold_results.generalist_ave.std(ddof=1))},
        "specialists": {"oof_macro_ave": sm["macro_ave"],
                        "fold_mean_ave": float(fold_results.specialist_ave.mean()),
                        "fold_std_ave": float(fold_results.specialist_ave.std(ddof=1))},
        "paired_fold_delta_mean": float(fold_results.delta.mean()),
        "paired_fold_delta_std": float(fold_results.delta.std(ddof=1)),
        "specialists_better_folds": int((fold_results.delta < 0).sum()),
        "specialists_better_trajectories": int((trajectories.delta < 0).sum()),
        "n_windows": len(generalist), "n_trajectories": len(trajectories),
        "specialist_group_update_ratio": group_ratio,
        "specialist_group_update_ratio_by_fold": {str(fold): float(ratio) for fold, ratio in ratios.items()},
        "measured_compute_totals": compute_totals(budget),
        "notes": ["Final snapshot at the fixed budget; best.pt is not used for this comparison.",
                  "Delta = specialists minus generalist; negative favors specialists.",
                  "Specialists use known platform labels locally, unavailable on test, and have 4x total parameters.",
                  f"The specialist group uses {group_ratio:g}x the generalist's summed updates across folds.",
                  "Specialist snapshots share a training prefix; they are not independent training runs.",
                  "Compute totals are measured accumulated prefixes summed across model-folds, not FLOPs or overall scenario wall time.",
                  "Training, evaluation and normalization times are separate; shared run/loading overhead is recorded in shared_runs.json.",
                  "Fold std is descriptive, not a confidence interval. AVE only; no ATE20, official-val or challenge score."]}
    if reference_epochs is not None:
        summary["reference_epochs"] = reference_epochs
        summary["notes"].append(f"Each generalist budget equals {reference_epochs} complete passes with full batches; specialist budgets are measured in updates.")
    else:
        summary["generalist_steps_per_fold"] = int(generalist_steps.iloc[0])
        summary["specialist_steps_each_per_fold"] = int(specialist_steps.iloc[0, 0])
    output.mkdir(parents=True, exist_ok=False)
    per_platform.to_csv(output / "per_platform.csv", index=False)
    fold_results.to_csv(output / "fold_metrics.csv", index=False)
    trajectories.to_csv(output / "per_trajectory.csv", index=False)
    budget.to_csv(output / "budget.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def compare(generalist_dir, specialist_dirs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    generalist_run = read_run(generalist_dir)
    config = generalist_run["config"]
    if config["platform"] is not None or set(generalist_run["folds"].platform) != set(PLATFORMS):
        raise ValueError("Generalist must cover all four platforms")
    settings = {k: v for k, v in config.items() if k not in ("output_dir", "platform")}
    runs = []
    for platform, directory in zip(PLATFORMS, specialist_dirs, strict=True):
        run = read_run(directory)
        if run["config"]["platform"] != platform:
            raise ValueError(f"Expected {platform} specialist in {directory}")
        if {k: v for k, v in run["config"].items() if k not in ("output_dir", "platform")} != settings:
            raise ValueError(f"Training hyperparameters differ for {platform}")
        pd.testing.assert_frame_equal(generalist_run["folds"], run["folds"])
        if generalist_run["environment"]["index_sha256"] != run["environment"]["index_sha256"]:
            raise ValueError(f"Dataset index checksums differ for {platform}")
        if "fold_budgets" in generalist_run:
            pd.testing.assert_frame_equal(generalist_run["fold_budgets"][["generalist_train_windows", "generalist_steps"]],
                                          run["fold_budgets"][["generalist_train_windows", "generalist_steps"]])
        runs.append(run)
    reference = config.get("reference_epochs")
    generalist, generalist_budget = read_snapshot(generalist_run, "final" if reference else config["max_steps"])
    generalist_budget = generalist_budget.assign(model="generalist", group="generalist", batch_size=256)
    comparisons = {}
    if reference:
        scenarios = ((f"per_model_{reference}passes", "final", "per_model"),
                     (f"total_{reference}passes", "equal_total", "total"))
    else:
        maximum, lower = config["max_steps"], config["snapshot_steps"][0]
        lower_name = f"total_{maximum}" if len(PLATFORMS) * lower == maximum else f"specialists_{lower}"
        scenarios = ((f"per_model_{maximum}", maximum, "per_model"), (lower_name, lower, "lower"))
    for name, step, regime in scenarios:
        frames, budgets = [], [generalist_budget]
        for platform, run in zip(PLATFORMS, runs, strict=True):
            frame, budget = read_snapshot(run, step)
            frames.append(frame)
            budgets.append(budget.assign(model=platform, group="specialists", batch_size=256))
        # Validate every snapshot before creating reports.
        specialists = pd.concat(frames, ignore_index=True)
        align_oof(generalist, specialists)
        comparisons[name] = (specialists, budgets, regime, None if reference else step)
    output.mkdir(parents=True, exist_ok=False)
    shared = {"note": "Scenario snapshots share training prefixes. Actual run overhead below is not charged twice.",
              "runs": {str(run["directory"]): run["run_compute"] for run in [generalist_run, *runs]}}
    (output / "shared_runs.json").write_text(json.dumps(shared, indent=2, allow_nan=False) + "\n")
    for name, (specialists, budgets, regime, specialist_budget) in comparisons.items():
        summary = compare_scenario(generalist, specialists, budgets, regime, output / name,
                                   reference, specialist_budget)
        print(name, json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--generalist", default="runs/mlp_10k_cv")
    parser.add_argument("--specialists", nargs=4,
                        default=[f"runs/mlp_10k_{p}" for p in PLATFORMS],
                        metavar=("CAR", "DOG", "DRONE", "HUMAN"))
    parser.add_argument("--output", default="reports/comparison_10k")
    args = parser.parse_args()
    compare(args.generalist, args.specialists, args.output)
    config = json.loads((Path(args.generalist) / "config.json").read_text())
    if config.get("reference_epochs") is None:
        from src.evaluation.plot_comparison import plot_comparison
        plot_comparison(args.generalist, args.specialists, args.output)


if __name__ == "__main__":
    main()
