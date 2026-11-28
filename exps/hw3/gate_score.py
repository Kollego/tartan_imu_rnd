"""Compare velocity metrics and measure platform prediction for the supervised gate."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from exps.baseline.train_steps import write_json
from exps.hw2.results import PRED, expected_rows, validate
from exps.hw2.study import config_for
from exps.hw3.audit_score import SCORER_COMMIT, SCORER_SHA256, attach_pose, load_scorer, summarize
from exps.hw3.bootstrap import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from exps.hw3.budget_score import METRICS, metric_rows
from exps.hw3.epoch_score import slices
from exps.hw3.epoch_train import output_for
from exps.hw3.gate_train import GATE_WEIGHT, RUN, STEPS, VARIANT
from src.data import IMUDataset, PLATFORMS
from src.inference import load_model


@torch.inference_mode()
def read_gate(checkpoint, dataset):
    model, state = load_model(checkpoint)
    batches = [model(imu, return_gate_logits=True) for imu, _ in DataLoader(dataset, batch_size=512)]
    velocity = torch.cat([velocity for velocity, _ in batches]).numpy()
    probabilities = torch.cat([logits for _, logits in batches]).softmax(dim=1).numpy()
    return velocity, probabilities, state


def check_pair(state, baseline, rows, val, fold, step, seed):
    expected = (fold, step, seed + fold)
    training = set(rows.loc[rows.fold.ne(fold), "traj_id"])
    validation = set(val.traj_id)
    for checkpoint in (state, baseline):
        if ((checkpoint["fold"], checkpoint["step"], checkpoint["seed"]) != expected
                or set(checkpoint["train_trajectories"]) != training
                or set(checkpoint["validation_trajectories"]) != validation):
            raise ValueError(f"Unexpected fold, step, seed, or trajectory split for fold {fold}, step {step}")
    ignored = {"output_dir", "gate_domain_weight"}
    configs = [{key: value for key, value in item["config"].items() if key not in ignored}
               for item in (state, baseline)]
    if configs[0] != configs[1] or state["config"]["gate_domain_weight"] != GATE_WEIGHT:
        raise ValueError("Training protocols differ beyond the gate domain loss")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    scorer = load_scorer(args.scorer)
    config = config_for(VARIANT, "all")
    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    rows = expected_rows()
    poses = attach_pose(rows, "train", (100, 199))
    dataset = IMUDataset(config["data_root"], "train", rows)
    baseline = output_for(steps=STEPS) / "all"
    summary, comparisons, gate_summary, confusion = [], [], [], []
    for step in STEPS:
        snapshots = {0.0: [], GATE_WEIGHT: []}
        gate_windows = []
        for fold in config["folds"]:
            indices = np.flatnonzero(rows.fold.eq(fold).to_numpy())
            val = rows.iloc[indices]
            for weight, root in ((0.0, baseline), (GATE_WEIGHT, args.output)):
                directory = root / f"fold_{fold}"
                pred = validate(pd.read_csv(directory / f"step_{step}_oof.csv"), val)
                snapshots[weight].append(pred[["window_id", *PRED]])
            checkpoint = f"fold_{fold}/step_{step}.pt"
            velocity, probabilities, state = read_gate(args.output / checkpoint, Subset(dataset, indices))
            reference = torch.load(baseline / checkpoint, map_location="cpu", weights_only=True)
            check_pair(state, reference, rows, val, fold, step, config["seed"])
            stored = snapshots[GATE_WEIGHT][-1].set_index("window_id").loc[val.window_id, PRED]
            np.testing.assert_allclose(velocity, stored, rtol=1e-6, atol=1e-6)
            codes = pd.Categorical(val.platform, categories=PLATFORMS).codes
            routed = np.asarray(PLATFORMS)[probabilities.argmax(axis=1)]
            gate_windows.append(val[["window_id", "platform", "fold"]].assign(
                prediction=routed, true_platform_weight=probabilities[np.arange(len(val)), codes],
            ))
        scores = {}
        for weight, predictions in snapshots.items():
            scored = poses.merge(pd.concat(predictions), on="window_id", validate="one_to_one")
            _, trajectories = summarize(scorer, scored)
            scores[weight] = trajectories[METRICS]
            for platform, subset in slices(scores[weight]):
                summary.extend(dict(step=step, gate_domain_weight=weight, platform=platform, **row)
                               for row in metric_rows(subset))
        pd.testing.assert_index_equal(scores[GATE_WEIGHT].index, scores[0.0].index)
        for platform, delta in slices(scores[GATE_WEIGHT] - scores[0.0]):
            comparisons.extend(dict(step=step, platform=platform, **row) for row in metric_rows(delta))
        gate = pd.concat(gate_windows, ignore_index=True)
        gate["correct"] = gate.platform.eq(gate.prediction)
        domain = gate.groupby("platform").agg(windows=("correct", "size"), accuracy=("correct", "mean"),
                                              mean_true_platform_weight=("true_platform_weight", "mean"))
        gate_summary.append(pd.concat([domain, pd.DataFrame({"windows": len(gate),
            "accuracy": gate.correct.mean(), "balanced_accuracy": domain.accuracy.mean(),
            "mean_true_platform_weight": gate.true_platform_weight.mean()}, index=["all"])])
                            .rename_axis("platform").reset_index().assign(step=step))
        confusion.append(gate.groupby(["platform", "prediction"]).size().rename("windows").reset_index().assign(step=step))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output / "comparisons.csv", index=False)
    pd.concat(gate_summary, ignore_index=True).to_csv(args.output / "gate_accuracy.csv", index=False)
    pd.concat(confusion, ignore_index=True).to_csv(args.output / "gate_confusion.csv", index=False)
    write_json(args.output / "evaluation.json", dict(
        variant=VARIANT, steps=STEPS, gate_domain_weight=GATE_WEIGHT, gate_platform_order=PLATFORMS,
        baseline=str(baseline), split="TartanIMU train, out of fold", windows=len(rows),
        trajectories=rows.traj_id.nunique(), split_seed=config["seed"],
        train_seeds=[config["seed"] + fold for fold in config["folds"]],
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        scorer_commit=SCORER_COMMIT, scorer_sha256=SCORER_SHA256, quaternion_offset=100, position_offset=199,
    ))
    print(summary[summary.platform.eq("all")].pivot(index=["step", "metric"], columns="gate_domain_weight",
                                                     values="value").round(5).to_string())
    print(pd.concat(gate_summary, ignore_index=True).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
