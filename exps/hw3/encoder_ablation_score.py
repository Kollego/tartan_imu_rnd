"""Score the changes in patch count and filter count on the same HW2 trajectories."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from exps.baseline.train_steps import write_json
from exps.hw2.results import PRED, expected_rows, validate
from exps.hw3.audit_score import SCORER_COMMIT, SCORER_SHA256, attach_pose, load_scorer, summarize
from exps.hw3.bootstrap import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from exps.hw3.budget_score import METRICS, metric_rows
from exps.hw3.encoder_ablation import BASELINE, BASELINE_RUN, FILTERS, GEOMETRIES, PATCHES, RUN, ablation_config
from exps.hw3.epoch_score import slices
from exps.hw3.fixed_generalist import STEPS
from src.model import build_model


def checkpoint_protocol(state):
    model = dict(state["model_kwargs"])
    model.setdefault("patch_filters", 32)
    config = {key: value for key, value in state["config"].items() if key not in {"output_dir", "model_kwargs"}}
    return model, config


def score_geometry(root, geometry, rows, step, protocols):
    expected = ablation_config(geometry)
    predictions = []
    for fold in expected["folds"]:
        directory = root / f"fold_{fold}"
        val = rows[rows.fold.eq(fold)]
        snapshot = torch.load(directory / f"step_{step}.pt", weights_only=True)
        model, protocol = checkpoint_protocol(snapshot)
        if ((snapshot["fold"], snapshot["step"], snapshot["seed"]) != (fold, step, expected["seed"] + fold)
                or set(snapshot["validation_trajectories"]) != set(val.traj_id)
                or set(snapshot["train_trajectories"]) != set(rows.loc[rows.fold.ne(fold), "traj_id"])
                or model != expected["model_kwargs"]):
            raise ValueError(f"Mismatched checkpoint for {geometry}, fold {fold}, step {step}")
        if geometry == BASELINE:
            protocols[fold] = protocol
        elif protocol != protocols[fold]:
            raise ValueError(f"Training protocol differs from the control for {geometry}, fold {fold}")
        pred = validate(pd.read_csv(directory / f"step_{step}_oof.csv"), val)
        predictions.append(pred[["window_id", *PRED]])
    return pd.concat(predictions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--through", choices=(PATCHES, FILTERS), default=FILTERS)
    args = parser.parse_args()
    scorer = load_scorer(args.scorer)
    rows = expected_rows()
    poses = attach_pose(rows, "train", (100, 199))
    geometries = (BASELINE, PATCHES) if args.through == PATCHES else (BASELINE, PATCHES, FILTERS)
    summary, comparisons, models = [], [], []
    for geometry in geometries:
        kwargs = ablation_config(geometry)["model_kwargs"]
        model = build_model("window", **kwargs)
        size, filters = GEOMETRIES[geometry]
        models.append(dict(geometry=geometry, patches_per_window=200 // size, samples_per_patch=size,
                           patch_filters=filters, gru_hidden_size=model.encoder.gru.hidden_size,
                           parameters=sum(p.numel() for p in model.parameters())))
    for step in STEPS:
        protocols, scores = {}, {}
        for geometry in geometries:
            root = BASELINE_RUN if geometry == BASELINE else args.output / geometry
            predictions = score_geometry(root, geometry, rows, step, protocols)
            scored = poses.merge(predictions, on="window_id", validate="one_to_one")
            _, trajectories = summarize(scorer, scored)
            scores[geometry] = trajectories[METRICS]
            for platform, subset in slices(scores[geometry]):
                summary.extend(dict(step=step, geometry=geometry, platform=platform, **row)
                               for row in metric_rows(subset))
        for reference, treatment in zip(geometries, geometries[1:]):
            pd.testing.assert_index_equal(scores[treatment].index, scores[reference].index)
            for platform, delta in slices(scores[treatment] - scores[reference]):
                comparisons.extend(dict(step=step, geometry=treatment, reference=reference, platform=platform, **row)
                                   for row in metric_rows(delta))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output / "comparisons.csv", index=False)
    pd.DataFrame(models).to_csv(args.output / "models.csv", index=False)
    write_json(args.output / "evaluation.json", dict(
        steps=STEPS, models=models, loss="velocity MSE", baseline=str(BASELINE_RUN),
        split="TartanIMU train, out of fold", windows=len(rows), trajectories=rows.traj_id.nunique(),
        split_seed=42, train_seeds=list(range(42, 47)), bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, scorer_commit=SCORER_COMMIT, scorer_sha256=SCORER_SHA256,
        quaternion_offset=100, position_offset=199,
    ))
    print(pd.DataFrame(models).to_string(index=False))
    print(summary[summary.platform.eq("all")].pivot(index=["step", "metric"], columns="geometry",
                                                     values="value").round(5).to_string())


if __name__ == "__main__":
    main()
