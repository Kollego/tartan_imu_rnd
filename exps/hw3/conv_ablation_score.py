"""Verify the non-overlapping control and score the Conv1d/GRU grid."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from exps.baseline.train_steps import write_json
from exps.hw2.results import PRED, expected_rows, validate
from exps.hw3.audit_score import SCORER_COMMIT, SCORER_SHA256, attach_pose, load_scorer, summarize
from exps.hw3.bootstrap import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from exps.hw3.budget_score import METRICS, metric_rows
from exps.hw3.conv_ablation import CONTROL, CONTROL_RUN, GEOMETRIES, RUN, convolution_config
from exps.hw3.encoder_ablation_score import checkpoint_protocol
from exps.hw3.epoch_score import slices
from exps.hw3.fixed_generalist import STEPS
from src.data import IMUDataset
from src.inference import predict
from src.model import build_model


def fold_ready(root, fold):
    folder = root / f"fold_{fold}"
    return all((folder / f"step_{step}.pt").is_file() and (folder / f"step_{step}_oof.csv").is_file()
               for step in STEPS)


def prepare_control(output, rows):
    config = convolution_config(CONTROL)
    output.mkdir(parents=True, exist_ok=True)
    missing = [fold for fold in config["folds"] if not fold_ready(output, fold)]
    if not missing:
        return
    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    for fold in missing:
        val = rows[rows.fold.eq(fold)].reset_index(drop=True)
        dataset = IMUDataset(config["data_root"], "train", val)
        destination = output / f"fold_{fold}"
        with TemporaryDirectory(prefix=f"control_{fold}_", dir=output) as temporary:
            directory = Path(temporary) / destination.name
            directory.mkdir()
            errors = {}
            for step in STEPS:
                source = CONTROL_RUN / f"fold_{fold}"
                state = torch.load(source / f"step_{step}.pt", map_location="cpu", weights_only=True)
                kernel, filters = 20, 64
                expected = {"encoder": "temporal", "mixing": "gate", "patch_size": kernel, "patch_filters": filters}
                if state["model_kwargs"] != expected:
                    raise ValueError("The original patch control has an unexpected geometry")
                weights = state["state_dict"]
                key = "encoder.patch_conv.weight"
                weights[key] = weights[key].reshape(filters, kernel, 6).transpose(1, 2).contiguous()
                state["model_kwargs"] = config["model_kwargs"]
                state["config"] = dict(state["config"], model_kwargs=config["model_kwargs"], output_dir=str(output))
                model = build_model(state["model_name"], **state["model_kwargs"])
                model.load_state_dict(weights)
                velocity = predict(model.eval(), dataset)
                original = validate(pd.read_csv(source / f"step_{step}_oof.csv"), val).set_index("window_id")
                original = original.loc[val.window_id, PRED].to_numpy()
                np.testing.assert_allclose(velocity, original, rtol=1e-5, atol=3e-6)
                errors[step] = float(np.max(np.abs(velocity - original)))
                oof = val[["window_id", "platform", "traj_id", "win_idx", "vx", "vy", "vz", "fold"]]
                oof.assign(**dict(zip(PRED, velocity.T))).to_csv(directory / f"step_{step}_oof.csv", index=False)
                torch.save(state, directory / f"step_{step}.pt")
            write_json(directory / "manifest.json", dict(converted_from=str(CONTROL_RUN / destination.name),
                       weight_layout="[filters, kernel, channels] to [filters, channels, kernel]",
                       max_abs_oof_delta_mps=errors))
            directory.rename(destination)
        print(f"Verified control fold {fold}: maximum prediction delta {max(errors.values()):.2e} m/s", flush=True)


def load_geometry(root, geometry, rows, step, protocols):
    expected = convolution_config(geometry)
    predictions = []
    for fold in expected["folds"]:
        directory = root / f"fold_{fold}"
        val = rows[rows.fold.eq(fold)]
        state = torch.load(directory / f"step_{step}.pt", weights_only=True)
        model, protocol = checkpoint_protocol(state)
        if ((state["fold"], state["step"], state["seed"]) != (fold, step, expected["seed"] + fold)
                or set(state["validation_trajectories"]) != set(val.traj_id)
                or set(state["train_trajectories"]) != set(rows.loc[rows.fold.ne(fold), "traj_id"])
                or model != expected["model_kwargs"]):
            raise ValueError(f"Mismatched checkpoint for {geometry}, fold {fold}, step {step}")
        if geometry == CONTROL:
            protocols[fold] = protocol
        elif protocol != protocols[fold]:
            raise ValueError(f"Training protocol differs from the control for {geometry}, fold {fold}")
        pred = validate(pd.read_csv(directory / f"step_{step}_oof.csv"), val)
        predictions.append(pred[["window_id", *PRED]])
    return pd.concat(predictions, ignore_index=True)


def comparisons_for(geometries):
    for geometry in geometries:
        if geometry != CONTROL:
            yield "vs_control", CONTROL, geometry
    for fixed, comparison in ((0, "stride"), (1, "kernel_size")):
        for value in (50, 20, 10):
            ordered = sorted((g for g in geometries if GEOMETRIES[g][fixed] == value),
                             key=lambda g: GEOMETRIES[g][1 - fixed], reverse=True)
            for reference, geometry in zip(ordered, ordered[1:]):
                yield comparison, reference, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path)
    parser.add_argument("--output", type=Path, default=RUN)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--control-only", action="store_true")
    mode.add_argument("--available", action="store_true", help="Score the geometries that have completed all folds")
    args = parser.parse_args()
    if args.scorer is None and not args.control_only:
        parser.error("--scorer is required to score the grid")
    rows = expected_rows()
    prepare_control(args.output / CONTROL, rows)
    if args.control_only:
        return
    scorer = load_scorer(args.scorer)
    poses = attach_pose(rows, "train", (100, 199))
    layouts = (CONTROL, *(g for g in GEOMETRIES if g != CONTROL))
    geometries = [g for g in layouts if all(fold_ready(args.output / g, fold) for fold in range(5))]
    if not args.available and len(geometries) != len(layouts):
        raise ValueError(f"Missing completed geometries: {set(layouts) - set(geometries)}")
    summary, comparisons, models = [], [], []
    for geometry in geometries:
        kwargs = convolution_config(geometry)["model_kwargs"]
        model = build_model("window", **kwargs)
        kernel, stride = GEOMETRIES[geometry]
        right_padding = model.encoder.right_padding
        models.append(dict(geometry=geometry, kernel_size=kernel, stride=stride, right_padding=right_padding,
                           gru_steps=1 + (200 + right_padding - kernel) // stride,
                           parameters=sum(p.numel() for p in model.parameters()), converted_control=geometry == CONTROL))
    for step in STEPS:
        protocols, scores = {}, {}
        for geometry in geometries:
            predictions = load_geometry(args.output / geometry, geometry, rows, step, protocols)
            scored = poses.merge(predictions, on="window_id", validate="one_to_one")
            _, trajectories = summarize(scorer, scored)
            scores[geometry] = trajectories[METRICS]
            for platform, subset in slices(scores[geometry]):
                summary.extend(dict(step=step, geometry=geometry, platform=platform, **row)
                               for row in metric_rows(subset))
        for comparison, reference, geometry in comparisons_for(geometries):
            pd.testing.assert_index_equal(scores[geometry].index, scores[reference].index)
            for platform, delta in slices(scores[geometry] - scores[reference]):
                comparisons.extend(dict(step=step, comparison=comparison, geometry=geometry, reference=reference,
                                        platform=platform, **row) for row in metric_rows(delta))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output / "comparisons.csv", index=False)
    pd.DataFrame(models).to_csv(args.output / "models.csv", index=False)
    write_json(args.output / "evaluation.json", dict(
        steps=STEPS, models=models, filters=64, gru_hidden_size=64, loss="velocity MSE",
        control=str(CONTROL_RUN), padding="minimal right zero-padding after normalization to cover all samples",
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
