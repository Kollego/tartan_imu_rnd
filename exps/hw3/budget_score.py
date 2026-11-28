"""Compare fixed-budget temporal_gate_p25 checkpoints on the HW2 folds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exps.baseline.train_steps import write_json
from exps.hw2.results import PRED, expected_rows, read_run, validate
from exps.hw2.study import ROOT
from exps.hw3.audit_score import SCORER_COMMIT, SCORER_SHA256, attach_pose, load_scorer, summarize
from exps.hw3.bootstrap import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, TRAJECTORY, interval, macro_mean
from src.data.imu import TARGETS


VARIANT = "temporal_gate_p25"
STEPS = (10_000, 20_000, 30_000, 50_000)
RUN = ROOT / ("steps_" + "_".join(map(str, STEPS))) / VARIANT
METRICS = ["ave", "ate20", "score"]


def score_checkpoints(scorer):
    expected = expected_rows()
    poses = attach_pose(expected, "train", (100, 199))
    drone_poses = poses[poses.platform.eq("drone")]
    scopes = {"generalist": {}, "generalist_drone": {}, "specialist_drone": {}}
    for domain, reference in (("all", poses), ("drone", drone_poses)):
        historical = validate(read_run(VARIANT, domain)[0], reference)
        for step in STEPS:
            predictions = pd.concat([
                pd.read_csv(RUN / domain / f"fold_{fold}/step_{step}_oof.csv")
                for fold in range(5)
            ], ignore_index=True)
            predictions = validate(predictions, reference)
            if step == STEPS[0]:
                np.testing.assert_allclose(predictions[PRED], historical[PRED], rtol=1e-6, atol=1e-6)
            windows = reference.merge(predictions[["window_id", *PRED]], on="window_id", validate="one_to_one")
            _, trajectories = summarize(scorer, windows)
            if domain == "all":
                scopes["generalist"][step] = trajectories[METRICS]
            drone = windows[windows.platform.eq("drone")]
            target = drone[TARGETS].to_numpy()
            fast = drone[np.linalg.norm(target, axis=1) >= 2].copy()
            fast["fast_ave"] = np.linalg.norm(fast[TARGETS].to_numpy() - fast[PRED].to_numpy(), axis=1)
            fast_ave = fast.groupby(TRAJECTORY).fast_ave.mean()
            drone_trajectories = trajectories.xs("drone", level="platform", drop_level=False)[METRICS]
            scope = "generalist_drone" if domain == "all" else "specialist_drone"
            scopes[scope][step] = drone_trajectories.join(fast_ave)
    return scopes


def metric_rows(frame):
    for metric, values in frame.items():
        values = values.dropna().to_frame()
        estimate = values.groupby(level=["fold", "platform"]).mean().groupby(level="fold").mean().mean().iloc[0]
        low, high = interval(macro_mean(values)[:, 0])
        yield dict(metric=metric, value=estimate, ci95_low=low, ci95_high=high, trajectories=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    args = parser.parse_args()
    scopes = score_checkpoints(load_scorer(args.scorer))
    summary, comparisons, folds = [], [], []
    for scope, results in scopes.items():
        baseline = results[STEPS[0]]
        for step, trajectories in results.items():
            pd.testing.assert_index_equal(trajectories.index, baseline.index)
            summary.extend(dict(scope=scope, step=step, **row) for row in metric_rows(trajectories))
            comparisons.extend(dict(comparison="vs_10000", scope=scope, step=step, **row)
                               for row in metric_rows(trajectories - baseline))
            by_fold = trajectories.groupby(level=["fold", "platform"]).mean().groupby(level="fold").mean()
            folds.append(by_fold.reset_index().assign(scope=scope, step=step))
    for step in STEPS:
        generalist = scopes["generalist_drone"][step]
        specialist = scopes["specialist_drone"][step]
        pd.testing.assert_index_equal(specialist.index, generalist.index)
        comparisons.extend(dict(comparison="specialist_minus_generalist", scope="drone", step=step, **row)
                           for row in metric_rows(specialist - generalist))
    summary = pd.DataFrame(summary)
    summary.to_csv(RUN / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(RUN / "comparisons.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(RUN / "folds.csv", index=False)
    write_json(RUN / "evaluation.json", dict(
        variant=VARIANT, steps=STEPS, split="TartanIMU train, out of fold",
        split_seed=42, training_seeds=list(range(42, 47)), historical_10000_oof_verified=True,
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        scorer_commit=SCORER_COMMIT, scorer_sha256=SCORER_SHA256, quaternion_offset=100,
        position_offset=199, fast_speed_min_mps=2,
    ))
    print(summary.pivot(index=["scope", "metric"], columns="step", values="value").round(5).to_string())


if __name__ == "__main__":
    main()
