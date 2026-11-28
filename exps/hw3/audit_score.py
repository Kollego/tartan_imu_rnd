"""Check the local HW2 submission and compute the full score on labelled data."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from exps.hw2.results import PRED, expected_rows, read_run, validate
from exps.hw2.study import FOLDS
from exps.hw3.bootstrap import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, TRAJECTORY, interval, macro_mean
from src.data import IMUDataset, make_folds, read_index
from src.data.imu import TARGETS
from src.inference import load_model, predict


DATA = Path("data/tartan-imu-challenge-iros2026")
FINAL = Path("runs/hw2/final")
CHECKPOINT = FINAL / "fold_0/step_10000.pt"
SCORER_COMMIT = "8a8025000f85d2c6b0692b88e15c2c91e761f7d7"
SCORER_SHA256 = "84bfe63500909326a3189be7326b167d4d54bb0fad9098056148630bdb9e0704"
QUAT = ["qx", "qy", "qz", "qw"]
POSITION = ["gx", "gy", "gz"]
VARIANTS = ("mlp_plain", "temporal_gate", "temporal_gate_p25")
POSE_OFFSETS = (0, 100, 199)


def load_scorer(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SCORER_SHA256:
        raise ValueError(f"Expected scorer from upstream commit {SCORER_COMMIT}")
    spec = importlib.util.spec_from_file_location("tartanimu_scorer", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def attach_pose(rows, split, offsets):
    """Build local scorer inputs from the labelled 200 Hz pose arrays."""
    outputs = []
    for (platform, trajectory), group in rows.groupby(["platform", "traj_id"]):
        with np.load(DATA / split / platform / f"{trajectory}.npz", allow_pickle=False) as archive:
            start = 200 * group.win_idx.to_numpy()
            pos, quat = archive["pos"], archive["quat"]
            part = group[["window_id"]].copy()
            part["dt"] = 200 / float(archive["fs"])
            for offset in offsets:
                part[[f"{key}{offset}" for key in POSITION]] = pos[start + offset]
                part[[f"{key}{offset}" for key in QUAT]] = quat[start + offset]
            outputs.append(part)
    return rows.merge(pd.concat(outputs), on="window_id", validate="one_to_one")


def summarize(scorer, rows, q_offset=100, p_offset=199):
    solution = rows.rename(columns={**{f"{k}{q_offset}": k for k in QUAT},
                                    **{f"{k}{p_offset}": k for k in POSITION},
                                    **dict(zip(TARGETS, [f"{k}_gt" for k in TARGETS]))})
    submission = solution[["window_id", *PRED]].rename(columns=dict(zip(PRED, TARGETS)))
    scored = solution.rename(columns=dict(zip(PRED, [f"{k}_pred" for k in TARGETS])))
    records = []
    for keys, group in scored.groupby(TRAJECTORY):
        ordered = group.sort_values("win_idx")
        records.append(dict(zip(TRAJECTORY, keys), windows=len(group),
                            ave=scorer._ave_traj(ordered), ate20=scorer._ate_traj(ordered)))
    trajectories = pd.DataFrame(records).set_index(TRAJECTORY)
    if not np.isfinite(trajectories[["ave", "ate20"]]).all().all():
        raise ValueError("Non-finite trajectory score")
    trajectories["score"] = (scorer.W_AVE * trajectories.ave / scorer.AVE_REF
                             + scorer.W_ATE * trajectories.ate20 / scorer.ATE_REF)
    fold_platform = trajectories[["ave", "ate20", "score"]].groupby(level=["fold", "platform"]).mean()
    folds = fold_platform.groupby(level="fold").mean()
    for fold, part in solution.groupby("fold"):
        np.testing.assert_allclose(
            scorer.score(part.drop(columns=PRED), submission[submission.window_id.isin(part.window_id)], "window_id"),
            folds.loc[fold, "score"], rtol=1e-12,
        )
    return dict(macro=folds.mean().to_dict(), folds=folds.reset_index().to_dict("records"),
                platforms=fold_platform.groupby(level="platform").mean().to_dict("index")), trajectories


def verify_export():
    local = FINAL / "submission.csv"
    with TemporaryDirectory() as directory:
        rebuilt = Path(directory) / "submission.csv"
        subprocess.run([sys.executable, "-m", "scripts.predict", "--checkpoint", str(CHECKPOINT),
                        "--output", str(rebuilt)], check=True)
        payload = local.read_bytes()
        if payload != rebuilt.read_bytes():
            raise ValueError("The rebuilt CSV differs from the saved HW2 export")
    sha256 = hashlib.sha256(payload).hexdigest()
    if sha256 != json.loads((FINAL / "verified.json").read_text())["submission_sha256"]:
        raise ValueError("The export differs from the HW2 manifest")
    return dict(path=str(local), sha256=sha256, windows=len(pd.read_csv(local)), byte_identical=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/hw3/score-audit.json"))
    args = parser.parse_args()
    scorer = load_scorer(args.scorer)
    torch.set_num_threads(4)
    val = read_index(DATA, "val").assign(fold=0)
    train = expected_rows()
    pd.testing.assert_frame_equal(pd.read_csv(FOLDS), make_folds(train, 5, 42))
    model, checkpoint = load_model(CHECKPOINT)
    if (set(checkpoint["train_trajectories"]) != set(train.traj_id)
            or set(checkpoint["validation_trajectories"]) != set(val.traj_id)
            or set(checkpoint["train_trajectories"]) & set(val.traj_id)):
        raise ValueError("The checkpoint does not match the official train/val split")
    pred = pd.read_csv(FINAL / "val_predictions.csv")[["window_id", *PRED]]
    if set(pred.window_id) != set(val.window_id):
        raise ValueError("Saved val predictions must cover all official val windows")
    val = val.merge(pred, on="window_id", validate="one_to_one")
    np.testing.assert_allclose(predict(model, IMUDataset(DATA, "val", val)), val[PRED].to_numpy(),
                               rtol=1e-5, atol=1e-5)
    val = attach_pose(val, "val", POSE_OFFSETS)
    val_summary, val_trajectories = summarize(scorer, val)
    val_summary["score_ci95"] = interval(macro_mean(val_trajectories[["score"]])[:, 0])
    val_summary["pose_sensitivity"] = [
        dict(quaternion_offset=q, position_offset=p, **summarize(scorer, val, q, p)[0]["macro"])
        for q, p in ((0, 0), (100, 100), (100, 199), (199, 199))
    ]
    val_summary["zero"] = summarize(scorer, val.assign(**dict.fromkeys(PRED, 0.0)))[0]["macro"]
    val_summary["target_as_prediction"] = summarize(
        scorer, val.assign(**{pred: val[target] for pred, target in zip(PRED, TARGETS)}))[0]["macro"]
    val_summary.update(windows=len(val), trajectories=len(val_trajectories), train_seed=checkpoint["seed"],
                       checkpoint=str(CHECKPOINT), checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())

    cv = attach_pose(train, "train", (100, 199))
    cv_results, cv_trajectories = {}, {}
    for variant in VARIANTS:
        rows = validate(read_run(variant, "all")[0], train)
        checked = cv.merge(rows[["window_id", *PRED]], on="window_id", validate="one_to_one")
        cv_results[variant], cv_trajectories[variant] = summarize(scorer, checked)
    delta = cv_trajectories["temporal_gate_p25"][["score"]] - cv_trajectories["temporal_gate"][["score"]]
    result = dict(
        scorer_commit=SCORER_COMMIT, scorer_sha256=SCORER_SHA256,
        quaternion_offset=100, position_offset=199, samples_per_window=200,
        val=val_summary, cv=cv_results, cv_windows=len(train), cv_trajectories=train.traj_id.nunique(),
        cv_fold_seed=42, cv_train_seeds=list(range(42, 47)),
        cv_score_delta_p25_minus_original=dict(
            estimate=cv_results["temporal_gate_p25"]["macro"]["score"] - cv_results["temporal_gate"]["macro"]["score"],
            ci95=interval(macro_mean(delta)[:, 0])),
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES, submission=verify_export(),
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
