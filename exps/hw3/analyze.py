"""Rebuild the HW3 analyses from the fixed-step HW2 out-of-fold runs."""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch

from exps.hw2.results import OUT, PRED, expected_rows, read_run, validate
from exps.hw2.study import FOLDS, ROOT, config_for
from exps.hw3.architecture_study import RUN as ARCHITECTURE_STUDY
from exps.hw3.bootstrap import (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, TRAJECTORY,
                                interval, macro_mean, sufficiently_sampled, window_rate_intervals)
from exps.hw3.figure import save_overview
from src.data import IMUDataset, PLATFORMS, make_folds
from src.data.imu import TARGETS
from src.inference import load_model
from src.metrics import velocity_metrics
from src.model import build_model


HERE = Path(__file__).parent
SEED_RESULTS = HERE / "results/seed-p25/summary.csv"
SPEED_LABELS = ("< 0,1", "0,1–< 0,5", "0,5–< 2", "≥ 2")
BATCH_SIZE = 512
FOLD_SEED = 42
STEP = 10_000


def error_features(rows):
    target, prediction = rows[TARGETS].to_numpy(), rows[PRED].to_numpy()
    speed, predicted = np.linalg.norm(target, axis=1), np.linalg.norm(prediction, axis=1)
    norm_product = speed * predicted
    cosine = np.divide((target * prediction).sum(axis=1), norm_product,
                       out=np.full(len(rows), np.nan), where=norm_product > 0)
    ordered = rows.sort_values(["traj_id", "win_idx"])
    adjacent = ordered.traj_id.eq(ordered.traj_id.shift()) & ordered.win_idx.diff().eq(1)
    target_change = ordered[TARGETS].diff().pow(2).sum(axis=1).pow(0.5).where(adjacent)
    predicted_change = ordered[PRED].diff().pow(2).sum(axis=1).pow(0.5).where(adjacent)
    return rows.assign(error=np.linalg.norm(prediction - target, axis=1),
                       speed=speed, predicted=predicted,
                       angle=np.degrees(np.arccos(np.clip(cosine, -1, 1))),
                       target_change=target_change, predicted_change=predicted_change,
                       speed_group=pd.cut(speed, [0, 0.1, 0.5, 2, np.inf],
                                          right=False, labels=SPEED_LABELS))


def slice_metrics(rows, label):
    grouped = rows.groupby(TRAJECTORY, observed=True)[["error"]].mean()
    strata = grouped.groupby(level=["fold", "platform"])
    folds = strata.error.mean().groupby(level="fold").mean()
    ci = interval(macro_mean(grouped)[:, 0]) if sufficiently_sampled(grouped) else None
    return dict(label=label, n=len(rows), trajectories=len(grouped),
                folds=len(folds), min_stratum=int(strata.size().min()), ave=float(folds.mean()), ave_ci95=ci,
                window_ave=float(rows.error.mean()), p90=float(rows.error.quantile(0.90)),
                above_one=float(rows.error.ge(1).mean()))


def drone_speed_comparison(rows):
    """Paired change within drones with observations in both speed bands."""
    drone = rows[rows.platform.eq("drone") & rows.speed.ge(0.5)]
    means = drone.groupby([*TRAJECTORY, "speed_group"], observed=True).error.mean().unstack()
    matched = means.dropna(subset=list(SPEED_LABELS[-2:]))
    change = (matched[SPEED_LABELS[-1]] - matched[SPEED_LABELS[-2]]).to_frame("delta")
    fold_ave = matched.groupby(level="fold").mean().mean()
    return dict(n=len(matched), folds=int(matched.index.get_level_values("fold").nunique()),
                lower_ave=float(fold_ave[SPEED_LABELS[-2]]), faster_ave=float(fold_ave[SPEED_LABELS[-1]]),
                delta=float(change.groupby(level="fold").delta.mean().mean()),
                ci95=interval(macro_mean(change)[:, 0]))


def verify_oof(rows, variant, config):
    """Verify each checkpoint only on its own held-out IMU windows."""
    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    inputs = []

    with torch.inference_mode():
        for fold, part in rows.groupby("fold"):
            part = part.reset_index(drop=True)
            directory = ROOT / variant / "all" / f"fold_{fold}"
            checkpoint = directory / f"step_{STEP}.pt"
            model, state = load_model(checkpoint)
            if (state["fold"] != fold or state["step"] != STEP
                    or state["seed"] != config["seed"] + fold
                    or state["model_name"] != config["model_name"]
                    or state["model_kwargs"] != config["model_kwargs"]
                    or state["config"] != config):
                raise ValueError(f"Unexpected training configuration for {checkpoint}")
            validation_ids = set(part.traj_id)
            training_ids = set(rows.loc[rows.fold.ne(fold), "traj_id"])
            if (training_ids & validation_ids or set(state["validation_trajectories"]) != validation_ids
                    or set(state["train_trajectories"]) != training_ids):
                raise ValueError(f"Unexpected train/validation assignment for {checkpoint}")
            imu = IMUDataset(config["data_root"], "train", part).x
            std = model.std.numpy()
            inputs.append(dict(fold=int(fold), train_seed=state["seed"],
                               validation_windows=len(part), validation_trajectories=len(validation_ids),
                               train_windows=len(rows) - len(part), train_trajectories=len(training_ids),
                               checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                               oof=str(directory / "oof.csv"), oof_sha256=sha256_file(directory / "oof.csv"),
                               validation_imu_float32_sha256=hashlib.sha256(imu.astype("<f4", copy=False)).hexdigest(),
                               normalization_mean=model.mean.tolist(), normalization_std=std.tolist()))
            predicted = np.concatenate([model(torch.from_numpy(batch)).numpy()
                                        for batch in (imu[start:start + BATCH_SIZE]
                                                      for start in range(0, len(part), BATCH_SIZE))])
            np.testing.assert_allclose(predicted, part[PRED].to_numpy(), rtol=1e-5, atol=1e-5)
            print(f"Verified OOF for fold {fold}: {len(part)} windows", flush=True)
    return inputs


def typical_errors(rows, config):
    steady = rows.speed.lt(0.1)
    fast = rows.speed.ge(2)
    moving = rows.speed.ge(0.5)
    change = rows.target_change.ge(1)
    forecast_change = rows.predicted_change.le(0.5 * rows.target_change)
    definitions = [
        ("phantom", steady & rows.predicted.ge(0.3), steady),
        ("under", fast & rows.predicted.lt(0.7 * rows.speed) & rows.angle.lt(30), fast),
        ("direction", moving & rows.predicted.ge(0.25) & rows.angle.ge(90), moving),
        ("change", change & forecast_change & rows.error.ge(0.5), change),
    ]
    cases = {}
    counts = {}
    for name, match, population in definitions:
        counts[name] = pd.DataFrame({"match": match, "eligible": population})
        candidates = rows.loc[match].sort_values(["error", "window_id"])
        row = candidates.iloc[int(0.75 * (len(candidates) - 1))]
        trajectory = rows.loc[rows.traj_id.eq(row.traj_id)
                              & rows.win_idx.between(row.win_idx - 5, row.win_idx + 5)].sort_values("win_idx")
        axis = int(np.argmax(np.abs(row[TARGETS].to_numpy(dtype=float) - row[PRED].to_numpy(dtype=float))))
        current = rows.loc[[row.name]]
        imu = IMUDataset(config["data_root"], "train", current).x[0]
        context = [{"second": int(w.win_idx), "speed": w.speed, "predicted": w.predicted,
                    "targetAxis": getattr(w, TARGETS[axis]), "predictedAxis": getattr(w, PRED[axis])}
                   for w in trajectory.itertuples()]
        cases[name] = dict(**{key: int(row[key]) for key in ("window_id", "win_idx", "fold")},
                           **{key: row[key] for key in ("traj_id", "platform", "error", "speed", "predicted", "angle")},
                           target=row[TARGETS].tolist(), prediction=row[PRED].tolist(), context=context,
                           targetChange=None if pd.isna(row.target_change) else row.target_change,
                           predictedChange=None if pd.isna(row.predicted_change) else row.predicted_change,
                           axis="XYZ"[axis], n=len(candidates), eligible=int(population.sum()),
                           trajectories=int(candidates.traj_id.nunique()),
                           medianError=float(candidates.error.median()),
                           imu=imu[::2].round(4).tolist())
    fast_tail = rows.speed.ge(2)
    counts["fast_tail"] = pd.DataFrame({"match": fast_tail & rows.error.ge(1), "eligible": fast_tail})
    by_trajectory = pd.concat(counts, axis=1).groupby([rows[k] for k in TRAJECTORY]).sum()
    rates = window_rate_intervals(by_trajectory)
    for name, case in cases.items():
        case["rate_ci95"] = rates[name]
    return cases, rates["fast_tail"]


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(config, inputs):
    index = Path(config["data_root"]) / "index"
    paths = [FOLDS, OUT / "selection.json", index / "train_windows.csv", index / "train_targets.csv", Path("uv.lock"),
             Path(config["output_dir"]) / "manifest.json", HERE / "analyze.py", HERE / "bootstrap.py",
             HERE / "figure.py", HERE / "report.html"]
    return dict(split="TartanIMU train, HW2 trajectory holdouts (OOF)",
                fold_seed=FOLD_SEED, train_config=config, folds=inputs,
                inference=dict(device="cpu", batch_size=BATCH_SIZE, threads=config["num_threads"], deterministic=True),
                bootstrap=dict(generator="NumPy PCG64", seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                               interval="percentile 2.5%, 97.5%", strata="fold × platform", unit="whole trajectory",
                               scope="conditional on fixed trained models and folds"),
                files={os.path.relpath(path): sha256_file(path) for path in paths},
                runtime=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                             scikit_learn=sklearn.__version__, torch=str(torch.__version__), matplotlib=matplotlib.__version__,
                             platform=platform.system(), architecture=platform.machine()))


def flat_intervals(records, keys):
    flattened = []
    for record in records:
        row = dict(record)
        for key in keys:
            bounds = row.pop(key)
            row[f"{key}_lo"], row[f"{key}_hi"] = bounds if bounds is not None else (None, None)
        flattened.append(row)
    return flattened


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/hw3"))
    parser.add_argument("--architecture-study", type=Path, default=ARCHITECTURE_STUDY)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    selection = json.loads((OUT / "selection.json").read_text())
    variant = selection["variant"]
    config = config_for(variant, "all")
    rows = error_features(validate(read_run(variant, "all")[0], expected_rows()))
    pd.testing.assert_frame_equal(pd.read_csv(FOLDS), make_folds(rows, config["n_splits"], FOLD_SEED))
    overall = slice_metrics(rows, "Все окна")
    np.testing.assert_allclose(overall["ave"], selection["cv_mean"], atol=1e-9)
    folds = [slice_metrics(part, str(fold)) for fold, part in rows.groupby("fold")]
    platforms = [slice_metrics(rows[rows.platform.eq(p)], p) for p in PLATFORMS]
    speeds = [slice_metrics(rows[rows.speed_group.eq(s)], s) for s in SPEED_LABELS]
    intersections = [slice_metrics(part, [p, s]) for (p, s), part in
                     rows.groupby(["platform", "speed_group"], observed=True)]
    cases, fast_tail_ci = typical_errors(rows, config)
    inputs = verify_oof(rows, variant, config)
    manifest = provenance(config, inputs)
    zeros = [velocity_metrics(part, np.zeros((len(part), 3)))["macro_ave"]
             for _, part in rows.groupby("fold")]
    seed_results = pd.read_csv(SEED_RESULTS).query("platform == 'all'")
    data = dict(variant=variant, folds=folds, overall=overall, platforms=platforms,
                speeds=speeds, intersections=intersections, cases=cases,
                studyReport=os.path.relpath(args.architecture_study / "index.html", start=output),
                seedResults=seed_results.to_dict("records"),
                seedResultsFile=os.path.relpath(SEED_RESULTS, start=output),
                fastTailCI=fast_tail_ci, droneSpeed=drone_speed_comparison(rows), protocol=manifest,
                zeroAVE=float(np.mean(zeros)),
                parameters=sum(p.numel() for p in build_model("window", **config["model_kwargs"]).parameters()))
    links = pd.read_csv(OUT / "clearml.csv")
    data["clearml"] = links.loc[links.variant.eq(variant) & links.platform.eq("all"), "url"].item()
    slices = [
        {"platform": platform, "speed": speed, **{key: value for key, value in row.items() if key != "label"}}
        for platform, speed, row in [
            *((row["label"], "", row) for row in platforms),
            *(("", row["label"], row) for row in speeds),
            *((*row["label"], row) for row in intersections),
        ]
    ]
    pd.DataFrame(flat_intervals(slices, ("ave_ci95",))).to_csv(output / "slices.csv", index=False)
    case_rows = [dict(case=key, **{k: v for k, v in case.items() if k not in ("imu", "context")})
                 for key, case in cases.items()]
    pd.DataFrame(flat_intervals(case_rows, ("rate_ci95",))).to_csv(output / "error-cases.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (output / "analytics.json").write_text(payload + "\n")
    template = (HERE / "report.html").read_text()
    save_overview(data, output / "overview.png")
    (output / "index.html").write_text(template.replace("{{REPORT_DATA}}", payload.replace("<", "\\u003c")))
    print(f"Report saved to {(output / 'index.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
