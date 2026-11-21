"""Reconstruct the paired CV table from complete, ID-checked OOF predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exps.hw2.study import FOLDS, ROOT, VARIANTS, config_for
from src.data import PLATFORMS, read_index
from src.metrics import velocity_metrics
from src.model import build_model

PRED = ["pred_vx", "pred_vy", "pred_vz"]
OUT = Path("exps/hw2/results")


def archive(platform):
    return Path("runs") / ("mlp_10k_" + ("cv" if platform == "all" else platform))


def is_archived(variant, platform):
    return (variant == "mlp_plain" and not (ROOT / variant / platform / "manifest.json").exists()
            and (archive(platform) / "step_10000/oof_predictions.csv").exists())


def read_run(variant, platform):
    if is_archived(variant, platform):
        directory = archive(platform)
        pd.testing.assert_frame_equal(pd.read_csv(FOLDS), pd.read_csv(directory / "folds.csv"))
        return (pd.read_csv(directory / "step_10000/oof_predictions.csv"),
                pd.read_csv(directory / "history.csv"))
    folds = [ROOT / variant / platform / f"fold_{fold}" for fold in range(5)]
    return tuple(pd.concat([pd.read_csv(f / file) for f in folds], ignore_index=True)
                 for file in ("oof.csv", "history.csv"))


def validate(frame, expected):
    cols = ["window_id", "platform", "traj_id", "win_idx", "fold", "vx", "vy", "vz"]
    frame = frame.sort_values("window_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[cols], expected.sort_values("window_id").reset_index(drop=True)[cols],
                                  check_exact=False, rtol=1e-8, atol=1e-8)
    if not np.isfinite(frame[PRED]).all().all():
        raise ValueError("Invalid OOF predictions")
    return frame


def expected_rows():
    config = config_for("mlp_plain", "all")
    return read_index(config["data_root"], "train").merge(pd.read_csv(FOLDS), on=["traj_id", "platform"],
                                                          validate="many_to_one")


def summarize():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = expected_rows()
    results, folds, domains, speeds, curves = [], [], [], [], []
    for variant in VARIANTS:
        config = config_for(variant, "all")
        model = build_model(config.get("model_name", "mlp"), **config.get("model_kwargs", {"hidden_sizes": [128, 64]}))
        n_params = sum(p.numel() for p in model.parameters())
        for role, platforms in [("generalist", ["all"]), ("specialists", PLATFORMS)]:
            pairs = [read_run(variant, p) for p in platforms]
            frame = validate(pd.concat([x[0] for x in pairs]), expected)
            history = pd.concat([x[1] for x in pairs], ignore_index=True)
            averaged = history.groupby(["fold", "step"]).cv_macro_ave.mean().reset_index()
            curves.extend(averaged.assign(variant=variant, role=role).to_dict("records"))
            per_fold = []
            for fold, group in frame.groupby("fold"):
                metric = velocity_metrics(group, group[PRED].to_numpy())["macro_ave"]
                per_fold.append(metric)
                folds.append(dict(variant=variant, role=role, fold=fold, ave=metric))
            metrics = velocity_metrics(frame, frame[PRED].to_numpy())
            for domain, ave in metrics["per_platform"].items():
                domains.append(dict(variant=variant, role=role, platform=domain, ave=ave))
            target = frame[["vx", "vy", "vz"]].to_numpy()
            speed = np.linalg.norm(target, axis=1)
            predicted = frame[PRED].to_numpy()
            for label, mask in [("stationary_<0.1", speed < 0.1), ("fast_>2", speed > 2)]:
                speeds.append(dict(variant=variant, role=role, speed_bin=label, windows=int(mask.sum()),
                                   ave=np.linalg.norm(predicted[mask]-target[mask], axis=1).mean(),
                                   target_speed=speed[mask].mean(),
                                   predicted_speed=np.linalg.norm(predicted[mask], axis=1).mean()))
            end = history[history.step.eq(10000)]
            if len(end) != 5 * len(platforms):
                raise ValueError("Incomplete fixed-step run")
            results.append(dict(variant=variant, role=role, cv_mean=np.mean(per_fold), cv_std=np.std(per_fold, ddof=1),
                                oof_ave=metrics["macro_ave"], params_per_model=n_params, models_per_fold=len(platforms),
                                updates=int(end.step.sum()), train_wall_s=end.train_seconds.sum()))
    for name, rows in [("experiments", results), ("folds", folds), ("domains", domains), ("speed", speeds),
                       ("curves", curves)]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    table = pd.DataFrame(results)
    winner = table[table.role.eq("generalist")].sort_values(["cv_mean", "variant"]).iloc[0]
    (OUT / "selection.json").write_text(json.dumps(dict(variant=winner.variant, cv_mean=winner.cv_mean,
                                                        criterion="minimum mean five-fold macro AVE, step 10000"),
                                                   indent=2) + "\n")
    print(table.round(5).to_string(index=False))


if __name__ == "__main__":
    summarize()
