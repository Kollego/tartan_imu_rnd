"""Check fixed-step OOF predictions against constant references and speed slices."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import velocity_metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", default="runs/mlp_10k_cv")
    parser.add_argument("--step", type=int, help="Optional numbered snapshot for older fixed-step runs")
    parser.add_argument("--output", default="reports/baseline_checks_10k")
    args = parser.parse_args()
    run = Path(args.run)
    if args.step is not None:
        snapshot = f"step_{args.step}"
    elif (run / "final").is_dir():
        snapshot = "final"
    else:
        snapshot = f"step_{json.loads((run / 'config.json').read_text())['max_steps']}"
    frame = pd.read_csv(run / snapshot / "oof_predictions.csv")
    if set(frame.fold) != set(range(5)):
        raise ValueError("Expected OOF predictions for all five folds")
    targets = frame[["vx", "vy", "vz"]].to_numpy()
    predicted = frame[["pred_vx", "pred_vy", "pred_vz"]].to_numpy()
    train_mean = np.empty_like(targets)
    for fold in range(5):
        validation = frame.fold.eq(fold).to_numpy()
        train_mean[validation] = targets[~validation].mean(axis=0)
    references = []
    for name, values in (("zero", np.zeros_like(targets)), ("train_mean", train_mean), ("model", predicted)):
        metrics = velocity_metrics(frame, values)
        references.append({"predictor": name, "macro_ave": metrics["macro_ave"],
                           "window_mse": float(np.square(values - targets).mean())})
    frame["true_speed"] = np.linalg.norm(targets, axis=1)
    frame["pred_speed"] = np.linalg.norm(predicted, axis=1)
    frame["error"] = np.linalg.norm(predicted - targets, axis=1)
    frame["speed_bin"] = pd.cut(frame.true_speed, [0, 0.1, 0.5, 1, 2, np.inf],
                                labels=["0-0.1", "0.1-0.5", "0.5-1", "1-2", ">2"], include_lowest=True)
    slices = frame.groupby("speed_bin", observed=True).agg(
        windows=("window_id", "size"), mean_error=("error", "mean"),
        mean_true_speed=("true_speed", "mean"), mean_pred_speed=("pred_speed", "mean"))
    spread = []
    for platform, rows in frame.groupby("platform"):
        for axis in ("x", "y", "z"):
            spread.append({"platform": platform, "axis": axis,
                           "target_std": float(rows[f"v{axis}"].std(ddof=0)),
                           "prediction_std": float(rows[f"pred_v{axis}"].std(ddof=0))})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(references).to_csv(output / "constant_references.csv", index=False)
    slices.to_csv(output / "errors_by_speed.csv")
    pd.DataFrame(spread).to_csv(output / "prediction_spread.csv", index=False)
    (output / "source.json").write_text(json.dumps({"run": args.run, "snapshot": snapshot,
        "notes": ["Constant train means are computed separately from the four training folds.",
                  "Speed slices and MSE are window-weighted diagnostics, not macro AVE.",
                  "A nonzero prediction spread alone does not prove a model is useful."]}, indent=2) + "\n")
    print(pd.DataFrame(references).to_string(index=False))
    print(slices.to_string())


if __name__ == "__main__":
    main()
