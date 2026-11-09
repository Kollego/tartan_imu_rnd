"""Velocity error with the challenge's trajectory/platform aggregation."""

import numpy as np
import pandas as pd


def velocity_metrics(frame, predictions):
    """Predictions must be in frame row order; lower AVE is better (m/s)."""
    columns = {"window_id", "traj_id", "platform", "vx", "vy", "vz"}
    if not columns.issubset(frame.columns):
        raise ValueError(f"Missing metric columns: {columns - set(frame.columns)}")
    if frame.empty or frame[list(columns)].isna().any().any() or not frame.window_id.is_unique:
        raise ValueError("Metric rows need unique window IDs and complete metadata/targets")
    if not frame.platform.isin(("car", "dog", "drone", "human")).all():
        raise ValueError("Unknown platform in metric rows")
    if (frame.groupby("traj_id").platform.nunique() != 1).any():
        raise ValueError("Each trajectory must belong to one platform")
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    if predictions.shape != targets.shape or not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Predictions and targets must be finite arrays shaped (N, 3)")
    errors = np.linalg.norm(predictions - targets, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("Velocity errors are not finite")
    rows = frame[["platform", "traj_id"]].copy()
    rows["ave"] = errors
    per_trajectory = rows.groupby(["platform", "traj_id"], sort=True, as_index=False).ave.mean()
    per_platform = per_trajectory.groupby("platform", sort=True).ave.mean()
    return {
        "macro_ave": float(per_platform.mean()),
        "per_platform": {str(platform): float(value) for platform, value in per_platform.items()},
        "per_trajectory": per_trajectory.to_dict(orient="records"),
        "window_ave": float(errors.mean()),
    }
