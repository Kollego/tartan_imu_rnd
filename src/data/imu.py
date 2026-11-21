"""Load only IMU features; pose and platform labels never enter the model."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

PLATFORMS = ("car", "dog", "drone", "human")
TARGETS = ["vx", "vy", "vz"]


def _validate_frame(frame):
    required = {"window_id", "traj_id", "win_idx"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing index columns: {required - set(frame.columns)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Window index must be nonempty and have no missing identifiers")
    if not frame.window_id.is_unique or frame.duplicated(["traj_id", "win_idx"]).any():
        raise ValueError("Window IDs and trajectory/window pairs must be unique")
    for column in ("window_id", "win_idx"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain nonnegative integers")
    if "platform" in frame:
        if not frame.platform.isin(PLATFORMS).all():
            raise ValueError(f"platform must be one of {PLATFORMS}")
        if (frame.groupby("traj_id").platform.nunique() != 1).any():
            raise ValueError("Each trajectory must belong to exactly one platform")


def read_index(data_root, split, with_targets=True):
    """Read the supplied split and match its targets by unique window_id."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    index_root = Path(data_root) / "index"
    frame = pd.read_csv(index_root / f"{split}_windows.csv")
    _validate_frame(frame)
    if split != "test" and "platform" not in frame:
        raise ValueError("Labelled split is missing platform")
    if with_targets and split != "test":
        targets = pd.read_csv(index_root / f"{split}_targets.csv")
        if not {"window_id", *TARGETS}.issubset(targets.columns):
            raise ValueError("Targets must contain window_id, vx, vy, vz")
        if targets.window_id.isna().any() or not targets.window_id.is_unique:
            raise ValueError("Target window IDs must be present and unique")
        if set(frame.window_id) != set(targets.window_id):
            raise ValueError("Index and target window IDs do not match exactly")
        frame = frame.merge(targets[["window_id", *TARGETS]], on="window_id", how="left", validate="one_to_one", sort=False)
        if not np.isfinite(frame[TARGETS].to_numpy(dtype=np.float64)).all():
            raise ValueError("Velocity targets must be finite")
    return frame


def make_folds(frame, n_splits=5, seed=42):
    """Stratify whole trajectories by platform; never split adjacent windows."""
    _validate_frame(frame)
    if "platform" not in frame:
        raise ValueError("Fold construction requires known platforms")
    trajectories = frame[["traj_id", "platform"]].drop_duplicates().sort_values(["platform", "traj_id"]).reset_index(drop=True)
    if n_splits < 2 or (trajectories.platform.value_counts() < n_splits).any():
        raise ValueError("Each represented platform needs at least n_splits trajectories")
    trajectories["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(trajectories.traj_id, trajectories.platform)):
        trajectories.loc[validation, "fold"] = fold
    return trajectories[["traj_id", "platform", "fold"]]


def compute_normalization(x):
    """Channel statistics from training windows only; accumulate in float64."""
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (200, 6) or len(x) == 0:
        raise ValueError("Expected nonempty IMU array shaped (N, 200, 6)")
    if not np.isfinite(x).all():
        raise ValueError("IMU values must be finite")
    mean = np.mean(x, axis=(0, 1), dtype=np.float64)
    std = np.std(x, axis=(0, 1), dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics are not finite")
    return mean, np.maximum(std, 1e-6)


class IMUDataset(Dataset):
    """Preload requested windows in frame order. x has shape (N, 200, 6)."""

    def __init__(self, data_root, split, frame):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        _validate_frame(frame)
        if split != "test" and "platform" not in frame:
            raise ValueError("Train/val file lookup requires platform")
        self.frame = frame.reset_index(drop=True).copy()
        self.x = np.empty((len(frame), 200, 6), dtype=np.float32)
        for trajectory, rows in self.frame.groupby("traj_id", sort=False):
            # Identifiers come from the supplied index, never from model inputs.
            if Path(str(trajectory)).name != str(trajectory) or str(trajectory) in (".", ".."):
                raise ValueError("Invalid trajectory file identifier")
            base = Path(data_root) / split
            if split != "test":
                base = base / rows.platform.iloc[0]
            with np.load(base / f"{trajectory}.npz", allow_pickle=False) as archive:
                imu = archive["imu"]
                if imu.ndim != 2 or imu.shape[1] != 6:
                    raise ValueError(f"Invalid IMU shape in {trajectory}: {imu.shape}")
                for row in rows.itertuples():
                    start = int(row.win_idx) * 200
                    window = imu[start:start + 200]
                    if window.shape != (200, 6):
                        raise ValueError(f"Incomplete window {row.window_id} in {trajectory}")
                    self.x[row.Index] = window
        if not np.isfinite(self.x).all():
            raise ValueError("IMU windows must contain only finite float32 values")
        has_targets = [column in self.frame for column in TARGETS]
        if any(has_targets) and not all(has_targets):
            raise ValueError("Provide all three velocity target columns")
        self.y = self.frame[TARGETS].to_numpy(dtype=np.float32, copy=True) if all(has_targets) else None
        if self.y is not None and not np.isfinite(self.y).all():
            raise ValueError("Velocity targets must be finite float32 values")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        target = torch.from_numpy(self.y[index]) if self.y is not None else torch.empty(0, dtype=torch.float32)
        return torch.from_numpy(self.x[index]), target
