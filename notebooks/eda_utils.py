"""Small data checks used by the EDA notebook; no model predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

PLATFORMS = ["car", "dog", "drone", "human"]
TARGETS = ["vx", "vy", "vz"]
WINDOW = 200


def load_indices(data):
    """Join each supplied index to its targets by window_id."""
    frames = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(Path(data) / "index" / f"{split}_windows.csv")
        required = ["window_id", "traj_id", "win_idx"]
        assert frame.window_id.is_unique
        assert not frame.duplicated(["traj_id", "win_idx"]).any()
        if split != "test":
            required.append("platform")
            targets = pd.read_csv(Path(data) / "index" / f"{split}_targets.csv")
            assert targets.window_id.is_unique
            assert set(targets.window_id) == set(frame.window_id)
            frame = frame.merge(targets, on="window_id", validate="one_to_one", sort=False)
            assert np.isfinite(frame[TARGETS].to_numpy()).all()
        assert not frame[required].isna().any().any()
        frames[split] = frame
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert set(frames[a].traj_id).isdisjoint(frames[b].traj_id)
        assert set(frames[a].window_id).isdisjoint(frames[b].window_id)
    return frames


def audit_trajectories(data, frames):
    """Check all NPZ and collect one row of signal statistics per trajectory."""
    result = []
    for split, frame in frames.items():
        paths = sorted((Path(data) / split).rglob("*.npz"))
        assert len(paths) == frame.traj_id.nunique()
        for path in paths:
            rows = frame[frame.traj_id.eq(path.stem)].sort_values("win_idx")
            with np.load(path, allow_pickle=False) as archive:
                imu, ts = archive["imu"], archive["ts"]
                assert imu.shape == (len(ts), 6), path
                assert np.all(np.diff(ts) > 0), path
                n_windows = len(imu) // WINDOW
                assert np.array_equal(rows.win_idx.to_numpy(), np.arange(n_windows)), path
                nonfinite = sum(int((~np.isfinite(archive[key])).sum()) for key in archive.files)
                assert nonfinite == 0, path
                error = np.nan
                if split != "test":
                    assert archive["vel_body"].shape == (len(ts), 3), path
                    assert archive["pos"].shape == (len(ts), 3), path
                    assert archive["quat"].shape == (len(ts), 4), path
                    targets = archive["vel_body"][:n_windows * WINDOW].reshape(-1, WINDOW, 3).mean(axis=1)
                    error = float(np.abs(targets - rows[TARGETS].to_numpy()).max())
                    assert error < 2e-6, path
                acc = np.linalg.norm(imu[:, :3], axis=1)
                gyro = np.linalg.norm(imu[:, 3:], axis=1)
                means = imu[:, :3].mean(axis=0, dtype=np.float64)
                result.append({
                    "split": split, "platform": path.parent.name if split != "test" else "unknown",
                    "traj_id": path.stem, "samples": len(imu), "tail_samples": len(imu) % WINDOW,
                    "nonfinite": nonfinite, "target_max_abs_error": error,
                    "acc_max": float(acc.max()), "gyro_max": float(gyro.max()),
                    "acc_mean_x": means[0], "acc_mean_y": means[1], "acc_mean_z": means[2],
                })
    return pd.DataFrame(result)
