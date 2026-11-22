"""Compare two generalists on the drone flight with median baseline OOF AVE."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exps.hw2.results import OUT, PRED, expected_rows, read_run, validate


def main():
    expected = expected_rows()
    rows = expected.sort_values("window_id").reset_index(drop=True)
    target = rows[["vx", "vy", "vz"]].to_numpy()
    windows = rows[["traj_id", "platform", "fold", "win_idx"]].copy()
    windows["target_speed"] = np.linalg.norm(target, axis=1)
    for name in ("mlp_plain", "temporal_gate"):
        predictions = validate(read_run(name, "all")[0], expected)[PRED].to_numpy()
        windows[f"{name}_speed"] = np.linalg.norm(predictions, axis=1)
        windows[f"{name}_error"] = np.linalg.norm(predictions - target, axis=1)
    errors = windows.groupby(["platform", "traj_id"])[["mlp_plain_error", "temporal_gate_error"]].mean()
    counts = errors.assign(improved=errors.temporal_gate_error < errors.mlp_plain_error)
    counts.groupby("platform").improved.agg(["sum", "size"]).rename(
        columns={"sum": "improved", "size": "total"}).to_csv(OUT / "trajectory_counts.csv")
    ranked = errors.loc["drone"].sort_values("mlp_plain_error", kind="stable")
    trajectory = ranked.index[len(ranked) // 2]
    flight = windows[windows.traj_id.eq(trajectory)].sort_values("win_idx")
    flight.to_csv(OUT / "flight_example.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True, sharey=True, layout="constrained")
    for ax, name, title, color in zip(
        axes, ("mlp_plain", "temporal_gate"), ("Было · MLP", "Стало · Conv1d + GRU с gate"), ("#92502e", "#087678")
    ):
        ax.plot(flight.win_idx, flight.target_speed, color="#263746", label="Истинная скорость", lw=2.2)
        ax.plot(flight.win_idx, flight[f"{name}_speed"], color=color, label="Прогноз", lw=1.8, ls="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Секунда полёта")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Модуль скорости, м/с")
    axes[0].set_ylim(bottom=0)
    fig.savefig(OUT / "flight_before_after.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
