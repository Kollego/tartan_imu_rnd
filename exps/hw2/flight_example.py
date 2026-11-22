"""Compare two generalists on the drone flight with median baseline OOF AVE."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exps.hw2.results import OUT, PRED, expected_rows, read_run, validate
from exps.hw2.study import config_for
from src.data import IMUDataset


def plot_window(row, vectors):
    data_root = config_for("temporal_gate", "all")["data_root"]
    imu = IMUDataset(data_root, "train", row).x[0]
    seconds = np.arange(len(imu)) / 200
    fig = plt.figure(figsize=(10.8, 4.7), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.32, 1])
    for i, (title, unit) in enumerate([("Акселерометр", "м/с²"), ("Гироскоп", "рад/с")]):
        ax = fig.add_subplot(grid[i, 0])
        for j, (axis, color) in enumerate(zip("XYZ", ["#087984", "#bb6026", "#6b5795"])):
            ax.plot(seconds, imu[:, 3 * i + j], color=color, lw=1.3, label=axis)
        for boundary in [0.25, 0.50, 0.75]:
            ax.axvline(boundary, color="#888888", ls=":", lw=1)
        ax.set(xlim=(0, 1), xticks=[0, .25, .50, .75, 1], ylabel=unit)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        if i:
            ax.set_xlabel("Время от начала окна, с")
        ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[:, 1])
    ax.set_axis_off()
    ax.text(0, .80, "Скорость по осям платформы", weight="bold", fontsize=13)
    ax.text(0, .72, "Средняя за эту секунду, м/с", fontsize=10)
    cells = [[label, *(f"{v:.2f}".replace(".", ",").replace("-", "−") for v in values)]
             for label, values in vectors]
    table = ax.table(cellText=cells, colLabels=["", "vx", "vy", "vz"],
                     cellLoc="center", colWidths=[.43, .19, .19, .19], bbox=[0, .28, 1, .34])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e0eeee" if r == 0 else "#f0f4f5")
        if c == 0:
            cell.get_text().set_ha("left")
    fig.suptitle(f"Одна секунда полёта · {row.traj_id.iloc[0]} · окно {row.win_idx.iloc[0]}", fontsize=14)
    fig.savefig(OUT / "imu_window.png", dpi=170)
    plt.close(fig)


def main():
    expected = expected_rows()
    rows = expected.sort_values("window_id").reset_index(drop=True)
    target = rows[["vx", "vy", "vz"]].to_numpy()
    windows = rows[["traj_id", "platform", "fold", "win_idx"]].copy()
    windows["target_speed"] = np.linalg.norm(target, axis=1)
    forecasts = {}
    for name in ("mlp_plain", "temporal_gate"):
        predictions = validate(read_run(name, "all")[0], expected)[PRED].to_numpy()
        forecasts[name] = predictions
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
    # Window 31 is discussed in the report to distinguish speed from vector error.
    example = rows[rows.traj_id.eq("drone_train_0224") & rows.win_idx.eq(31)]
    idx = example.index.item()
    plot_window(example, [("MLP", forecasts["mlp_plain"][idx]),
                          ("GRU + gate", forecasts["temporal_gate"][idx]), ("Разметка", target[idx])])

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
