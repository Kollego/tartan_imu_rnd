"""Compare two generalists on the drone flight with median baseline OOF AVE."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exps.hw2.results import OUT, PRED, expected_rows, read_run, validate
from exps.hw2.study import config_for
from src.data import IMUDataset


def plot_window(row, predictions, target):
    data_root = config_for("temporal_gate", "all")["data_root"]
    imu = IMUDataset(data_root, "train", row).x[0]
    seconds = np.arange(len(imu)) / 200
    fig = plt.figure(figsize=(11.5, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1])
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
    def decimal(value):
        return f"{value:.2f}".replace(".", ",").replace("-", "−")

    cells = [["Разметка", *map(decimal, target), "—"]]
    for label, vector in predictions:
        error = np.linalg.norm(vector - target)
        cells.append([label, *map(decimal, vector), decimal(error)])
    ax.text(0, .82, "Кто точнее на этой секунде?", weight="bold", fontsize=14)
    ax.text(0, .72, "Сравните прогноз со строкой разметки. Все числа в м/с.", fontsize=10)
    table = ax.table(cellText=cells, colLabels=["", "vx", "vy", "vz", "Ошибка ↓"],
                     cellLoc="center", colWidths=[.31, .15, .15, .15, .24], bbox=[0, .34, 1, .32])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e0eeee" if r == 0 or c == 4 else "#f0f4f5")
        if c == 0:
            cell.get_text().set_ha("left")
        if c == 4:
            cell.get_text().set_weight("bold")
    ax.text(0, .20, "Ошибка — длина разности двух векторов скорости.", fontsize=10)
    fig.suptitle(f"Одинаковый вход для обеих моделей · {row.traj_id.iloc[0]} · окно {row.win_idx.iloc[0]}", fontsize=14)
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
                          ("GRU + gate", forecasts["temporal_gate"][idx])], target[idx])

    gains = flight.mlp_plain_error - flight.temporal_gate_error
    folds = pd.read_csv(OUT / "folds.csv")
    cv = folds[folds.role.eq("generalist")].pivot(index="fold", columns="variant", values="ave")
    old, new = cv["mlp_plain"], cv["temporal_gate"]

    def summary(before, after):
        change = 100 * (after - before) / before
        return f"{before:.3f} → {after:.3f} м/с  ({change:+.1f}%)".replace(".", ",").replace("-", "−")

    fig, (local, overall) = plt.subplots(1, 2, figsize=(11.8, 4.5), layout="constrained",
                                         gridspec_kw={"width_ratios": [1.45, 1]})
    teal, rust = "#087678", "#a94f31"
    local.bar(flight.win_idx, gains, width=.82, color=np.where(gains >= 0, teal, rust))
    local.axhline(0, color="#263746", lw=1)
    limit = gains.abs().max() * 1.24
    local.set(ylim=(-limit, limit), xlabel="Секунда полёта", ylabel="Ошибка MLP минус ошибка GRU + gate, м/с")
    local.set_title("Один полёт · средняя ошибка вектора\n"
                    + summary(flight.mlp_plain_error.mean(), flight.temporal_gate_error.mean()),
                    loc="left", fontsize=12, pad=12)
    for idx in [gains.idxmax(), gains.idxmin()]:
        second, gain = flight.at[idx, "win_idx"], gains.at[idx]
        label = f"{second} с · {gain:+.2f}".replace(".", ",").replace("-", "−")
        local.annotate(label, (second, gain), xytext=(0, 5 if gain > 0 else -5),
                       textcoords="offset points", ha="center", va="bottom" if gain > 0 else "top",
                       color=teal if gain > 0 else rust, fontsize=9, weight="bold")
    local.text(.02, .96, f"GRU + gate точнее · {(gains > 0).sum()} с", transform=local.transAxes,
               va="top", color=teal, fontsize=10)
    local.text(.02, .04, f"MLP точнее · {(gains < 0).sum()} с", transform=local.transAxes,
               color=rust, fontsize=10)

    overall.hlines(cv.index, new, old, color="#aab6b8", lw=2)
    overall.scatter(new, cv.index, color=teal, marker="D", label="GRU + gate", s=44, zorder=3)
    overall.scatter(old, cv.index, color="#414b56", label="MLP", s=48, zorder=3)
    overall.set(yticks=cv.index, ylabel="Фолд", xlabel="Macro AVE, м/с · меньше лучше")
    overall.invert_yaxis()
    overall.margins(y=.20, x=.12)
    overall.legend(loc="best", frameon=False, ncol=2, fontsize=9)
    overall.set_title("Пять фолдов · средняя macro AVE\n" + summary(old.mean(), new.mean()),
                      loc="left", fontsize=12, pad=12)
    for ax in (local, overall):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "flight_before_after.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
