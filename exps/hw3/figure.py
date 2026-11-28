"""Static overview of the HW3 analyses for the repository README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


PLATFORMS = {"car": "Автомобиль", "dog": "Робособака", "drone": "Дрон", "human": "Человек"}
TEAL = "#086b5d"
ORANGE = "#a54426"
INK = "#172a27"


def decimal(value, precision):
    return f"{value:.{precision}f}".replace(".", ",").replace("-", "−")


def draw_slices(ax, rows, labels, title, max_x):
    for i, row in enumerate(rows):
        ave, bounds = row["ave"], row["ave_ci95"]
        ax.barh(i, ave, color=TEAL, height=0.52)
        if bounds:
            ax.hlines(i - 0.34, *bounds, color=ORANGE, linewidth=2)
        ax.text(ave + max_x * 0.02, i, decimal(ave, 3), color=INK, va="center", fontsize=9)
    ax.set(xlim=(0, max_x), ylim=(len(rows) - 0.5, -0.62),
           yticks=range(len(rows)), yticklabels=labels, xlabel="AVE, м/с",
           xticks=np.linspace(0, max_x, 6 if max_x == 1 else 5))
    ax.set_title(title, loc="left", weight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal(value, 1)))
    ax.grid(axis="x", color="#e0e8e1")


def save_overview(data, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), layout="constrained")
    fig.suptitle("TartanIMU · Ошибка оценки скорости", color=INK, weight="bold")
    for ax in axes.flat:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#a8b9ad")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelcolor=INK)
    platforms = sorted(data["platforms"], key=lambda r: -r["ave"])
    draw_slices(axes[0], platforms, [PLATFORMS[r["label"]] for r in platforms], "Платформы", 1)
    draw_slices(axes[1], data["speeds"], [r["label"] for r in data["speeds"]], "Скорость, м/с", 2)
    fig.supxlabel("Отрезки показывают 95% ДИ. Для быстрого среза показана точечная оценка.", fontsize=10, color=INK)
    fig.savefig(destination, dpi=145)
    plt.close(fig)
