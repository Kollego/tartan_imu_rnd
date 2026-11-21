"""Plot matched five-fold learning curves and final OOF comparisons."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PLATFORMS


def read_history(directory):
    frame = pd.read_csv(Path(directory) / "history.csv")
    columns = ["fold", "step", "cv_macro_ave"]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Missing validation history columns: {directory}")
    frame = frame[columns].sort_values(["fold", "step"])
    if (frame.empty or frame.duplicated(["fold", "step"]).any()
            or set(frame.fold) != set(range(5))
            or not np.isfinite(frame.to_numpy(dtype=float)).all()
            or (frame.step <= 0).any() or (frame.cv_macro_ave < 0).any()
            or not frame.groupby("step").fold.nunique().eq(5).all()):
        raise ValueError(f"Expected complete, unique five-fold histories: {directory}")
    return frame.set_index(["fold", "step"]).cv_macro_ave


def learning_curves(generalist_dir, specialist_dirs):
    """Each curve equally weights folds; specialist folds equally weight domains."""
    generalist = read_history(generalist_dir)
    domains = []
    for platform, directory in zip(PLATFORMS, specialist_dirs, strict=True):
        curve = read_history(directory)
        if not generalist.index.equals(curve.index):
            raise ValueError(f"Generalist and {platform} validation steps/folds differ")
        domains.append(curve)
    specialist = pd.concat(domains, axis=1).mean(axis=1)
    curves = pd.DataFrame({"generalist": generalist, "specialists": specialist})
    result = curves.groupby(level="step").agg(["mean", "std"])
    result.columns = [f"{group}_{stat}" for group, stat in result.columns]
    return result.reset_index()


def plot_comparison(generalist_dir, specialist_dirs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    config = json.loads((Path(generalist_dir) / "config.json").read_text())
    maximum, lower = config["max_steps"], config["snapshot_steps"][0]
    ratio = len(PLATFORMS) * lower / maximum
    lower_name = f"total_{maximum}" if ratio == 1 else f"specialists_{lower}"
    full = json.loads((output / f"per_model_{maximum}" / "summary.json").read_text())
    short = json.loads((output / lower_name / "summary.json").read_text())
    curves = learning_curves(generalist_dir, specialist_dirs)
    if not {lower, maximum}.issubset(set(curves.step)):
        raise ValueError("Validation histories must include both requested snapshots")
    curves.to_csv(output / "learning_curves.csv", index=False)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.55, 1]})
    colors = {"generalist": "#3b6fb6", "specialists": "#df8b36"}
    for group, label in (("generalist", "Генералист"), ("specialists", "Специалисты")):
        mean, std = curves[f"{group}_mean"], curves[f"{group}_std"]
        left.plot(curves.step, mean, label=label, color=colors[group], linewidth=2)
        left.fill_between(curves.step, np.maximum(0, mean - std), mean + std,
                          color=colors[group], alpha=0.13, linewidth=0)
    point = curves.loc[curves.step.eq(lower), "specialists_mean"].iloc[0]
    left.axvline(lower, color="0.6", linewidth=1, linestyle="--")
    left.scatter([lower], [point], color=colors["specialists"], zorder=3)
    left.set(title="Обучение: среднее по 5 фолдам", xlabel="Обновления весов одной модели", ylabel="CV macro AVE, м/с ↓")
    left.legend(frameon=False)
    left.grid(alpha=0.2)
    values = [full["generalist"]["oof_macro_ave"], short["specialists"]["oof_macro_ave"],
              full["specialists"]["oof_macro_ave"]]
    bars = right.bar([0, 1, 2], values,
                     color=[colors["generalist"], "#edbb87", colors["specialists"]], width=0.65)
    right.set_xticks([0, 1, 2], [f"Генералист\n{maximum:,} шагов", f"Специалисты\nпо {lower:,}",
                                f"Специалисты\nпо {maximum:,}"])
    right.bar_label(bars, fmt="%.4f", padding=4)
    right.set(title="Итог: все OOF-предсказания", ylabel="OOF macro AVE, м/с ↓", ylim=(0, max(values) * 1.2))
    right.grid(axis="y", alpha=0.2)
    right.set_axisbelow(True)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    caption = (f"Полоса: ±1 std по фолдам. Четыре специалиста по {lower:,} шагов: "
               f"{ratio:g}× суммарных обновлений генералиста; по {maximum:,}: 4×.\n"
               f"Точка {lower:,} — сохранённый этап того же обучения специалистов.")
    fig.text(0.5, 0.015, caption, ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
