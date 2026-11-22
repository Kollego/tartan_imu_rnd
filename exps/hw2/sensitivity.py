"""Summarize paired patch-size and training-budget experiments on the HW1 folds."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from exps.hw2.results import OUT, PRED, expected_rows, read_run, validate
from exps.hw2.study import PATCH_SIZES, PROJECT, ROOT, config_for
from src.data import PLATFORMS
from src.metrics import velocity_metrics
from src.model import build_model

REFERENCE = "temporal_gate"
BUDGETS = (2500, 5000, 10000)
ROLES = {"generalist": ("all",), "specialists": PLATFORMS}


def measurements():
    expected = expected_rows()
    per_run = []
    for variant, patch in {REFERENCE: 50, **PATCH_SIZES}.items():
        kwargs = config_for(variant, "all")["model_kwargs"]
        params = sum(p.numel() for p in build_model("window", **kwargs).parameters())
        for role, platforms in ROLES.items():
            for platform in platforms:
                frame, history = read_run(variant, platform)
                target = expected if platform == "all" else expected[expected.platform.eq(platform)]
                frame = validate(frame, target)
                selected = history[history.step.isin(BUDGETS)].set_index(["fold", "step"]).sort_index()
                keys = pd.MultiIndex.from_product([range(5), BUDGETS], names=["fold", "step"])
                pd.testing.assert_index_equal(selected.index, keys)
                for fold, group in frame.groupby("fold"):
                    actual = velocity_metrics(group, group[PRED].to_numpy())["macro_ave"]
                    np.testing.assert_allclose(actual, selected.loc[(fold, 10000), "cv_macro_ave"], atol=1e-7)
                per_run.append(selected.cv_macro_ave.reset_index().assign(
                    variant=variant, role=role, platform=platform, patches=200 // patch,
                    patch_samples=patch, params_per_model=params))
    data = pd.concat(per_run, ignore_index=True)
    # For specialists each platform contributes 1/4, exactly as in fold macro AVE.
    keys = ["variant", "patches", "patch_samples", "params_per_model", "role", "step", "fold"]
    folds = data.groupby(keys, as_index=False).cv_macro_ave.mean()
    reference = folds[folds.variant.eq(REFERENCE) & folds.step.eq(10000)]
    folds = folds.merge(reference[["role", "fold", "cv_macro_ave"]].rename(
        columns={"cv_macro_ave": "reference_ave"}), on=["role", "fold"], validate="many_to_one")
    folds["delta_vs_reference"] = folds.cv_macro_ave - folds.reference_ave
    summary = folds.groupby(keys[:-1], as_index=False).agg(
        cv_mean=("cv_macro_ave", "mean"), cv_std=("cv_macro_ave", "std"),
        delta_vs_reference=("delta_vs_reference", "mean"),
        folds_better_than_reference=("delta_vs_reference", lambda x: (x < 0).sum()))
    folds.to_csv(OUT / "sensitivity_folds.csv", index=False)
    summary.to_csv(OUT / "sensitivity.csv", index=False)
    return summary, folds


def plot(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained", sharey=True)
    for ax, (role, title) in zip(axes, [("generalist", "Генералист"), ("specialists", "Специалисты")]):
        table = summary[summary.role.eq(role)].pivot(index="patches", columns="step", values="cv_mean")
        values = table.to_numpy()
        # A separate scale per role makes the within-role sensitivity visible.
        ax.imshow(values, cmap="Blues_r", aspect="auto", vmin=values.min(), vmax=values.max())
        for (i, j), value in np.ndenumerate(values):
            dark = value < (values.min() + values.max()) / 2
            ax.text(j, i, f"{value:.4f}".replace(".", ","), ha="center", va="center",
                    color="white" if dark else "#102939", fontsize=14)
        i, j = np.unravel_index(values.argmin(), values.shape)
        ax.add_patch(Rectangle((j-.48, i-.46), .96, .92, color="#f3ac32", fill=False, linewidth=3))
        ax.set_xticks(range(len(BUDGETS)), ["2 500", "5 000", "10 000"])
        ax.set_yticks(range(len(table)), [f"{n} × {1000 // n} мс" for n in table.index])
        ax.set_xlabel("Шаги обучения на модель")
        ax.set_title(title, fontweight="bold")
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Участки в одной секунде")
    fig.suptitle("Средняя CV macro AVE, м/с · чем меньше, тем лучше\n"
                 "Внутри панели темнее — точнее. Рамка — минимум из девяти сочетаний.", fontsize=13)
    fig.savefig(OUT / "sensitivity.png", dpi=170)
    plt.close(fig)


def plot_fold_effects(folds):
    comparisons = [(2, 10000, "2 участка"), (8, 10000, "8 участков"),
                   (4, 2500, "2 500 шагов"), (4, 5000, "5 000 шагов")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained", sharex=True, sharey=True)
    for ax, (role, title, color) in zip(axes, [("generalist", "Генералист", "#086b83"),
                                                ("specialists", "Специалисты", "#993e15")]):
        for i, (patches, steps, _) in enumerate(comparisons):
            rows = folds[(folds.role == role) & (folds.patches == patches) & (folds.step == steps)]
            values = rows.sort_values("fold").delta_vs_reference
            ax.scatter(values, i + np.linspace(-.15, .15, 5), s=28, color=color, alpha=.75)
            ax.plot(values.mean(), i, marker="D", markerfacecolor="white",
                    markeredgecolor="#152f3b", markersize=7)
        ax.axvline(0, color="#526470", linewidth=1, zorder=0)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(title)
        ax.set_xlabel("Изменение CV macro AVE, м/с")
    axes[0].set_yticks(range(len(comparisons)), [row[2] for row in comparisons])
    axes[0].invert_yaxis()
    fig.suptitle("Разница с 4 участками и 10 000 шагов · левее нуля — точнее\n"
                 "Каждая точка — один фолд, ромб — среднее пяти фолдов", fontsize=13)
    fig.savefig(OUT / "sensitivity_deltas.png", dpi=170)
    plt.close(fig)


def publish():
    from clearml import Task
    from exps.baseline.train_steps import write_json

    links = []
    for variant in (REFERENCE, *PATCH_SIZES):
        for platform in ("all", *PLATFORMS):
            marker = ROOT / "clearml" / variant / platform / "completed.json"
            record = json.loads(marker.read_text())
            links.append(dict(variant=variant, platform=platform, url=record["url"]))
    directory = ROOT / "clearml" / "sensitivity"
    directory.mkdir(exist_ok=True)
    task_file = directory / "task.json"
    if task_file.exists():
        task = Task.get_task(task_id=json.loads(task_file.read_text())["id"])
    else:
        task = Task.create(project_name=PROJECT, task_name="Patch size and training budget / paired CV",
                           task_type="data_processing", detect_repository=False, add_task_init_call=False)
        write_json(task_file, {"id": task.id})
    links.append(dict(variant="sensitivity", platform="summary", url=task.get_output_log_web_page()))
    pd.DataFrame(links).to_csv(OUT / "sensitivity_clearml.csv", index=False)
    task.mark_started(force=True)
    task.add_tags(["HW2", "sensitivity"])
    task.set_comment("Can finer temporal resolution help the GRU? Does training for longer improve "
                     "the generalist and the specialists equally? Each comparison reuses the same folds.")
    task.set_parameters_as_dict({"patch_samples": [100, 50, 25], "training_steps": list(BUDGETS),
                                 "seed_by_fold": "42 + fold", "reference": "4 patches, step 10000",
                                 "metric": "five-fold mean macro AVE, m/s"})
    logger = task.get_logger()
    logger.report_table("Sensitivity", "Five-fold means", iteration=0,
                        table_plot=pd.read_csv(OUT / "sensitivity.csv"))
    logger.report_image("Sensitivity", "Patches and steps", iteration=0, local_path=str(OUT / "sensitivity.png"))
    logger.report_image("Sensitivity", "Paired fold effects", iteration=0,
                        local_path=str(OUT / "sensitivity_deltas.png"))
    for name in ("sensitivity.csv", "sensitivity_folds.csv", "sensitivity_clearml.csv", "sensitivity.png", "sensitivity_deltas.png"):
        if not task.upload_artifact(name, OUT / name, wait_on_upload=True, retries=2):
            raise RuntimeError(f"Upload failed: {name}")
    task.flush(wait_for_uploads=True)
    task.mark_completed(ignore_errors=False)
    task.reload()
    if str(task.status) != "completed":
        raise RuntimeError(f"Unexpected ClearML status: {task.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--publish", action="store_true", help="Also sync the comparison table and plot to ClearML")
    args = parser.parse_args()
    summary, folds = measurements()
    plot(summary)
    plot_fold_effects(folds)
    if args.publish:
        publish()
