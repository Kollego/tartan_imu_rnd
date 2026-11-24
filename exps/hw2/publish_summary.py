"""Publish the selected full-train run and the paired study report to ClearML."""

import json

import pandas as pd
import torch
from clearml import Task

from exps.baseline.train_steps import write_json
from exps.hw2.results import OUT
from exps.hw2.study import PROJECT, ROOT, VARIANTS
from src.data import PLATFORMS


def main():
    final = ROOT / "final"
    marker = final / "clearml_completed.json"
    if marker.exists():
        return
    verified = json.loads((final / "verified.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    path = final / "clearml_task.json"
    if path.exists():
        task = Task.get_task(task_id=json.loads(path.read_text())["id"])
    else:
        task = Task.create(project_name=PROJECT, task_name="Selected generalist / full train and official val",
                           task_type="training", detect_repository=False, add_task_init_call=False)
        write_json(path, {"id": task.id, "url": task.get_output_log_web_page()})
    task.mark_started(force=True)
    task.add_tags(["HW2", "final", verified["variant"]])
    checkpoint = torch.load(final / "fold_0/final.pt", map_location="cpu", weights_only=True)
    task.set_parameters_as_dict({"Selection": selection, "Training": checkpoint["config"],
                                 "Refit": {"split": "all official train", "seed": 42, "steps": 10000,
                                           "val_evaluations": 1}})
    links = []
    for variant in VARIANTS:
        for platform in ("all", *PLATFORMS):
            record = json.loads((ROOT / "clearml" / variant / platform / "completed.json").read_text())
            links.append({"variant": variant, "platform": platform, "url": record["url"]})
    links.append({"variant": selection["variant"], "platform": "full train", "url": task.get_output_log_web_page()})
    pd.DataFrame(links).to_csv(OUT / "clearml.csv", index=False)
    logger = task.get_logger()
    history = pd.read_csv(final / "fold_0/history.csv").iloc[-1]
    logger.report_scalar("Train", "mean MSE over full fit", float(history.train_mse), 10000)
    logger.report_scalar("Official val", "macro AVE", verified["val_macro_ave"], 10000)
    for domain, metric in verified["val_per_platform"].items():
        logger.report_scalar("Official val by platform", domain, metric, 10000)
    for name in ["experiments", "paired", "domains", "speed"]:
        logger.report_table("Paired CV study", name, iteration=0, table_plot=pd.read_csv(OUT / f"{name}.csv"))
    for image in ["ablations", "learning"]:
        logger.report_image("HW2 figures", image, iteration=0, local_path=str(OUT / f"{image}.png"))
    for p in [final / "verified.json", final / "fold_0/final.pt", final / "fold_0/history.csv",
              OUT / "experiments.csv", OUT / "paired.csv", OUT / "folds.csv", OUT / "clearml.csv"]:
        if not task.upload_artifact(p.name, p, wait_on_upload=True, retries=2):
            raise RuntimeError(f"Upload failed for {p.name}")
    task.flush(wait_for_uploads=True)
    task.mark_completed(ignore_errors=False)
    task.reload()
    if str(task.status) != "completed":
        raise RuntimeError(f"Unexpected server state: {task.status}")
    write_json(marker, {**json.loads(path.read_text()), "status": str(task.status)})
    print(marker.read_text())


if __name__ == "__main__":
    main()
