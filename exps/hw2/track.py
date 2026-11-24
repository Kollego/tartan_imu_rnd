"""Synchronize durable training records with ClearML after each five-fold run."""

import argparse
import json
import subprocess

from clearml import Task

from exps.baseline.train_steps import write_json
from exps.hw2.results import PRED, archive, expected_rows, is_archived, read_run, validate
from exps.hw2.study import FOLDS, HYPOTHESES, MARGIN, PATCH_SIZES, PROJECT, ROOT, VARIANTS, config_for
from src.data import PLATFORMS
from src.metrics import velocity_metrics
from src.model import build_model


def sync(variant, platform):
    directory = ROOT / "clearml" / variant / platform
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "completed.json"
    if marker.is_file():
        return
    frame, history = read_run(variant, platform)
    expected = expected_rows()
    frame = validate(frame, expected if platform == "all" else expected[expected.platform.eq(platform)])
    historical = is_archived(variant, platform)
    config = config_for(variant, platform)
    if historical:
        config = json.loads((archive(platform) / "config.json").read_text())
    encoder, mixing = variant.split("_")[:2]
    metrics = velocity_metrics(frame, frame[PRED].to_numpy())
    history = history.sort_values(["fold", "step"])
    end = history[history.step.eq(10000)]
    assert set(end.fold) == set(range(5)) and len(end) == 5
    task_info = directory / "task.json"
    if task_info.exists():
        task = Task.get_task(task_id=json.loads(task_info.read_text())["id"])
    else:
        task = Task.create(project_name=PROJECT, task_name=f"{variant} / {platform}", task_type="training",
                           detect_repository=False, add_task_init_call=False)
        write_json(task_info, {"id": task.id, "url": task.get_output_log_web_page()})
    task.mark_started(force=True)
    task.add_tags([encoder, mixing, platform, "HW1-import" if historical else "HW2"])
    hypothesis = ("Reference from HW1, fixed 10000-step checkpoint." if historical else HYPOTHESES[mixing])
    if variant in PATCH_SIZES:
        task.add_tags(["patch-size"])
        hypothesis = ("Longer patches aggregate more samples but give the GRU fewer time steps."
                      if PATCH_SIZES[variant] == 100 else
                      "Shorter patches give the GRU more time steps for within-second motion.")
    task.set_comment(hypothesis)
    source = (archive(platform) / "environment.json" if historical else ROOT / variant / platform / "manifest.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    model = build_model(config.get("model_name", "mlp"), **config.get("model_kwargs", {"hidden_sizes": [128, 64]}))
    task.set_parameters_as_dict({"Training": config, "Study": {"encoder": encoder, "mixing": mixing, "domain": platform,
                                 "hypothesis": hypothesis, "specialist_margin_mps": MARGIN,
                                 "parameters": sum(p.numel() for p in model.parameters()), "seed_by_fold": "42 + fold",
                                 "record_source": "HW1 archive" if historical else "HW2 training logs",
                                 "record_import_commit": commit}})
    logger = task.get_logger()
    for row in history.itertuples():
        logger.report_scalar("CV macro AVE", f"fold {row.fold}", row.cv_macro_ave, row.step)
        logger.report_scalar("Train MSE", f"fold {row.fold}", row.train_mse, row.step)
    final = {"CV mean": end.cv_macro_ave.mean(), "CV std": end.cv_macro_ave.std(), "OOF macro AVE": metrics["macro_ave"],
             "training seconds (summed)": end.train_seconds.sum()}
    for name, value in final.items():
        logger.report_scalar("Fixed step", name, float(value), 10000)
    for domain, value in metrics["per_platform"].items():
        logger.report_scalar("OOF AVE per platform", domain, value, 10000)
    for name, artifact in [("trajectory_folds", FOLDS), ("source_manifest", source), ("history", history),
                           ("oof_predictions", frame[["window_id", "fold", *PRED]])]:
        if not task.upload_artifact(name, artifact, wait_on_upload=True, retries=2):
            raise RuntimeError(f"Upload failed: {name}")
    task.flush(wait_for_uploads=True)
    task.mark_completed(ignore_errors=False)
    task.reload()
    if str(task.status) != "completed":
        raise RuntimeError(f"Unexpected server state {task.status}")
    write_json(marker, {**json.loads(task_info.read_text()), "status": str(task.status), "metrics": final})
    print(f"Synced {variant}/{platform}: {task.id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("variant", choices=(*VARIANTS, *PATCH_SIZES))
    parser.add_argument("platform", choices=("all", *PLATFORMS))
    args = parser.parse_args()
    sync(args.variant, args.platform)
