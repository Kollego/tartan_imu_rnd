"""Build separate ablation tables, paired role comparisons, and seed statistics."""

import argparse
import fcntl
from functools import lru_cache
from html import escape
from pathlib import Path

import pandas as pd
import torch

from exps.baseline.train_steps import write_json
from exps.hw2.results import PRED, expected_rows, validate
from exps.hw3.architecture_study import (ABLATION_STEP, ARCHITECTURES, BASELINE, CONTROL_STEPS, FOLDS, RUN,
                                        SEEDS, TABLES, Job, snapshot_directory, study_config)
from exps.hw3.audit_score import SCORER_COMMIT, SCORER_SHA256, attach_pose, load_scorer, summarize
from exps.hw3.bootstrap import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from exps.hw3.budget_score import METRICS, metric_rows
from exps.hw3.conv_ablation import GEOMETRIES
from exps.hw3.conv_ablation_score import comparisons_for
from exps.hw3.epoch_score import slices
from src.data import PLATFORMS
from src.model import build_model


ROLES = {"generalist": ("all",), "specialists": PLATFORMS}
TITLES = {"conv1d": "Conv1d", "gru": "GRU и MLP", "gate": "Gate и вес лосса платформы",
          "learning_rate": "Learning rate", "steps": "Бюджет обучения"}


def read_fold(path, job, step, rows):
    config = study_config(job.variant, job.domain, job.seed)
    domain = rows if job.domain == "all" else rows[rows.platform.eq(job.domain)]
    validation = domain[domain.fold.eq(job.fold)]
    state = torch.load(path / f"step_{step}.pt", map_location="cpu", weights_only=True)
    fields = ("batch_size", "drop_last", "learning_rate", "weight_decay", "platform", "gate_domain_weight", "device")
    if ((state["fold"], state["step"], state["seed"]) != (job.fold, step, job.seed + job.fold)
            or state["model_name"] != config["model_name"] or state["model_kwargs"] != config["model_kwargs"]
            or any(state["config"].get(key, 0) != config.get(key, 0) for key in fields)
            or set(state["validation_trajectories"]) != set(validation.traj_id)
            or set(state["train_trajectories"]) != set(domain.loc[domain.fold.ne(job.fold), "traj_id"])):
        raise ValueError(f"Snapshot disagrees with the study protocol: {path}, step {step}")
    frame = validate(pd.read_csv(path / f"step_{step}_oof.csv"), validation)
    return frame[["window_id", *PRED]]


def describe(variant):
    config = study_config(variant, "all", SEEDS[0])
    model = config["model_kwargs"]
    network = build_model(config["model_name"], **model)
    return dict(variant=variant, kernel_size=model["patch_size"], stride=model["conv_stride"],
                filters=model["patch_filters"], temporal="MLP" if model["encoder"] == "conv_mlp" else "GRU",
                mixing=model["mixing"], gate_lambda=config.get("gate_domain_weight", 0),
                learning_rate=config["learning_rate"], parameters=sum(p.numel() for p in network.parameters()))


def describe_pair(generalist, specialists):
    pd.testing.assert_index_equal(generalist.index, specialists.index)
    gen = dict(slices(generalist))
    spec = dict(slices(specialists))
    for platform, delta in slices(specialists - generalist):
        general_values = {row["metric"]: row["value"] for row in metric_rows(gen[platform])}
        special_values = {row["metric"]: row["value"] for row in metric_rows(spec[platform])}
        for row in metric_rows(delta):
            metric = row["metric"]
            yield dict(platform=platform, generalist=general_values[metric], specialists=special_values[metric],
                       delta_specialists_minus_generalist=row.pop("value"), **row)


def paired_deltas(axis, contrast, role, variant, reference, step, reference_step, trajectories, baseline):
    if baseline is None or trajectories is None:
        return
    pd.testing.assert_index_equal(trajectories.index, baseline.index)
    for platform, delta in slices(trajectories - baseline):
        for row in metric_rows(delta):
            yield dict(axis=axis, contrast=contrast, variant=variant, reference=reference, step=step,
                       reference_step=reference_step, role=role, platform=platform, **row)


def save_html(output, tables, winner, comparison, seed, seed_summary):
    sheet = []
    dimensions = {"conv1d": ["kernel_size", "stride", "filters"], "gru": ["temporal", "parameters"],
                  "gate": ["mixing", "gate_lambda"], "learning_rate": ["learning_rate"], "steps": ["step"]}
    labels = {"kernel_size": "Ядро", "stride": "Stride", "filters": "Фильтры", "temporal": "После Conv1d",
              "parameters": "Параметры", "mixing": "Вес ветвей", "gate_lambda": "λ", "learning_rate": "LR",
              "step": "Шаги", "generalist_folds": "Фолды генералиста", "specialists_folds": "Фолды специалистов",
              "generalist_ave": "AVE генералиста", "specialists_ave": "AVE специалистов",
              "generalist_score": "Score генералиста", "specialists_score": "Score специалистов"}
    for name, frame in tables.items():
        display = dimensions[name] + ["generalist_folds", "specialists_folds", "generalist_ave",
                                      "specialists_ave", "generalist_score", "specialists_score"]
        view = frame[display].copy()
        for role, count in (("generalist", 5), ("specialists", 20)):
            column = f"{role}_folds"
            view[column] = view[column].map(lambda value: f"{value}/{count}")
        if name == "gate":
            view["mixing"] = view.mixing.map({"gate": "Gate по входу", "fixed": "Общие веса"})
        link = f'<a href="{name}.csv">CSV с AVE, ATE20 и Score</a>'
        sheet.append(f'<section><h2>{TITLES[name]}</h2><div class="table">'
                     + view.rename(columns=labels).round(5).to_html(index=False, na_rep="—", border=0)
                     + f'</div><p>{link}</p></section>')
    head = (f'<p>Выбранная конфигурация <code>{escape(winner)}</code></p>' if winner else
            '<p>Выбор модели будет сделан после оценки всех архитектур на одинаковых фолдах.</p>')
    comparison = (comparison.round(5).to_html(index=False, border=0) if not comparison.empty else '')
    seeds = '<p>Проверка разных seed запустится после выбора архитектуры.</p>'
    if not seed.empty:
        by_seed = seed.query('platform == "all"').pivot(index=["seed_base", "role"], columns="metric", values="value")
        seeds = by_seed.reset_index().round(5).to_html(index=False, border=0)
        seeds += seed_summary.query('platform == "all"').round(5).to_html(index=False, border=0, na_rep="—")
    html = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Абляции TartanIMU</title><style>
body{{margin:0;font:15px/1.5 system-ui;color:#172a27;background:#f6f8f4}}main{{max-width:1240px;margin:auto;padding:32px 24px}}
h1{{font-size:2rem}}section{{margin:28px 0}}a{{color:#086b5d}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:white}}
th,td{{padding:8px 10px;border-bottom:1px solid #dce4da;text-align:right;white-space:nowrap}}th{{background:#e6efea}}
tr>:first-child{{text-align:left}}@media(max-width:600px){{main{{padding:18px 12px}}}}</style>
<main><h1>Абляции Conv1d + GRU + gate</h1><p>Пять CV-фолдов по целым траекториям TartanIMU. Score и AVE — меньше лучше.
У генералиста 5 чекпойнтов на условие. Группа специалистов включает четыре платформы по 5 чекпойнтов.
Метрики появляются после завершения всех фолдов соответствующей группы.</p>{''.join(sheet)}
<section id="comparison"><h2>Генералист и специалисты на одной архитектуре</h2>{head}<div class="table">{comparison}</div></section>
<section id="seeds"><h2>Чувствительность к seed обучения</h2><p>Разбиение, архитектура и бюджет фиксированы. В таблице показан разброс по полным CV-прогонам.</p>
<div class="table">{seeds}</div><p><a href="seed-results.csv">Метрики по seed</a> · <a href="seed-folds.csv">Метрики по фолдам</a></p></section>
<p><a href="evaluation.json">Протокол</a> · <a href="ablation-deltas.csv">Парные 95% ДИ</a></p></main></html>'''
    target = output / "index.html"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--report-dir", type=Path, help="Write the tables separately from the checkpoints")
    parser.add_argument("--scorer", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--partial", action="store_true")
    mode.add_argument("--require-seeds", action="store_true")
    args = parser.parse_args()
    report_dir = args.report_dir or args.output
    report_dir.mkdir(parents=True, exist_ok=True)
    with (args.output / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        save_results(args)


def save_results(args):
    rows = expected_rows()
    poses = attach_pose(rows, "train", (100, 199))
    scorer = load_scorer(args.scorer)
    output = args.report_dir or args.output

    @lru_cache(maxsize=None)
    def evaluation(variant, role, step, seed):
        jobs = [Job(variant, domain, fold, seed) for domain in ROLES[role] for fold in FOLDS]
        paths = [snapshot_directory(job, step, args.output) for job in jobs]
        ready = sum(path is not None for path in paths)
        if ready != len(jobs):
            return ready, None, None
        predictions = pd.concat([read_fold(path, job, step, rows) for job, path in zip(jobs, paths)], ignore_index=True)
        scored = poses.merge(predictions, on="window_id", validate="one_to_one")
        info, trajectories = summarize(scorer, scored)
        return ready, info, trajectories[METRICS]

    tables, all_deltas = {}, []
    designs = {axis: [(variant, ABLATION_STEP) for variant in variants] for axis, variants in TABLES.items()}
    designs["steps"] = [(BASELINE, step) for step in CONTROL_STEPS]
    missing = []
    for axis, configs in designs.items():
        table = []
        for variant, step in configs:
            record = dict(**describe(variant), step=step)
            for role in ROLES:
                ready, info, trajectories = evaluation(variant, role, step, SEEDS[0])
                record[f"{role}_folds"] = ready
                record.update({f"{role}_{metric}": None if info is None else info["macro"][metric] for metric in METRICS})
                if info is None:
                    missing.append((variant, role, step))
                _, _, baseline = evaluation(BASELINE, role, ABLATION_STEP, SEEDS[0])
                if (variant, step) == (BASELINE, ABLATION_STEP):
                    continue
                all_deltas.extend(paired_deltas(axis, "baseline", role, variant, BASELINE, step, ABLATION_STEP,
                                               trajectories, baseline))
            table.append(record)
        tables[axis] = pd.DataFrame(table)
        tables[axis].to_csv(output / f"{axis}.csv", index=False)
    if missing and not args.partial:
        raise ValueError(f"Some primary CV evaluations are incomplete: {missing}")
    for contrast, reference, variant in comparisons_for(tuple(GEOMETRIES)):
        if contrast == "vs_control":
            continue
        for role in ROLES:
            _, _, trajectories = evaluation(variant, role, ABLATION_STEP, SEEDS[0])
            _, _, baseline = evaluation(reference, role, ABLATION_STEP, SEEDS[0])
            all_deltas.extend(paired_deltas("conv1d", contrast, role, variant, reference, ABLATION_STEP, ABLATION_STEP,
                                           trajectories, baseline))
    columns = ["axis", "contrast", "variant", "reference", "step", "reference_step", "role", "platform", "metric",
               "value", "ci95_low", "ci95_high", "trajectories"]
    pd.DataFrame(all_deltas, columns=columns).to_csv(output / "ablation-deltas.csv", index=False)

    ranking = [(variant, evaluation(variant, "generalist", ABLATION_STEP, SEEDS[0])[1]) for variant in ARCHITECTURES]
    winner = (min(ranking, key=lambda row: (row[1]["macro"]["score"], row[0]))[0]
              if all(info is not None for _, info in ranking) else None)
    paired, seed_rows, seed_folds = [], [], []
    if winner is not None:
        generalist = evaluation(winner, "generalist", ABLATION_STEP, SEEDS[0])[2]
        specialists = evaluation(winner, "specialists", ABLATION_STEP, SEEDS[0])[2]
        if specialists is not None:
            paired = list(describe_pair(generalist, specialists))
        for seed in SEEDS:
            for role in ROLES:
                _, info, _ = evaluation(winner, role, ABLATION_STEP, seed)
                if info is None:
                    if args.require_seeds:
                        raise ValueError(f"Missing seed={seed} for {role}")
                    continue
                for platform, metrics in {"all": info["macro"], **info["platforms"]}.items():
                    seed_rows.extend(dict(seed_base=seed, role=role, platform=platform, metric=metric, value=value)
                                     for metric, value in metrics.items())
                seed_folds.extend(dict(seed_base=seed, role=role, train_seed=seed + row["fold"], **row)
                                  for row in info["folds"])
        if not args.partial:
            (output / "winner.txt").write_text(winner + "\n")
    paired = pd.DataFrame(paired, columns=["platform", "metric", "generalist", "specialists",
                                          "delta_specialists_minus_generalist", "ci95_low", "ci95_high", "trajectories"])
    seed = pd.DataFrame(seed_rows, columns=["seed_base", "role", "platform", "metric", "value"])
    seed_summary = seed.groupby(["role", "platform", "metric"], as_index=False).value.agg(
        seeds="size", mean="mean", std="std", min="min", max="max")
    paired.to_csv(output / "generalist-vs-specialists.csv", index=False)
    seed.to_csv(output / "seed-results.csv", index=False)
    seed_summary.to_csv(output / "seed-summary.csv", index=False)
    pd.DataFrame(seed_folds, columns=["seed_base", "role", "fold", "train_seed", *METRICS]).to_csv(
        output / "seed-folds.csv", index=False)
    complete_seeds = {role: sorted(seed.loc[seed.role.eq(role), "seed_base"].unique().tolist()) for role in ROLES}
    write_json(output / "evaluation.json", dict(
        baseline=describe(BASELINE), ablation_step=ABLATION_STEP, budget_steps=CONTROL_STEPS,
        windows=len(rows), trajectories=rows.traj_id.nunique(), split_seed=42, split="TartanIMU train OOF",
        planned_seed_bases=SEEDS, train_seed_rule="seed_base + fold", missing_primary=list(dict.fromkeys(missing)),
        seed_bases_complete=complete_seeds, selected_architecture=winner,
        selection="minimum generalist CV Score among Conv1d, temporal and gate ablations at 40000 updates",
        scorer_sha256=SCORER_SHA256, scorer_commit=SCORER_COMMIT, quaternion_offset=100, position_offset=199,
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES,
    ))
    save_html(output, tables, winner, paired, seed, seed_summary)
    print(f"Saved tables to {output.resolve()}, {len(set(missing))} primary evaluations pending, seeds {complete_seeds}")


if __name__ == "__main__":
    main()
