"""Train the single-change architecture, optimization, and seed comparisons."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import fcntl
from pathlib import Path
import subprocess
import sys
from time import sleep

from exps.hw2.study import ROOT, config_for
from exps.hw3.conv_ablation import GEOMETRIES, RUN as CONV_RUN, convolution_config
from exps.hw3.epoch_train import DOMAINS, VARIANT
from exps.hw3.fixed_generalist import train_fixed_fold


RUN = ROOT / "architecture_study"
BASELINE = "k10_s10"
CONTROL_STEPS = (20_000, 40_000, 60_000)
ABLATION_STEP = 40_000
SEEDS = (42, 142, 242)
FOLDS = tuple(config_for(VARIANT, "all")["folds"])
SETTINGS = {
    **{geometry: {} for geometry in GEOMETRIES},
    "filters32": {"model_kwargs": {"patch_filters": 32}},
    "conv_mlp": {"model_kwargs": {"encoder": "conv_mlp"}},
    "fixed_mix": {"model_kwargs": {"mixing": "fixed"}},
    "gate_ce_0.01": {"gate_domain_weight": 0.01},
    "gate_ce_0.1": {"gate_domain_weight": 0.1},
    "lr_0.0003": {"learning_rate": 0.0003},
    "lr_0.003": {"learning_rate": 0.003},
}
TABLES = {
    "conv1d": (*GEOMETRIES, "filters32"),
    "gru": (BASELINE, "conv_mlp"),
    "gate": (BASELINE, "fixed_mix", "gate_ce_0.01", "gate_ce_0.1"),
    "learning_rate": ("lr_0.0003", BASELINE, "lr_0.003"),
}
ARCHITECTURES = tuple(dict.fromkeys(g for axis in ("conv1d", "gru", "gate") for g in TABLES[axis]))
DISPATCH_ORDER = ("conv_mlp", BASELINE, "fixed_mix", "gate_ce_0.01", "gate_ce_0.1", "filters32",
                  *(geometry for geometry in GEOMETRIES if geometry != BASELINE), "lr_0.0003", "lr_0.003")


@dataclass(frozen=True)
class Job:
    variant: str
    domain: str
    fold: int
    seed: int = SEEDS[0]

    def directory(self, output):
        return output / f"seed_{self.seed}" / self.variant / self.domain

    def steps(self):
        if self.seed == SEEDS[0] and self.variant == BASELINE:
            return CONTROL_STEPS
        return (ABLATION_STEP,)


def study_config(variant, domain, seed):
    config = convolution_config(variant if variant in GEOMETRIES else BASELINE)
    config.update(platform=None if domain == "all" else domain, seed=seed)
    settings = SETTINGS[variant]
    config.update({key: value for key, value in settings.items() if key != "model_kwargs"})
    config["model_kwargs"].update(settings.get("model_kwargs", {}))
    return config


def historical(job):
    if job.seed == SEEDS[0] and job.domain == "all" and job.variant in GEOMETRIES:
        return CONV_RUN / job.variant / f"fold_{job.fold}"
    return None


def snapshot_directory(job, step, output):
    source = historical(job)
    root = job.directory(output)
    candidates = [source, root / f"fold_{job.fold}", *sorted(root.glob(f"attempt_{job.fold}_*/fold_{job.fold}"), reverse=True)]
    for directory in candidates:
        if (directory is not None and (directory / f"step_{step}_oof.csv").is_file()
                and (directory / f"step_{step}.pt").is_file()):
            return directory
    return None


def run_job(job, output):
    root = job.directory(output)
    config = study_config(job.variant, job.domain, job.seed)
    train_fixed_fold(job.fold, root, config, Path(__file__), job.steps(),
                     {"variant": job.variant, "domain": job.domain, "seed_base": job.seed})


def initial_jobs():
    for variant in DISPATCH_ORDER:
        for domain in DOMAINS:
            for fold in FOLDS:
                job = Job(variant, domain, fold)
                source = historical(job)
                reusable = (source is not None and (source / f"step_{ABLATION_STEP}.pt").is_file()
                            and (source / f"step_{ABLATION_STEP}_oof.csv").is_file())
                if not reusable or CONTROL_STEPS[-1] in job.steps():
                    yield job


def seed_jobs(winner):
    for seed in SEEDS[1:]:
        for domain in DOMAINS:
            for fold in FOLDS:
                yield Job(winner, domain, fold, seed)


def run_batch(jobs, output, workers):
    def worker(job):
        root = job.directory(output)
        root.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "exps.hw3.architecture_study", "fold", job.variant, job.domain,
               str(job.fold), "--seed", str(job.seed), "--output", str(output)]
        with (root / f"fold_{job.fold}.log").open("ab") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"Started {job}, PID {process.pid}", flush=True)
            return job, process.wait()

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(pending):
            job, status = future.result()
            print(f"Finished {job}, status {status}", flush=True)
            if status:
                failures.append(job)
    if failures:
        raise RuntimeError(f"Failed training jobs: {failures}")


def scoring_command(output, scorer):
    return [sys.executable, "-m", "exps.hw3.architecture_study_score", "--output", str(output),
            "--scorer", str(scorer)]


def refresh_tables(output, scorer):
    """Refresh the study output when another full CV result is available."""
    last_ready = None
    lock_path = output / "study.lock"
    if not lock_path.is_file():
        raise ValueError(f"No study has started in {output}")
    while True:
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                return
        conditions = [(variant, SEEDS[0], ABLATION_STEP) for variant in SETTINGS]
        conditions.extend((BASELINE, SEEDS[0], step) for step in CONTROL_STEPS if step != ABLATION_STEP)
        winner_path = output / "winner.txt"
        if winner_path.is_file():
            winner = winner_path.read_text().strip()
            conditions.extend((winner, seed, ABLATION_STEP) for seed in SEEDS[1:])
        ready = frozenset((variant, seed, step, domains) for variant, seed, step in conditions
                          for domains in (DOMAINS[:1], DOMAINS[1:])
                          if all(snapshot_directory(Job(variant, domain, fold, seed), step, output) is not None
                                 for domain in domains for fold in FOLDS))
        if ready != last_ready:
            subprocess.run([*scoring_command(output, scorer), "--partial"], check=True)
            last_ready = ready
        sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fold")
    train.add_argument("variant", choices=SETTINGS)
    train.add_argument("domain", choices=DOMAINS)
    train.add_argument("fold", type=int, choices=FOLDS)
    train.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    train.add_argument("--output", type=Path, default=RUN)
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, default=RUN)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--scorer", type=Path, required=True)
    refresh = commands.add_parser("refresh")
    refresh.add_argument("--output", type=Path, default=RUN)
    refresh.add_argument("--scorer", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fold":
        run_job(Job(args.variant, args.domain, args.fold, args.seed), args.output)
        return
    if args.command == "refresh":
        refresh_tables(args.output, args.scorer)
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_batch(initial_jobs(), args.output, args.workers)
        scoring = scoring_command(args.output, args.scorer)
        subprocess.run(scoring, check=True)
        winner = (args.output / "winner.txt").read_text().strip()
        run_batch(seed_jobs(winner), args.output, args.workers)
        subprocess.run([*scoring, "--require-seeds"], check=True)
    print("Completed the architecture, optimization, and seed study.", flush=True)


if __name__ == "__main__":
    main()
