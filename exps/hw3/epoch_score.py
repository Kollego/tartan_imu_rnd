"""Score the matched-epoch and matched-step HW2 out-of-fold predictions."""

import argparse
from pathlib import Path

import pandas as pd

from exps.baseline.train_steps import write_json
from exps.hw2.results import PRED, expected_rows, validate
from exps.hw2.study import config_for
from exps.hw3.audit_score import SCORER_COMMIT, SCORER_SHA256, attach_pose, load_scorer, summarize
from exps.hw3.bootstrap import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from exps.hw3.budget_score import METRICS, metric_rows
from exps.hw3.epoch_train import DOMAINS, REGIMES, RUN, VARIANT, budget_plan


def slices(frame):
    yield "all", frame
    yield from frame.groupby(level="platform", sort=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    scorer = load_scorer(args.scorer)
    rows = expected_rows()
    config = config_for(VARIANT, "all")
    saved_plan = pd.concat([pd.read_csv(args.output / domain / "budgets.csv") for domain in DOMAINS])
    step_budget = "generalist_steps" in saved_plan
    budget_column = "generalist_steps" if step_budget else "epoch_budget"
    budgets = sorted(saved_plan[budget_column].unique().tolist())
    plan_args = {"steps" if step_budget else "epochs": budgets}
    plan = budget_plan(rows, config, **plan_args)
    keys = [budget_column, "regime", "domain", "fold"]
    pd.testing.assert_frame_equal(plan.set_index(keys).sort_index(), saved_plan.set_index(keys).sort_index())
    poses = attach_pose(rows, "train", (100, 199))
    summary, comparisons, folds, budget_comparisons = [], [], [], []
    baseline = {}
    for budget in budgets:
        budget_label = {budget_column: budget}
        scores = {}
        for regime in REGIMES:
            configs = plan[plan[budget_column].eq(budget) & plan.regime.eq(regime)]
            files = (args.output / row.domain / f"fold_{row.fold}/step_{row.optimizer_steps}_oof.csv"
                     for row in configs.itertuples())
            predictions = validate(pd.concat([pd.read_csv(path) for path in files], ignore_index=True), rows)
            scored = poses.merge(predictions[["window_id", *PRED]], on="window_id", validate="one_to_one")
            _, trajectories = summarize(scorer, scored)
            scores[regime] = trajectories[METRICS]
            for platform, frame in slices(scores[regime]):
                summary.extend(dict(**budget_label, regime=regime, platform=platform, **row)
                               for row in metric_rows(frame))
                by_fold = frame.groupby(level=["fold", "platform"]).mean().groupby(level="fold").mean()
                folds.append(by_fold.reset_index().assign(**budget_label, regime=regime, platform=platform))
        generalist, same_epochs, same_steps = REGIMES
        for left, right in ((same_epochs, generalist), (same_steps, generalist), (same_steps, same_epochs)):
            pd.testing.assert_index_equal(scores[left].index, scores[right].index)
            for platform, delta in slices(scores[left] - scores[right]):
                comparisons.extend(dict(**budget_label, regime=left, reference=right, platform=platform, **row)
                                   for row in metric_rows(delta))
        if budget == budgets[0]:
            baseline = scores
        elif step_budget:
            for regime, scored in scores.items():
                pd.testing.assert_index_equal(scored.index, baseline[regime].index)
                for platform, delta in slices(scored - baseline[regime]):
                    budget_comparisons.extend(dict(**budget_label, reference_steps=budgets[0], regime=regime,
                                                   platform=platform, **row) for row in metric_rows(delta))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output / "comparisons.csv", index=False)
    if budget_comparisons:
        pd.DataFrame(budget_comparisons).to_csv(args.output / "budget_comparisons.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(args.output / "fold_metrics.csv", index=False)
    plan.to_csv(args.output / "budgets.csv", index=False)
    write_json(args.output / "evaluation.json", dict(
        variant=VARIANT, **plan_args, split="TartanIMU train, out of fold", windows=len(rows),
        trajectories=rows.traj_id.nunique(), split_seed=config["seed"], train_seeds=sorted(plan.train_seed.unique().tolist()),
        batch_size=config["batch_size"], drop_last=config["drop_last"], bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, scorer_commit=SCORER_COMMIT, scorer_sha256=SCORER_SHA256,
        quaternion_offset=100, position_offset=199,
    ))
    print(summary[summary.platform.eq("all")].pivot(index=[budget_column, "metric"],
                                                     columns="regime", values="value").round(5).to_string())


if __name__ == "__main__":
    main()
