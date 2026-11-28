"""Trajectory bootstrap for the fixed HW2 validation folds."""

import numpy as np


BOOTSTRAP_SEED = 20261128
BOOTSTRAP_REPLICATES = 10_000
TRAJECTORY = ["fold", "platform", "traj_id"]


def resampled_totals(trajectories):
    """Draw whole trajectories with replacement within each fold and platform."""
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    for (fold, _), group in trajectories.sort_index().groupby(level=["fold", "platform"]):
        n = len(group)
        # Counts are bootstrap multiplicities. Reuse them for every column.
        counts = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP_REPLICATES)
        yield fold, n, counts @ group.to_numpy(dtype=np.float64)


def macro_mean(trajectories):
    per_fold = {}
    for fold, n, totals in resampled_totals(trajectories):
        per_fold.setdefault(fold, []).append(totals / n)
    return np.mean([np.mean(platforms, axis=0) for platforms in per_fold.values()], axis=0)


def interval(samples):
    """Return the marginal percentile 95% interval for one statistic."""
    return np.quantile(samples, [0.025, 0.975]).tolist()


def sufficiently_sampled(trajectories):
    # Singletons cannot express between-trajectory uncertainty in their stratum.
    return len(trajectories) >= 10 and trajectories.groupby(level=["fold", "platform"]).size().min() >= 2


def window_rate_intervals(trajectories):
    """Bootstrap ratios of summed counts, with common draws for all ratios."""
    totals = np.sum([sample for _, _, sample in resampled_totals(trajectories)], axis=0)
    result = {}
    for name in trajectories.columns.get_level_values(0).unique():
        numerator = totals[:, trajectories.columns.get_loc((name, "match"))]
        denominator = totals[:, trajectories.columns.get_loc((name, "eligible"))]
        if np.all(denominator > 0):
            result[name] = interval(numerator / denominator)
        else:
            result[name] = None
    return result
