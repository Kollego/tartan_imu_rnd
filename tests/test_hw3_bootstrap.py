import unittest

import numpy as np
import pandas as pd

from exps.hw3.bootstrap import TRAJECTORY, interval, macro_mean, sufficiently_sampled, window_rate_intervals


class TrajectoryBootstrapTest(unittest.TestCase):
    def test_macro_weights_folds_and_then_platforms_equally(self):
        rows = pd.DataFrame([
            (0, "dog", "d1", 1.0), (0, "dog", "d2", 1.0),
            (0, "car", "c1", 3.0), (0, "car", "c2", 3.0), (0, "car", "c3", 3.0),
            (1, "dog", "d3", 9.0), (1, "dog", "d4", 9.0),
        ], columns=[*TRAJECTORY, "ave"]).set_index(TRAJECTORY)

        np.testing.assert_allclose(macro_mean(rows), 5.5)

    def test_perturbed_and_clean_share_trajectory_draws(self):
        rows = pd.DataFrame([
            (0, "dog", "d1", 1.0, 1.25),
            (0, "dog", "d2", 3.0, 3.25),
            (0, "dog", "d3", 6.0, 6.25),
        ], columns=[*TRAJECTORY, "clean", "changed"]).set_index(TRAJECTORY)

        samples = macro_mean(rows)
        np.testing.assert_array_equal(samples, macro_mean(rows.iloc[::-1]))
        np.testing.assert_allclose(interval(samples[:, 1] - samples[:, 0]), [0.25, 0.25])
        self.assertGreater(np.ptp(samples[:, 0]), 0.0)

    def test_rate_weights_windows_but_resamples_trajectories(self):
        trajectories = pd.MultiIndex.from_tuples([(0, "dog", "d1"), (0, "car", "c1")], names=TRAJECTORY)
        columns = pd.MultiIndex.from_tuples([("error", "match"), ("error", "eligible")])
        rows = pd.DataFrame([[0, 2], [9, 10]], index=trajectories, columns=columns)

        self.assertEqual(window_rate_intervals(rows)["error"], [0.75, 0.75])

    def test_slice_with_single_trajectory_in_one_stratum_has_no_ci(self):
        trajectories = pd.MultiIndex.from_tuples(
            [(0, "dog", f"d{i}") for i in range(10)] + [(1, "dog", "d10")], names=TRAJECTORY)
        rows = pd.DataFrame({"ave": np.ones(len(trajectories))}, index=trajectories)

        self.assertFalse(sufficiently_sampled(rows))
