"""Small fixtures covering split leakage, sample alignment and metric weighting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import IMUDataset, PLATFORMS, compute_normalization, make_folds, read_index
from src.metrics import velocity_metrics


class DataTests(unittest.TestCase):
    def test_folds_keep_trajectories_together_and_are_reproducible(self):
        rows = []
        for platform in PLATFORMS:
            for trajectory in range(6):
                for window in range(2):
                    rows.append({"window_id": len(rows), "platform": platform,
                                 "traj_id": f"{platform}_{trajectory}", "win_idx": window})
        frame = pd.DataFrame(rows)
        folds = make_folds(frame)
        pd.testing.assert_frame_equal(folds, make_folds(frame.sample(frac=1, random_state=9)))
        self.assertEqual(len(folds), 24)
        self.assertEqual(set(folds.fold), set(range(5)))
        joined = frame.merge(folds, on=["traj_id", "platform"], validate="many_to_one")
        for fold in range(5):
            train = joined.loc[joined.fold != fold]
            validation = joined.loc[joined.fold == fold]
            self.assertTrue(set(train.traj_id).isdisjoint(validation.traj_id))
            self.assertEqual(set(validation.platform), set(PLATFORMS))

    def test_index_target_join_and_requested_npz_window_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index").mkdir()
            (root / "train" / "car").mkdir(parents=True)
            np.savez(root / "train" / "car" / "drive.npz",
                     imu=np.concatenate([np.full((200, 6), 5), np.full((200, 6), 17)]))
            pd.DataFrame({"window_id": [10, 11], "platform": ["car", "car"],
                          "traj_id": ["drive", "drive"], "win_idx": [0, 1]}).to_csv(root / "index" / "train_windows.csv", index=False)
            pd.DataFrame({"window_id": [11, 10], "vx": [2, 1], "vy": [0, 0], "vz": [0, 0]}).to_csv(root / "index" / "train_targets.csv", index=False)
            frame = read_index(root, "train")
            np.testing.assert_array_equal(frame.vx, [1, 2])
            dataset = IMUDataset(root, "train", frame.iloc[[1, 0]])
            self.assertEqual(dataset.x.shape, (2, 200, 6))
            self.assertEqual(dataset.x.dtype, np.float32)
            np.testing.assert_array_equal(dataset.x[:, 0, 0], [17, 5])
            x, y = dataset[0]
            self.assertIsInstance(x, torch.Tensor)
            self.assertEqual(x.shape, (200, 6))
            self.assertEqual(x.dtype, torch.float32)
            torch.testing.assert_close(y, torch.tensor([2., 0., 0.]))
            # Selecting only window 1 must not preload window 0 into .x.
            selected = IMUDataset(root, "train", frame.iloc[[1]])
            self.assertEqual(len(selected), 1)
            self.assertTrue(np.all(selected.x == 17))

    def test_test_split_needs_neither_targets_nor_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index").mkdir()
            (root / "test").mkdir()
            np.savez(root / "test" / "hidden.npz", imu=np.full((200, 6), 9.81))
            pd.DataFrame({"window_id": [4], "traj_id": ["hidden"], "win_idx": [0]}).to_csv(root / "index" / "test_windows.csv", index=False)
            frame = read_index(root, "test")
            self.assertNotIn("platform", frame)
            self.assertNotIn("vx", frame)
            dataset = IMUDataset(root, "test", frame)
            x, y = dataset[0]
            self.assertIsNone(dataset.y)
            self.assertEqual(y.shape, (0,))
            self.assertEqual(x.shape, (200, 6))
            self.assertAlmostEqual(float(x[0, 0]), 9.81, places=5)

    def test_normalization_uses_only_the_supplied_training_array(self):
        training = np.stack([np.zeros((200, 6)), np.full((200, 6), 2)]).astype(np.float32)
        validation = np.full((1, 200, 6), 1000, dtype=np.float32)
        before = training.copy()
        mean, std = compute_normalization(training)
        np.testing.assert_allclose(mean, np.ones(6))
        np.testing.assert_allclose(std, np.ones(6))
        np.testing.assert_array_equal(training, before)
        self.assertEqual(mean.dtype, np.float64)
        self.assertTrue(np.all((validation - mean) / std > 900))
        _, constant_std = compute_normalization(np.zeros((1, 200, 6)))
        self.assertTrue(np.isfinite(constant_std).all())
        self.assertTrue((constant_std > 0).all())
        with self.assertRaises(ValueError):
            compute_normalization(np.zeros((200, 6)))
        training[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_normalization(training)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"window_id": range(4), "traj_id": ["a", "a", "b", "c"],
                                   "platform": ["car", "car", "car", "dog"],
                                   "vx": [0.] * 4, "vy": [0.] * 4, "vz": [0.] * 4})
        self.predictions = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 4.], [6., 8., 0.]])

    def test_hierarchical_ave_gives_each_trajectory_and_platform_equal_weight(self):
        metrics = velocity_metrics(self.frame, self.predictions)
        self.assertEqual(metrics["per_platform"], {"car": 2., "dog": 10.})
        self.assertEqual(metrics["macro_ave"], 6.)
        self.assertEqual(metrics["window_ave"], 3.5)
        self.assertEqual(len(metrics["per_trajectory"]), 3)
        # Specialist scoring reports only the represented platform.
        specialist = velocity_metrics(self.frame.iloc[:3], self.predictions[:3])
        self.assertEqual(specialist["per_platform"], {"car": 2.})
        self.assertEqual(specialist["macro_ave"], 2.)

    def test_metrics_reject_wrong_shape_nonfinite_values_and_duplicate_ids(self):
        for predictions in [self.predictions[:2], np.full((4, 3), np.nan), np.full((4, 3), np.inf)]:
            with self.subTest(shape=predictions.shape), self.assertRaises(ValueError):
                velocity_metrics(self.frame, predictions)
        duplicate = self.frame.copy()
        duplicate.loc[1, "window_id"] = duplicate.loc[0, "window_id"]
        with self.assertRaises(ValueError):
            velocity_metrics(duplicate, self.predictions)


if __name__ == "__main__":
    unittest.main()
