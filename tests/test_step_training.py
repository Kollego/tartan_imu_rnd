"""Fixed budgets count real updates and snapshots preserve a shared prefix."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

import exps.baseline.train_steps as training
from src.inference import load_model
from src.data import PLATFORMS
from src.metrics import velocity_metrics
from src.model import build_model


class StepTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        rng = np.random.default_rng(17)
        # Ten training rows deliberately leave a two-row incomplete batch.
        x = rng.normal(size=(14, 200, 6)).astype(np.float32)
        x[:, 0, 0] = np.arange(14)
        x[10:, 1:, :] += 100  # Validation must not enter normalization.
        y = rng.normal(scale=0.2, size=(14, 3)).astype(np.float32)
        self.dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        self.dataset.x = x
        rows, assignments = [], []
        counts = {"car": 4, "dog": 3, "drone": 2, "human": 1}
        for partition in ("train", "val"):
            for platform, count in counts.items():
                trajectory = f"{platform}_{partition}"
                assignments.append({"traj_id": trajectory, "platform": platform,
                                    "fold": 1 if partition == "train" else 0})
                for window in range(count if partition == "train" else 1):
                    index = len(rows)
                    rows.append({"window_id": index, "platform": platform,
                                 "traj_id": trajectory, "win_idx": window,
                                 "vx": float(y[index, 0]), "vy": float(y[index, 1]),
                                 "vz": float(y[index, 2])})
        self.frame = pd.DataFrame(rows)
        self.assignments = pd.DataFrame(assignments)
        self.config = {
            "seed": 42, "hidden_sizes": [8], "batch_size": 4,
            "learning_rate": 0.001, "weight_decay": 0.0001,
            "device": "cpu", "platform": None, "drop_last": True,
            "max_steps": 7, "eval_every_steps": 2, "snapshot_steps": [3, 7],
        }

    def run_training(self, output, config):
        updates, batches = [], []
        adam = torch.optim.Adam

        class CountingAdam(adam):
            def step(self, *args, **kwargs):
                updates.append(1)
                return super().step(*args, **kwargs)

        def record_model(*args, **kwargs):
            model = build_model(*args, **kwargs)

            def record_batch(module, inputs):
                if module.training:
                    batches.append(inputs[0][:, 0, 0].tolist())

            model.register_forward_pre_hook(record_batch)
            return model

        with patch.object(training.torch.optim, "Adam", CountingAdam), \
                patch.object(training, "build_model", side_effect=record_model):
            result = training.train_fold(self.dataset, self.frame, self.assignments,
                                         0, config, output)
        return result, updates, batches

    def test_three_generalist_passes_and_both_specialist_budgets(self):
        # Reproduce all five real window counts without loading IMU arrays.
        validation_counts = [15500, 14248, 13852, 17016, 21315]
        rows, assignments = [], []
        for fold, count in enumerate(validation_counts):
            # Unequal domain sizes expose budgets computed after filtering.
            domain_counts = [count // 10, count // 5, count * 3 // 10]
            domain_counts.append(count - sum(domain_counts))
            for platform, windows in zip(PLATFORMS, domain_counts):
                trajectory = f"{platform}_{fold}"
                assignments.append({"traj_id": trajectory, "platform": platform, "fold": fold})
                rows.extend([{"traj_id": trajectory, "platform": platform}] * windows)
        frame, assignments = pd.DataFrame(rows), pd.DataFrame(assignments)
        config = {"reference_epochs": 3, "batch_size": 256,
                  "folds": list(range(5)), "platform": None}
        generalist = training.reference_budgets(frame, assignments, config)
        train_windows = np.array([66431, 67683, 68079, 64915, 60616])
        expected_steps = np.array([777, 792, 795, 759, 708])
        np.testing.assert_array_equal(generalist.generalist_train_windows, train_windows)
        np.testing.assert_array_equal(generalist.generalist_steps, expected_steps)
        np.testing.assert_array_equal(expected_steps, 3 * (train_windows // 256))
        # Floor applies to each pass, rather than once after tripling examples.
        self.assertTrue(np.any(expected_steps != 3 * train_windows // 256))
        shared = []
        for platform in PLATFORMS:
            specialist = training.reference_budgets(frame, assignments, {**config, "platform": platform})
            np.testing.assert_array_equal(specialist.generalist_train_windows, train_windows)
            np.testing.assert_array_equal(specialist.generalist_steps, expected_steps)
            shared.append(specialist.equal_total_steps.to_numpy())
            filtered = frame[frame.platform.eq(platform)]
            wrong = training.reference_budgets(filtered, assignments, {**config, "platform": platform})
            self.assertTrue(np.all(wrong.generalist_steps < expected_steps))
        shared = np.array(shared)
        np.testing.assert_array_equal(shared.sum(axis=0), expected_steps)
        self.assertTrue(np.all(shared.max(axis=0) - shared.min(axis=0) <= 1))
        np.testing.assert_array_equal(shared, [[195, 198, 199, 190, 177],
                                              [194, 198, 198, 190, 177],
                                              [194, 198, 199, 189, 177],
                                              [194, 198, 199, 190, 177]])

    def test_exact_updates_full_batches_new_shuffles_and_prefix_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            long_output, short_output = output / "long", output / "short"
            long_output.mkdir()
            short_output.mkdir()
            (snapshots, history, compute), updates, batches = self.run_training(long_output, self.config)
            self.assertEqual(len(updates), 7)
            self.assertEqual(len(batches), 7)
            self.assertTrue(all(len(batch) == 4 for batch in batches))
            self.assertTrue(all(0 <= identifier < 10 for batch in batches for identifier in batch))
            # Two batches per shuffled pass; each pass visits distinct rows.
            for first in (0, 2, 4):
                self.assertEqual(len(set(batches[first] + batches[first + 1])), 8)
            self.assertNotEqual(batches[:2], batches[2:4])
            self.assertEqual([row["step"] for row in history], [2, 3, 4, 6, 7])
            self.assertEqual(set(snapshots), {3, 7})
            self.assertEqual(compute["train_steps"], 7)
            self.assertEqual(compute["windows_seen"], 28)
            self.assertEqual(compute["validation_calls"], 5)
            for name in ("train_seconds", "train_cpu_seconds", "evaluation_seconds",
                         "evaluation_cpu_seconds", "normalization_seconds", "elapsed_seconds"):
                self.assertGreaterEqual(compute[name], 0)
            self.assertGreaterEqual(compute["total_wall_seconds"], compute["elapsed_seconds"])

            prefix_path = long_output / "fold_0/step_3.pt"
            prefix_bytes = prefix_path.read_bytes()
            prefix = torch.load(prefix_path, weights_only=True)
            short_config = {**self.config, "max_steps": 3, "snapshot_steps": [3]}
            (_, _, _), short_updates, short_batches = self.run_training(short_output, short_config)
            standalone = torch.load(short_output / "fold_0/step_3.pt", weights_only=True)
            self.assertEqual(len(short_updates), 3)
            self.assertEqual(batches[:3], short_batches)
            for name, tensor in prefix["state_dict"].items():
                torch.testing.assert_close(tensor, standalone["state_dict"][name], rtol=0, atol=0)
            self.assertEqual(prefix["step"], 3)
            self.assertEqual(prefix["compute"]["windows_seen"], 12)
            self.assertEqual(prefix_path.read_bytes(), prefix_bytes)

            model, final = load_model(long_output / "fold_0/step_7.pt")
            self.assertEqual(final["step"], 7)
            alias = torch.load(long_output / "fold_0/final.pt", weights_only=True)
            self.assertEqual(alias["step"], 7)
            for name, tensor in final["state_dict"].items():
                torch.testing.assert_close(tensor, alias["state_dict"][name], rtol=0, atol=0)
            expected_mean = self.dataset.x[:10].mean(axis=(0, 1), dtype=np.float64)
            np.testing.assert_allclose(model.mean.numpy(), expected_mean.astype(np.float32), rtol=0, atol=0)
            self.assertTrue(set(final["train_trajectories"]).isdisjoint(final["validation_trajectories"]))
            row, oof = snapshots[7]
            measured = velocity_metrics(oof, oof[["pred_vx", "pred_vy", "pred_vz"]].to_numpy())
            self.assertEqual(row["macro_ave"], measured["macro_ave"])
            self.assertEqual(final["cv_macro_ave"], measured["macro_ave"])
            disk_metrics = json.loads((long_output / "fold_0/step_7_metrics.json").read_text())
            self.assertEqual(disk_metrics["model"]["macro_ave"], measured["macro_ave"])

    def test_fixed_snapshots_are_not_replaced_by_diagnostic_best(self):
        calls = []

        def controlled_metric(frame, predictions):
            metrics = velocity_metrics(frame, predictions)
            # First call is zero baseline; force the first evaluation to be best.
            scores = [99., 0.1, 3., 4., 6., 7.]
            metrics["macro_ave"] = scores[len(calls)]
            calls.append(metrics["macro_ave"])
            return metrics

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(training, "velocity_metrics", side_effect=controlled_metric):
            (snapshots, _, _), updates, _ = self.run_training(Path(directory), self.config)
            self.assertEqual(len(updates), 7)
            best = torch.load(Path(directory) / "fold_0/best.pt", weights_only=True)
            final = torch.load(Path(directory) / "fold_0/step_7.pt", weights_only=True)
            self.assertEqual(best["step"], 2)
            self.assertEqual(best["cv_macro_ave"], 0.1)
            self.assertEqual(final["step"], 7)
            self.assertEqual(final["cv_macro_ave"], 7.)
            self.assertEqual(snapshots[3][0]["macro_ave"], 3.)
            self.assertEqual(snapshots[7][0]["macro_ave"], 7.)
            self.assertEqual(snapshots[7][0]["best_step_so_far"], 2)


if __name__ == "__main__":
    unittest.main()
